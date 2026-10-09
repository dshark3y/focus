"""Tests for CLI overrides, API-key handling, synth pacing and WAV output."""

import asyncio
import time

import numpy as np
from click.testing import CliRunner
from scipy.io import wavfile

from focus.audio.output import FileAudioOutput
from focus.cli import apply_overrides, main
from focus.generation import lyria_client
from focus.generation.lyria_client import EnhancedSynthClient, LyriaConfig, resolve_api_key
from focus.profiles import get_profile


class TestApplyOverrides:
    def test_no_overrides_returns_profile_unchanged(self):
        p = get_profile("deep-work")
        assert apply_overrides(p) is p

    def test_depth_zero_is_applied(self):
        p = apply_overrides(get_profile("deep-work"), depth=0.0)
        assert p.modulation_depth == 0.0

    def test_override_keeps_intro_and_outro_prompts(self):
        base = get_profile("deep-work")
        p = apply_overrides(base, frequency=16.0)
        assert p.modulation_freq == 16.0
        assert p.intro_prompt == base.intro_prompt
        assert p.outro_prompt == base.outro_prompt

    def test_band_zero_means_full_spectrum(self):
        p = apply_overrides(get_profile("deep-work"), band=0)
        assert p.modulation_band_hz is None


class TestCliValidation:
    def test_depth_out_of_range_is_rejected(self):
        result = CliRunner().invoke(main, ["start", "--mock", "--depth", "1.5"])
        assert result.exit_code == 2
        assert "1.5" in result.output

    def test_missing_api_key_fails_fast(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        result = CliRunner().invoke(main, ["start"])
        assert result.exit_code == 1
        assert "GEMINI_API_KEY" in result.output


class TestResolveApiKey:
    def test_prefers_google_api_key(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "g")
        monkeypatch.setenv("GEMINI_API_KEY", "m")
        assert resolve_api_key() == "g"

    def test_falls_back_to_gemini_api_key(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "m")
        assert resolve_api_key() == "m"


class TestSynthPacing:
    def test_synth_does_not_run_far_ahead_of_real_time(self, monkeypatch):
        monkeypatch.setattr(lyria_client, "SYNTH_LEAD_SECONDS", 0.2)

        async def produce(seconds: float) -> float:
            synth = EnhancedSynthClient(LyriaConfig(prompt="x"))
            await synth.connect()
            produced = 0.0
            start = time.monotonic()
            async for chunk in synth.generate_stream():
                produced += len(chunk) / synth.config.sample_rate
                if produced >= seconds:
                    break
            await synth.stop()
            return time.monotonic() - start

        wall = asyncio.run(produce(1.0))
        # 1s of audio with a 0.2s lead must take roughly 0.8s, not ~0.002s
        assert wall >= 0.6


class TestFileAudioOutput:
    def test_streams_wav_and_clips_out_of_range_peaks(self, tmp_path):
        path = tmp_path / "out.wav"
        out = FileAudioOutput(filepath=str(path), sample_rate=48000)
        out.start()
        out.write(np.full((100, 2), 1.5, dtype=np.float32))
        out.write(np.full((50, 2), -0.5, dtype=np.float32))
        out.stop()

        rate, data = wavfile.read(path)
        assert rate == 48000
        assert data.shape == (150, 2)
        assert data.dtype == np.int16
        assert data[0, 0] == 32767  # clipped, not wrapped negative
        assert data[-1, 0] == -16383

    def test_mono_input_is_written_as_stereo(self, tmp_path):
        path = tmp_path / "mono.wav"
        with FileAudioOutput(filepath=str(path)) as out:
            out.write(np.zeros(10, dtype=np.float32))
        _, data = wavfile.read(path)
        assert data.shape == (10, 2)

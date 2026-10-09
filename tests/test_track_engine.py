"""Tests for the Lyria 3.5 track engine: blending, caching, retries."""

import asyncio
import base64
import io
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.io import wavfile

from focus.generation import track_client as tc
from focus.generation.lyria_client import LyriaConfig
from focus.generation.track_client import (
    TrackCache,
    TrackClient,
    build_track_prompt,
    decode_audio,
)
from focus.generation.track_mix import (
    TrackMixer,
    body_level_db,
    level_envelope_db,
    normalize,
    overlap_gains,
    prepare_track,
    quiet_edges,
)

SR = 8000  # small rate keeps the tests fast


def tone(seconds, level=0.3, freq=220.0, sr=SR, fade_in=0.0, fade_out=0.0):
    """Stereo sine with optional quiet (x0.05) intro/ending sections."""
    t = np.arange(int(seconds * sr)) / sr
    x = level * np.sin(2 * np.pi * freq * t)
    if fade_in:
        x[: int(fade_in * sr)] *= 0.05
    if fade_out:
        x[-int(fade_out * sr) :] *= 0.05
    return np.column_stack([x, x]).astype(np.float32)


def wav_bytes(audio, sr):
    buf = io.BytesIO()
    wavfile.write(buf, sr, (audio * 32767).astype(np.int16))
    return buf.getvalue()


class TestTrackMix:
    def test_normalize_matches_body_level(self):
        y = normalize(tone(20, level=0.1), SR, target_db=-14.0)
        assert abs(body_level_db(y, SR) - -14.0) < 0.5

    def test_quiet_edges_detects_intro_and_ending(self):
        intro, tail = quiet_edges(tone(60, fade_in=12, fade_out=8), SR)
        assert 11 <= intro <= 13
        assert 7 <= tail <= 9

    def test_prepare_trims_intro_only_when_asked(self):
        x = tone(60, fade_in=12, fade_out=8)
        kept_all = prepare_track(x, SR, trim_intro=False)
        trimmed = prepare_track(x, SR, trim_intro=True)
        assert len(kept_all) / SR == pytest.approx(55, abs=1.5)  # 5 s of ending trimmed
        assert len(trimmed) / SR == pytest.approx(48, abs=1.5)  # plus 7 s of intro

    def test_overlap_gains_never_sum_below_one(self):
        g_out, g_in = overlap_gains(1000)
        assert g_out[0] == 1.0 and g_in[-1] == 1.0
        assert g_in[0] == pytest.approx(0.0) and g_out[-1] == pytest.approx(0.0, abs=1e-5)
        assert np.all(g_out + g_in >= 1.0 - 1e-6)

    def test_mixer_streams_overlapped_tracks_without_a_dip(self):
        mixer = TrackMixer(SR, overlap_seconds=4.0)
        a, b = tone(20, freq=220), tone(20, freq=330)
        mixer.push(a)
        mixer.push(b)
        out = np.concatenate([mixer.read(SR // 5) for _ in range(300)])
        assert len(out) == len(a) + len(b) - 4 * SR  # overlap shortens the total
        env = level_envelope_db(out, SR, window=0.5)
        assert env.min() > body_level_db(out, SR) - 1.5
        assert mixer.tracks_started == 2

    def test_mixer_runs_dry_then_fades_in_the_late_track(self):
        mixer = TrackMixer(SR, overlap_seconds=4.0)
        mixer.push(tone(2))
        assert len(mixer.read(3 * SR)) == 2 * SR  # short read: ran dry
        assert len(mixer.read(SR)) == 0
        mixer.push(tone(5))
        resumed = mixer.read(SR)
        assert np.abs(resumed[:10]).max() < 0.01  # starts from silence, no click

    def test_skip_hands_over_quickly(self):
        mixer = TrackMixer(SR, overlap_seconds=4.0)
        mixer.push(tone(30))
        mixer.push(tone(30, freq=440))
        mixer.read(SR)
        mixer.skip()
        mixer.read(5 * SR)  # 4 s quick overlap, then into the next track
        assert mixer.tracks_started == 2
        assert mixer.remaining_seconds < 30


class TestPromptAndDecode:
    def test_prompt_asks_for_instrumental_and_tempo(self):
        p = build_track_prompt("Dark ambient, deep bass.", 120, "Let the pads lead.")
        assert p.startswith("Instrumental only, no vocals.")
        assert "120 BPM" in p and p.endswith("Let the pads lead.")
        assert ".." not in p

    def test_decode_wav_resamples_to_target_rate(self):
        x = tone(1.0, sr=44100)
        y = decode_audio(wav_bytes(x, 44100), 48000)
        assert y.shape[1] == 2
        assert len(y) == pytest.approx(48000, abs=2)


class FakeInteractions:
    """Stands in for client.aio.interactions; scripted errors, then WAV tracks."""

    def __init__(self, errors=(), seconds=12.0):
        self.errors = list(errors)
        self.seconds = seconds
        self.prompts = []

    async def create(self, **kwargs):
        self.prompts.append(kwargs["input"])
        if self.errors:
            raise RuntimeError(self.errors.pop(0))
        data = base64.b64encode(wav_bytes(tone(self.seconds, sr=48000), 48000))
        return SimpleNamespace(output_audio=SimpleNamespace(data=data))


def make_client(tmp_path, fake=None, cached_only=False):
    client = TrackClient(
        LyriaConfig(prompt="Dark ambient", bpm=110),
        profile="deep-work",
        cached_only=cached_only,
        cache=TrackCache("deep-work", root=tmp_path),
    )
    client._running = True
    if fake is not None:
        client._client = SimpleNamespace(aio=SimpleNamespace(interactions=fake))
    return client


async def collect(client, seconds):
    got = 0.0
    async for chunk in client.generate_stream():
        got += len(chunk) / 48000
        if got >= seconds:
            break
    await client.stop()
    return got


@pytest.fixture(autouse=True)
def no_pacing(monkeypatch):
    monkeypatch.setattr(tc, "LEAD_SECONDS", 10_000.0)


class TestTrackClient:
    def test_generates_caches_and_streams(self, tmp_path):
        fake = FakeInteractions()
        client = make_client(tmp_path, fake)
        assert asyncio.run(collect(client, 15.0)) >= 15.0
        assert client.paid_requests >= 2  # first track + the one prefetched behind it
        assert len(TrackCache("deep-work", root=tmp_path).tracks()) == client.paid_requests
        assert all(p.startswith("Instrumental only") for p in fake.prompts)

    def test_blocked_prompt_is_reworded_not_fatal(self, tmp_path):
        fake = FakeInteractions(errors=["400 content_blocked: Input blocked"])
        client = make_client(tmp_path, fake)
        asyncio.run(collect(client, 1.0))
        assert client.paid_requests >= 1
        assert fake.prompts[1] != fake.prompts[0]  # retried with the variation dropped

    def test_blocked_phase_prompt_falls_back_to_profile_prompt(self, tmp_path):
        fake = FakeInteractions(errors=["Request blocked"] * 2)
        client = TrackClient(
            LyriaConfig(prompt="emerging from silence, Dark ambient", bpm=110),
            profile="deep-work",
            cache=TrackCache("deep-work", root=tmp_path),
            main_prompt="Dark ambient",
        )
        client._running = True
        client._client = SimpleNamespace(aio=SimpleNamespace(interactions=fake))
        asyncio.run(collect(client, 1.0))
        # The profile's main prompt (without the phase wording) is tried before the generic one
        main_tries = [
            i for i, p in enumerate(fake.prompts) if "Dark ambient" in p and "emerging" not in p
        ]
        generic = [i for i, p in enumerate(fake.prompts) if "Calm, steady" in p]
        assert main_tries and (not generic or main_tries[0] < generic[0])

    def test_refused_wav_format_falls_back_to_default_format(self, tmp_path):
        fake = FakeInteractions(
            errors=["400 Custom sample_rate in AudioResponseFormat is not supported"]
        )
        client = make_client(tmp_path, fake)
        asyncio.run(collect(client, 1.0))
        assert client._format_level == 1
        assert client.paid_requests >= 1

    def test_auth_error_without_cache_falls_back_to_synth(self, tmp_path):
        fake = FakeInteractions(errors=["API key not valid"] * 10)
        client = make_client(tmp_path, fake)
        assert asyncio.run(collect(client, 1.0)) >= 1.0
        assert client.using_synth
        assert "API key not valid" in client.fallback_reason

    def test_cached_only_plays_library_without_requests(self, tmp_path):
        cache = TrackCache("deep-work", root=tmp_path)
        cache.save(wav_bytes(tone(12, sr=48000), 48000), "p")
        client = make_client(tmp_path, cached_only=True)
        assert asyncio.run(collect(client, 5.0)) >= 5.0
        assert client.paid_requests == 0

    def test_cached_only_with_empty_cache_errors_clearly(self, tmp_path):
        client = make_client(tmp_path, cached_only=True)
        with pytest.raises(ValueError, match="No cached tracks"):
            asyncio.run(client.connect())

    def test_cache_prunes_oldest_beyond_cap(self, tmp_path):
        cache = TrackCache("deep-work", root=tmp_path, max_mb=1)
        big = wav_bytes(tone(8, sr=48000), 48000)  # ~1.5 MB each
        cache.save(big, "a")
        newest = cache.save(big, "b")
        assert cache.tracks() == [newest]  # oldest pruned, the one just saved kept


class TestTrackLibrary:
    def _lib(self, tmp_path, n=3, **kw):
        lib = TrackCache("deep-work", root=tmp_path, **kw)
        paths = [lib.save(wav_bytes(tone(1, sr=48000), 48000), f"p{i}") for i in range(n)]
        return lib, paths

    def test_record_play_counts_and_retires_at_cap(self, tmp_path):
        lib, (a, *_) = self._lib(tmp_path, n=1, max_plays=2, cooldown_hours=0)
        lib.record_play(a)
        assert lib.plays(a) == 1 and not lib.retired(a)
        lib.record_play(a)
        assert lib.retired(a)
        assert lib.pick() is None  # retired tracks are never reused

    def test_cooldown_blocks_recent_tracks(self, tmp_path):
        lib, (a, b, c) = self._lib(tmp_path, cooldown_hours=12)
        lib.record_play(a)
        lib.record_play(b)
        assert lib.pick() == c
        lib.record_play(c)
        assert lib.pick() is None  # everything played within 12 h

    def test_prefers_least_played(self, tmp_path):
        lib, (a, b, c) = self._lib(tmp_path, cooldown_hours=0)
        for t in (a, a, b):
            lib.record_play(t)
        assert lib.pick() == c

    def test_relaxed_pick_ignores_cap_and_takes_longest_unplayed(self, tmp_path):
        lib, (a, b) = self._lib(tmp_path, n=2, max_plays=1, cooldown_hours=12)
        lib.record_play(a)
        lib.record_play(b)
        assert lib.pick() is None
        assert lib.pick(relaxed=True) == a

    def test_prune_removes_retired_before_newer_tracks(self, tmp_path):
        big = wav_bytes(tone(8, sr=48000), 48000)  # ~1.5 MB
        lib = TrackCache("deep-work", root=tmp_path, max_mb=3, max_plays=1)
        old = lib.save(big, "old")
        retired = lib.save(big, "retired")
        lib.record_play(retired)
        newest = lib.save(big, "new")
        assert set(lib.tracks()) == {old, newest}

    def test_stats(self, tmp_path):
        lib, (a, b, c) = self._lib(tmp_path, max_plays=1, cooldown_hours=12)
        lib.record_play(a)
        st = lib.stats()
        assert (st["tracks"], st["retired"], st["ready"], st["plays"]) == (3, 1, 2, 1)


class TestReuseInEngine:
    def test_reuses_library_instead_of_paying(self, tmp_path):
        lib = TrackCache("deep-work", root=tmp_path)
        for i in range(3):
            lib.save(wav_bytes(tone(12, sr=48000), 48000), f"p{i}")
        fake = FakeInteractions()
        client = make_client(tmp_path, fake)
        asyncio.run(collect(client, 15.0))
        assert client.paid_requests == 0
        assert client.reused_tracks >= 2
        assert sum(lib.plays(t) for t in lib.tracks()) == client.reused_tracks

    def test_generates_when_library_is_cooling_down(self, tmp_path):
        lib = TrackCache("deep-work", root=tmp_path)
        lib.record_play(lib.save(wav_bytes(tone(12, sr=48000), 48000), "p"))
        fake = FakeInteractions()
        client = make_client(tmp_path, fake)
        asyncio.run(collect(client, 1.0))
        assert client.paid_requests >= 1

    def test_queued_but_unplayed_track_is_not_counted(self):
        mixer = TrackMixer(SR, overlap_seconds=1.0)
        started = []
        mixer.push(tone(5), on_start=lambda: started.append("a"))
        mixer.push(tone(5), on_start=lambda: started.append("b"))
        mixer.read(SR)
        assert started == ["a"]

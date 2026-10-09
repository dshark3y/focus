"""Lyria 3.5 track engine: generate whole songs ahead and blend them.

Lyria 3.5 has no live stream; each request returns a finished ~2.5 minute
track (about 35 s to generate, $0.08 each). This client keeps one track ready
ahead of playback, blends tracks with :class:`TrackMixer`, and exposes the same
``connect`` / ``generate_stream`` / ``set_prompt`` / ``stop`` interface as the
real-time client so the CLI pipeline is unchanged.

Every generated track is kept in a :class:`TrackCache` library and reused
while it has fewer than 10 plays and hasn't played in the last 12 hours; a new
track is generated only when nothing qualifies. ``offline`` plays the saved
library only, ignoring the cap and cooldown, with no network calls at all.
"""

import asyncio
import base64
import io
import random
import shutil
import subprocess
import time
import warnings
from collections import deque
from math import gcd
from pathlib import Path

import numpy as np

from focus.generation.lyria_client import (
    EnhancedSynthClient,
    LyriaConfig,
    _short_reason,
    resolve_api_key,
)
from focus.generation.track_library import (  # noqa: F401  (re-exported)
    DEFAULT_COOLDOWN_HOURS,
    DEFAULT_MAX_PLAYS,
    TrackCache,
    cache_root,
)
from focus.generation.track_mix import TrackMixer, prepare_track

MODEL = "lyria-3.5"
TRACK_LENGTH_HINT = "about 2 minutes 30 seconds"
PREFETCH_TRACKS = 1  # tracks kept ready beyond the one playing
LEAD_SECONDS = 4.0  # how far ahead of real time the stream may run (see synth)
CHUNK_SECONDS = 0.2
MAX_ATTEMPTS = 5
RECENT_TRACKS = 5  # in-session guard on top of the library cooldown

# Neutral, filter-safe ways to vary consecutive tracks. Brand or instrument
# names (e.g. "Rhodes") have tripped the content filter, so keep these plain.
VARIATIONS = [
    "",
    "Slightly sparser arrangement.",
    "Slightly fuller arrangement.",
    "Let the pads lead.",
    "Let the rhythm lead.",
]
SAFE_FALLBACK_PROMPT = "Calm, steady ambient electronic music for focused work."


def build_track_prompt(base_prompt: str, bpm: int, variation: str = "") -> str:
    """Prompt for one track: instrumental, steady tempo, gentle edges."""
    parts = [
        "Instrumental only, no vocals.",
        base_prompt.strip().rstrip(".") + ".",
        f"Tempo around {bpm} BPM.",
        f"Length {TRACK_LENGTH_HINT}.",
        "Short gentle intro and a gentle ending.",
    ]
    if variation:
        parts.append(variation)
    return " ".join(parts)


def decode_audio(data: bytes, sample_rate: int) -> np.ndarray:
    """Decode WAV (or MP3 via ffmpeg) bytes to float32 stereo at ``sample_rate``."""
    if data[:4] == b"RIFF":
        from scipy.io import wavfile

        rate, audio = wavfile.read(io.BytesIO(data))
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        else:
            audio = audio.astype(np.float32)
    else:
        if not shutil.which("ffmpeg"):
            raise RuntimeError("got MP3 audio but ffmpeg is not installed to decode it")
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", "-", "-f", "f32le", "-ac", "2", "-"],
            input=data,
            capture_output=True,
            check=True,
        ).stdout
        probe_rate = 44100  # ffmpeg keeps the source rate; Lyria MP3 is 44.1 kHz
        audio = np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).copy()
        rate = probe_rate
    if audio.ndim == 1:
        audio = np.column_stack([audio, audio])
    if rate != sample_rate:
        from scipy.signal import resample_poly

        g = gcd(sample_rate, rate)
        audio = resample_poly(audio, sample_rate // g, rate // g, axis=0).astype(np.float32)
    return audio


class TrackClient:
    """Lyria 3.5 engine with ahead-of-time generation, blending and caching."""

    resumable = True  # pause / next keep this client (and its queue) alive

    def __init__(
        self,
        config: LyriaConfig,
        profile: str,
        verbose: bool = False,
        offline: bool = False,
        cache: TrackCache | None = None,
        main_prompt: str | None = None,
        max_plays: int = DEFAULT_MAX_PLAYS,
        cooldown_hours: float = DEFAULT_COOLDOWN_HOURS,
    ):
        self.config = config
        self.verbose = verbose
        self.offline = offline
        self.cache = cache or TrackCache(
            profile, max_plays=max_plays, cooldown_hours=cooldown_hours
        )
        self.fallback_reason: str | None = None
        self.paid_requests = 0  # successful generations (what gets billed)
        self.reused_tracks = 0  # library tracks that started playing
        self.status: str = "starting"
        self._prompt = config.prompt
        # The profile's plain prompt, used if a phase-modified prompt is blocked
        self._main_prompt = main_prompt or config.prompt
        self._client = None
        self._running = False
        self._mixer = TrackMixer(config.sample_rate)
        self._producer: asyncio.Task | None = None
        self._recent: deque[Path] = deque(maxlen=RECENT_TRACKS)
        self._pending: set[Path] = set()  # picked/queued but not yet playing
        self._variation = random.randrange(len(VARIATIONS))
        # 0 = request WAV (decoded without ffmpeg), 1 = the default MP3
        self._format_level = 0
        self._synth: EnhancedSynthClient | None = None

    @property
    def engine_name(self) -> str:
        return "offline" if self.offline else MODEL

    @property
    def using_synth(self) -> bool:
        """True once generation failed with no cache to fall back on."""
        return self._synth is not None

    # -- interface shared with LyriaClient --------------------------------

    async def connect(self, api_key: str | None = None) -> None:
        self._running = True
        if self.offline:
            if not self.cache.tracks():
                # Nothing saved for this profile: borrow every profile's tracks
                self.cache = TrackCache(None, root=self.cache.root)
            if not self.cache.tracks():
                raise ValueError(
                    f"No saved tracks yet ({self.cache.root}). Play some sessions with "
                    "--engine lyria-3.5 to build the library."
                )
            return
        api_key = api_key or resolve_api_key()
        if not api_key:
            raise ValueError("API key required. Set GOOGLE_API_KEY (or GEMINI_API_KEY).")
        from google import genai

        self._client = genai.Client(api_key=api_key)

    async def set_prompt(self, new_prompt: str) -> None:
        """Use ``new_prompt`` for tracks generated from now on."""
        self._prompt = new_prompt

    def skip(self) -> None:
        """Move on to the next track (quick overlap once it is ready)."""
        self._mixer.skip()

    async def stop(self) -> None:
        self._running = False
        if self._producer is not None:
            self._producer.cancel()
            try:
                await self._producer
            except (asyncio.CancelledError, Exception):
                pass
            self._producer = None
        if self._synth is not None:
            await self._synth.stop()

    async def generate_stream(self):
        """Yield ~0.2 s stereo chunks, paced to real time.

        Safe to call again after the previous generator was closed (pause): the
        queue, mixer position and producer all live on the client.
        """
        if self._producer is None and self._running:
            self._producer = asyncio.create_task(self._produce())
        sr = self.config.sample_rate
        chunk_n = int(CHUNK_SECONDS * sr)
        produced = 0.0
        t0: float | None = None
        while self._running:
            if self._synth is not None:  # Lyria unusable and no cache: synth fallback
                async for chunk in self._synth.generate_stream():
                    if not self._running:
                        return
                    yield chunk
                return
            chunk = self._mixer.read(chunk_n)
            if len(chunk) == 0:
                # Ran dry (first track still generating, or a slow request):
                # wait, and restart the real-time clock once audio is back.
                self.status = "generating"
                t0 = None
                if self._producer is not None and self._producer.done():
                    return  # producer gave up (error already recorded)
                await asyncio.sleep(0.1)
                continue
            self.status = "playing"
            if t0 is None:
                t0, produced = time.monotonic(), 0.0
            yield chunk
            produced += len(chunk) / sr
            ahead = produced - (time.monotonic() - t0)
            await asyncio.sleep(max(0.0, ahead - LEAD_SECONDS))

    # -- producer -----------------------------------------------------------

    async def _produce(self) -> None:
        """Keep PREFETCH_TRACKS ready beyond the playing one."""
        first = True
        while self._running:
            if self._mixer.queued >= PREFETCH_TRACKS:
                await asyncio.sleep(0.5)
                continue
            picked = await self._next_track()
            if picked is None:
                return
            audio, path, reused = picked
            self._mixer.push(
                prepare_track(audio, self.config.sample_rate, trim_intro=not first),
                on_start=lambda p=path, r=reused: self._on_track_start(p, r),
            )
            first = False

    def _on_track_start(self, path: Path | None, reused: bool) -> None:
        """A queued track began playing: count the play in the library."""
        if path is None:
            return
        self._pending.discard(path)
        self._recent.append(path)
        self.cache.record_play(path, count=not self.offline)
        if reused:
            self.reused_tracks += 1

    async def _next_track(self) -> tuple[np.ndarray, Path | None, bool] | None:
        """Next track as (audio, library path, reused?), or None to stop.

        Reuse a library track when one qualifies (under the play cap and out of
        cooldown); otherwise generate a new one. If generation fails for good,
        keep going on the library (cap and cooldown relaxed), else the synth.
        """
        if self.offline:
            return self._load_cached(relaxed=True)
        reused = self._load_cached(relaxed=False)
        if reused is not None:
            return reused
        generated = await self._generate()
        if generated is not None:
            return generated
        fallback = self._load_cached(relaxed=True)
        if fallback is not None:
            self.offline = True
            return fallback
        self._synth = EnhancedSynthClient(self.config, verbose=self.verbose)
        await self._synth.connect()
        return None

    def _load_cached(self, relaxed: bool) -> tuple[np.ndarray, Path, bool] | None:
        avoid = set(self._recent) | self._pending
        for _ in range(3):  # skip over unreadable files
            path = self.cache.pick(avoid, relaxed=relaxed)
            if path is None:
                return None
            try:
                audio = decode_audio(path.read_bytes(), self.config.sample_rate)
            except Exception:
                avoid.add(path)
                continue
            self._pending.add(path)
            if self.verbose:
                plays = self.cache.plays(path)
                print(f"   [Lyria 3.5] Reusing {path.name} (played {plays}x)")
            return audio, path, True
        return None

    async def _generate(self) -> tuple[np.ndarray, Path | None, bool] | None:
        """Generate one track, retrying transient errors and rewording blocked prompts."""
        variation = VARIATIONS[self._variation % len(VARIATIONS)]
        self._variation += 1
        candidates = [
            build_track_prompt(self._prompt, self.config.bpm, variation),
            build_track_prompt(self._prompt, self.config.bpm),
            build_track_prompt(self._main_prompt, self.config.bpm),
            build_track_prompt(SAFE_FALLBACK_PROMPT, self.config.bpm),
        ]
        prompts = list(dict.fromkeys(candidates))  # drop duplicates, keep order
        attempt, delay = 0, 2.0
        while self._running and attempt < MAX_ATTEMPTS and prompts:
            attempt += 1
            prompt = prompts[0]
            try:
                started = time.monotonic()
                data = await self._request(prompt)
                audio = decode_audio(data, self.config.sample_rate)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                msg = str(e)
                low = msg.lower()
                if "blocked" in low or "prohibited" in low or "sensitive" in low:
                    prompts.pop(0)  # reword and try again straight away
                    if self.verbose:
                        print(f"   [Lyria 3.5] Prompt blocked, rewording ({_short_reason(msg)})")
                    continue
                if self._format_level == 0 and any(
                    s in low for s in ("responseformat", "response_format", "mime", "sample_rate")
                ):
                    self._format_level = 1  # WAV request refused: use the default MP3
                    continue
                transient = any(
                    s in low for s in ("429", "500", "502", "503", "504", "timeout", "unavailable")
                )
                self.fallback_reason = _short_reason(msg)
                if not transient:
                    return None
                if self.verbose:
                    print(f"   [Lyria 3.5] {self.fallback_reason}; retrying in {delay:.0f}s")
                await asyncio.sleep(delay)
                delay *= 2
                continue
            self.paid_requests += 1
            self.fallback_reason = None
            path = self.cache.save(data, prompt)
            if path is not None:
                self._pending.add(path)
            if self.verbose:
                took = time.monotonic() - started
                length = len(audio) / self.config.sample_rate
                n = self.paid_requests
                print(f"   [Lyria 3.5] Track {n} ready in {took:.0f}s ({length:.0f}s long)")
            return audio, path, False
        if self.fallback_reason is None:
            self.fallback_reason = "prompt blocked by content filter"
        return None

    async def _request(self, prompt: str) -> bytes:
        kwargs = {"model": MODEL, "input": prompt}
        if self._format_level == 0:
            # Custom sample rates are refused; decode_audio resamples to 48 kHz
            kwargs["response_format"] = {"type": "audio", "mime_type": "audio/wav"}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            interaction = await self._client.aio.interactions.create(**kwargs)
        audio = interaction.output_audio
        if not audio or not audio.data:
            raise RuntimeError("Lyria 3.5 returned no audio")
        return base64.b64decode(audio.data)

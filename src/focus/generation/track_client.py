"""Lyria 3.5 track engine: generate whole songs ahead and blend them.

Lyria 3.5 has no live stream; each request returns a finished ~2.5 minute
track (about 35 s to generate, $0.08 each). This client keeps one track ready
ahead of playback, blends tracks with :class:`TrackMixer`, and exposes the same
``connect`` / ``generate_stream`` / ``set_prompt`` / ``stop`` interface as the
real-time client so the CLI pipeline is unchanged.

Every generated track is cached on disk. A session starts instantly from the
cache when it has tracks for the profile, and ``cached_only`` replays the cache
without making any paid requests.
"""

import asyncio
import base64
import io
import json
import os
import random
import shutil
import subprocess
import time
import warnings
from collections import deque
from datetime import datetime
from math import gcd
from pathlib import Path

import numpy as np

from focus.generation.lyria_client import (
    EnhancedSynthClient,
    LyriaConfig,
    _short_reason,
    resolve_api_key,
)
from focus.generation.track_mix import TrackMixer, prepare_track

MODEL = "lyria-3.5"
TRACK_LENGTH_HINT = "about 2 minutes 30 seconds"
PREFETCH_TRACKS = 1  # tracks kept ready beyond the one playing
LEAD_SECONDS = 4.0  # how far ahead of real time the stream may run (see synth)
CHUNK_SECONDS = 0.2
MAX_ATTEMPTS = 5
RECENT_TRACKS = 5  # cached tracks not to repeat back to back
DEFAULT_CACHE_MB = 1024

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


def cache_root() -> Path:
    """Track cache: $FOCUS_CACHE_DIR, else $XDG_CACHE_HOME/focus/tracks, else ~/.cache."""
    override = os.environ.get("FOCUS_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or "~/.cache"
    return Path(base).expanduser() / "focus" / "tracks"


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in name.lower()) or "default"


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


class TrackCache:
    """Generated tracks on disk, one folder per profile, capped by total size."""

    def __init__(self, profile: str, root: Path | None = None, max_mb: int | None = None):
        self.root = root or cache_root()
        self.dir = self.root / _slug(profile)
        env_mb = os.environ.get("FOCUS_CACHE_MAX_MB")
        self.max_bytes = int(max_mb or (int(env_mb) if env_mb else DEFAULT_CACHE_MB)) * 1024**2

    def tracks(self) -> list[Path]:
        if not self.dir.exists():
            return []
        return sorted(p for p in self.dir.iterdir() if p.suffix in (".wav", ".mp3"))

    def save(self, data: bytes, prompt: str) -> Path | None:
        """Store a track (best-effort); returns its path or None on failure."""
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            path = self.dir / f"{stamp}.{'wav' if data[:4] == b'RIFF' else 'mp3'}"
            path.write_bytes(data)
            meta = {"model": MODEL, "prompt": prompt, "created": stamp}
            path.with_suffix(".json").write_text(json.dumps(meta))
            self._prune(keep=path)
            return path
        except OSError:
            return None

    def _prune(self, keep: Path) -> None:
        """Delete the oldest tracks (across profiles) beyond the size cap, never ``keep``."""
        files = sorted(
            (p for p in self.root.glob("*/*") if p.suffix in (".wav", ".mp3") and p != keep),
            key=lambda p: p.stat().st_mtime,
        )
        total = keep.stat().st_size + sum(p.stat().st_size for p in files)
        for p in files:
            if total <= self.max_bytes:
                break
            total -= p.stat().st_size
            p.unlink(missing_ok=True)
            p.with_suffix(".json").unlink(missing_ok=True)

    def pick(self, avoid: set[Path]) -> Path | None:
        """A random cached track, avoiding recently played ones when possible."""
        tracks = self.tracks()
        fresh = [p for p in tracks if p not in avoid]
        pool = fresh or tracks
        return random.choice(pool) if pool else None


class TrackClient:
    """Lyria 3.5 engine with ahead-of-time generation, blending and caching."""

    engine_name = MODEL
    resumable = True  # pause / next keep this client (and its queue) alive

    def __init__(
        self,
        config: LyriaConfig,
        profile: str,
        verbose: bool = False,
        cached_only: bool = False,
        cache: TrackCache | None = None,
        main_prompt: str | None = None,
    ):
        self.config = config
        self.verbose = verbose
        self.cached_only = cached_only
        self.cache = cache or TrackCache(profile)
        self.fallback_reason: str | None = None
        self.paid_requests = 0  # successful generations (what gets billed)
        self.status: str = "starting"
        self._prompt = config.prompt
        # The profile's plain prompt, used if a phase-modified prompt is blocked
        self._main_prompt = main_prompt or config.prompt
        self._client = None
        self._running = False
        self._mixer = TrackMixer(config.sample_rate)
        self._producer: asyncio.Task | None = None
        self._recent: deque[Path] = deque(maxlen=RECENT_TRACKS)
        self._variation = random.randrange(len(VARIATIONS))
        # 0 = request WAV (decoded without ffmpeg), 1 = the default MP3
        self._format_level = 0
        self._synth: EnhancedSynthClient | None = None

    @property
    def using_synth(self) -> bool:
        """True once generation failed with no cache to fall back on."""
        return self._synth is not None

    # -- interface shared with LyriaClient --------------------------------

    async def connect(self, api_key: str | None = None) -> None:
        self._running = True
        if self.cached_only:
            if not self.cache.tracks():
                raise ValueError(
                    f"No cached tracks for this profile yet ({self.cache.dir}). "
                    "Run once without --cached-only to build the library."
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
            audio = await self._next_track(first)
            if audio is None:
                return
            self._mixer.push(prepare_track(audio, self.config.sample_rate, trim_intro=not first))
            first = False

    async def _next_track(self, first: bool) -> np.ndarray | None:
        """Next track's audio: cache on start / cached-only, else a new generation."""
        use_cache = self.cached_only or (first and self.cache.tracks())
        if use_cache:
            audio = self._load_cached()
            if audio is not None:
                return audio
        if self.cached_only:
            return None
        audio = await self._generate()
        if audio is not None:
            return audio
        # Generation failed for good: keep playing from the cache if we can
        cached = self._load_cached()
        if cached is not None:
            self.cached_only = True
            return cached
        self._synth = EnhancedSynthClient(self.config, verbose=self.verbose)
        await self._synth.connect()
        return None

    def _load_cached(self) -> np.ndarray | None:
        path = self.cache.pick(set(self._recent))
        if path is None:
            return None
        try:
            audio = decode_audio(path.read_bytes(), self.config.sample_rate)
        except Exception:
            return None
        self._recent.append(path)
        if self.verbose:
            print(f"   [Lyria 3.5] Playing cached track {path.name}")
        return audio

    async def _generate(self) -> np.ndarray | None:
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
            self.cache.save(data, prompt)
            if self.verbose:
                took = time.monotonic() - started
                length = len(audio) / self.config.sample_rate
                n = self.paid_requests
                print(f"   [Lyria 3.5] Track {n} ready in {took:.0f}s ({length:.0f}s long)")
            return audio
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

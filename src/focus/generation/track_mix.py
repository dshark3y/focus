"""Blend whole generated tracks into one continuous stream.

Lyria 3.5 returns finished ~2.5 minute songs, each with its own quiet intro and
ending. Equal-power crossfading two such songs overlaps a fade-out with a
fade-in and leaves an audible lull. Instead each track is:

1. loudness-matched on its body (the middle half), so tracks sit at one level;
2. trimmed of most of its quiet intro (after the first track) and quiet ending;
3. overlapped with the next one: the incoming track rises over the first half
   of the overlap while the outgoing one stays at full level, then the outgoing
   track fades over the second half.
"""

from collections import deque

import numpy as np

TARGET_BODY_DB = -14.0  # RMS level each track's body is matched to
MAX_GAIN_DB = 12.0  # never boost a quiet track more than this
EDGE_THRESHOLD_DB = 6.0  # intro/ending = stretch quieter than body by this much
KEEP_INTRO_SECONDS = 5.0  # leave this much of a trimmed intro, so it still rises
KEEP_TAIL_SECONDS = 3.0
OVERLAP_SECONDS = 16.0
SKIP_OVERLAP_SECONDS = 4.0  # quicker handover when the listener presses "next"
RESUME_FADE_SECONDS = 2.0  # fade-in after the mixer ran dry


def level_envelope_db(audio: np.ndarray, sample_rate: int, window: float = 1.0) -> np.ndarray:
    """Short-term RMS level in dB per window (stereo averaged)."""
    n = max(1, int(window * sample_rate))
    k = len(audio) // n
    if k == 0:
        return np.array([20 * np.log10(np.sqrt(np.mean(audio**2)) + 1e-9)])
    frames = audio[: k * n].reshape(k, n, -1)
    return 20 * np.log10(np.sqrt(np.mean(frames**2, axis=(1, 2))) + 1e-9)


def body_level_db(audio: np.ndarray, sample_rate: int) -> float:
    """Median short-term level over the middle half of the track."""
    env = level_envelope_db(audio, sample_rate)
    if len(env) < 4:
        return float(np.median(env))
    return float(np.median(env[len(env) // 4 : 3 * len(env) // 4]))


def normalize(
    audio: np.ndarray,
    sample_rate: int,
    target_db: float = TARGET_BODY_DB,
    max_gain_db: float = MAX_GAIN_DB,
) -> np.ndarray:
    """Scale so the track's body sits at ``target_db`` (gain capped)."""
    gain_db = min(max_gain_db, target_db - body_level_db(audio, sample_rate))
    return (audio * 10 ** (gain_db / 20)).astype(np.float32)


def quiet_edges(
    audio: np.ndarray,
    sample_rate: int,
    threshold_db: float = EDGE_THRESHOLD_DB,
) -> tuple[float, float]:
    """Seconds of quiet intro and quiet ending (below body level - threshold)."""
    env = level_envelope_db(audio, sample_rate)
    loud = np.where(env >= body_level_db(audio, sample_rate) - threshold_db)[0]
    if len(loud) == 0:
        return 0.0, 0.0
    return float(loud[0]), float(len(env) - 1 - loud[-1])


def prepare_track(audio: np.ndarray, sample_rate: int, trim_intro: bool = True) -> np.ndarray:
    """Normalize, then trim the quiet ending (and the quiet intro if asked)."""
    audio = normalize(audio, sample_rate)
    intro, tail = quiet_edges(audio, sample_rate)
    start = int(max(0.0, intro - KEEP_INTRO_SECONDS) * sample_rate) if trim_intro else 0
    end = len(audio) - int(max(0.0, tail - KEEP_TAIL_SECONDS) * sample_rate)
    if end - start < sample_rate:  # degenerate track: keep it whole
        return audio
    return audio[start:end]


def _rise(n: int) -> np.ndarray:
    """Smooth 0 -> 1 curve (raised sine) of length n."""
    return (np.sin(np.linspace(0.0, np.pi / 2, n, dtype=np.float32)) ** 2).astype(np.float32)


def overlap_gains(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Gains for an n-sample overlap: (outgoing, incoming).

    Incoming rises over the first half while outgoing holds at 1; outgoing then
    falls over the second half while incoming holds at 1. The sum never drops
    below 1, so there is no lull; the session limiter catches the brief bump.
    """
    half = n // 2
    g_in = np.ones(n, dtype=np.float32)
    g_out = np.ones(n, dtype=np.float32)
    g_in[:half] = _rise(half)
    g_out[half:] = _rise(n - half)[::-1]
    return g_out, g_in


class TrackMixer:
    """Pull-based mixer: queue prepared tracks, read a continuous stream.

    ``read(n)`` returns up to n samples. It returns fewer (possibly zero) only
    when it has run dry: the current track ended and no next track is queued.
    """

    def __init__(self, sample_rate: int, overlap_seconds: float = OVERLAP_SECONDS):
        self.sample_rate = sample_rate
        self.overlap_n = int(overlap_seconds * sample_rate)
        self._queue: deque[np.ndarray] = deque()
        self._current: np.ndarray | None = None
        self._pos = 0
        self._pending_skip = False
        self._resume_fade = False
        self.tracks_started = 0

    @property
    def queued(self) -> int:
        """Tracks waiting after the current one."""
        return len(self._queue)

    @property
    def remaining_seconds(self) -> float:
        """Seconds left in the current track (0 if none)."""
        if self._current is None:
            return 0.0
        return (len(self._current) - self._pos) / self.sample_rate

    def push(self, track: np.ndarray) -> None:
        self._queue.append(track.astype(np.float32, copy=False))

    def skip(self) -> None:
        """Hand over to the next track quickly (as soon as one is queued)."""
        self._pending_skip = True

    def _start_next(self) -> None:
        self._current = self._queue.popleft()
        self._pos = 0
        self.tracks_started += 1
        if self._resume_fade:  # coming back from silence: don't start abruptly
            n = min(len(self._current), int(RESUME_FADE_SECONDS * self.sample_rate))
            self._current = self._current.copy()
            self._current[:n] *= _rise(n)[:, np.newaxis]
            self._resume_fade = False

    def _maybe_begin_overlap(self) -> None:
        """Splice the next track in once the current one reaches its overlap zone."""
        if self._current is None or not self._queue:
            return
        cur, nxt = self._current, self._queue[0]
        if self._pending_skip:
            skip_n = int(SKIP_OVERLAP_SECONDS * self.sample_rate)
            cur = cur[: min(len(cur), self._pos + skip_n)]
            ov = len(cur) - self._pos
            self._pending_skip = False
        else:
            ov = min(self.overlap_n, len(cur) // 2, len(nxt) // 2)
            if self._pos < len(cur) - ov:
                return
            ov = len(cur) - self._pos  # may be shorter if the next track arrived late
        ov = min(ov, len(nxt))
        if ov <= 0:
            return
        self._queue.popleft()
        g_out, g_in = overlap_gains(ov)
        blended = cur[self._pos : self._pos + ov] * g_out[:, None] + nxt[:ov] * g_in[:, None]
        self._current = np.concatenate([blended, nxt[ov:]]).astype(np.float32)
        self._pos = 0
        self.tracks_started += 1

    def read(self, n: int) -> np.ndarray:
        parts = []
        need = n
        while need > 0:
            if self._current is None or self._pos >= len(self._current):
                if not self._queue:
                    if self._current is not None:
                        self._current = None
                        self._resume_fade = True
                    break
                self._start_next()
            self._maybe_begin_overlap()
            cur = self._current
            # Stop this slice at the overlap point so the splice happens on time
            limit = len(cur)
            if self._queue and not self._pending_skip:
                ov = min(self.overlap_n, len(cur) // 2, len(self._queue[0]) // 2)
                if self._pos < len(cur) - ov:
                    limit = len(cur) - ov
            take = min(need, limit - self._pos)
            if take <= 0:
                continue
            parts.append(cur[self._pos : self._pos + take])
            self._pos += take
            need -= take
        if not parts:
            return np.zeros((0, 2), dtype=np.float32)
        return np.concatenate(parts).astype(np.float32)

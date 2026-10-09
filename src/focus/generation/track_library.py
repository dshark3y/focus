"""On-disk library of generated Lyria 3.5 tracks, with play counts.

Every generated track is kept (one folder per profile) next to a JSON sidecar
holding its prompt and play history. A track is reused only while it has been
played fewer than ``max_plays`` times and not within ``cooldown_hours``; a new
track is generated only when nothing qualifies. Early on that means mostly new
tracks; once the library covers a cooldown window, most plays are free reuse,
and tracks retire after their last play so the rotation keeps refreshing.
"""

import json
import os
import random
from datetime import datetime, timedelta
from pathlib import Path

MODEL = "lyria-3.5"
DEFAULT_MAX_PLAYS = 10
DEFAULT_COOLDOWN_HOURS = 12.0
DEFAULT_CACHE_MB = 1024
AUDIO_SUFFIXES = (".wav", ".mp3")


def cache_root() -> Path:
    """Library root: $FOCUS_CACHE_DIR, else $XDG_CACHE_HOME/focus/tracks, else ~/.cache."""
    override = os.environ.get("FOCUS_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or "~/.cache"
    return Path(base).expanduser() / "focus" / "tracks"


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in name.lower()) or "default"


def _now() -> datetime:
    return datetime.now().astimezone()


class TrackCache:
    """Generated tracks for one profile, with play counts and a size cap."""

    def __init__(
        self,
        profile: str | None,
        root: Path | None = None,
        max_mb: int | None = None,
        max_plays: int = DEFAULT_MAX_PLAYS,
        cooldown_hours: float = DEFAULT_COOLDOWN_HOURS,
    ):
        self.root = root or cache_root()
        # profile=None reads every profile's tracks (offline fallback); it can't save
        self.dir = self.root / _slug(profile) if profile else self.root
        self.all_profiles = profile is None
        env_mb = os.environ.get("FOCUS_CACHE_MAX_MB")
        self.max_bytes = int(max_mb or (int(env_mb) if env_mb else DEFAULT_CACHE_MB)) * 1024**2
        self.max_plays = max_plays
        self.cooldown = timedelta(hours=cooldown_hours)

    # -- files and metadata -------------------------------------------------

    def tracks(self) -> list[Path]:
        if not self.dir.exists():
            return []
        pattern = "*/*" if self.all_profiles else "*"
        return sorted(p for p in self.dir.glob(pattern) if p.suffix in AUDIO_SUFFIXES)

    def meta(self, track: Path) -> dict:
        try:
            return json.loads(track.with_suffix(".json").read_text())
        except (OSError, ValueError):
            return {}

    def _write_meta(self, track: Path, meta: dict) -> None:
        try:
            track.with_suffix(".json").write_text(json.dumps(meta))
        except OSError:
            pass

    def plays(self, track: Path) -> int:
        return int(self.meta(track).get("plays", 0))

    def last_played(self, track: Path) -> datetime | None:
        stamp = self.meta(track).get("last_played")
        try:
            return datetime.fromisoformat(stamp) if stamp else None
        except ValueError:
            return None

    def retired(self, track: Path) -> bool:
        return self.plays(track) >= self.max_plays

    def save(self, data: bytes, prompt: str) -> Path | None:
        """Store a new track (best-effort); returns its path or None on failure."""
        if self.all_profiles:
            return None
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            path = self.dir / f"{stamp}.{'wav' if data[:4] == b'RIFF' else 'mp3'}"
            path.write_bytes(data)
            meta = {"model": MODEL, "prompt": prompt, "created": stamp, "plays": 0}
            self._write_meta(path, meta)
            self._prune(keep=path)
            return path
        except OSError:
            return None

    def record_play(self, track: Path, count: bool = True) -> None:
        """Record a play when the track actually starts.

        ``count=False`` (offline mode) only updates ``last_played``, so offline
        listening rotates fairly without using up a track's play cap.
        """
        if not track.exists():
            return
        meta = self.meta(track)
        if count:
            meta["plays"] = int(meta.get("plays", 0)) + 1
        else:
            meta["offline_plays"] = int(meta.get("offline_plays", 0)) + 1
        meta["last_played"] = _now().isoformat(timespec="seconds")
        self._write_meta(track, meta)

    # -- choosing a track -----------------------------------------------------

    def pick(self, avoid: set[Path] | None = None, relaxed: bool = False) -> Path | None:
        """Choose a track to reuse, or None if none qualifies.

        Strict (default): fewer than ``max_plays`` plays, not played within the
        cooldown, not in ``avoid``; least-played first, random among ties.
        Relaxed (offline mode): any track, ignoring cap and cooldown, not in
        ``avoid`` when possible; never-played first, then longest since last
        played, so the whole library rotates before anything repeats.
        """
        avoid = avoid or set()
        tracks = self.tracks()
        if not relaxed:
            now = _now()
            ok = []
            for t in tracks:
                if t in avoid or self.retired(t):
                    continue
                last = self.last_played(t)
                if last is not None and now - last < self.cooldown:
                    continue
                ok.append(t)
            if not ok:
                return None
            fewest = min(self.plays(t) for t in ok)
            return random.choice([t for t in ok if self.plays(t) == fewest])

        pool = [t for t in tracks if t not in avoid] or tracks
        if not pool:
            return None
        oldest = datetime.min.replace(tzinfo=_now().tzinfo)
        return min(pool, key=lambda t: self.last_played(t) or oldest)

    # -- housekeeping -----------------------------------------------------------

    def _prune(self, keep: Path) -> None:
        """Free space beyond the size cap: retired tracks first, then the oldest.

        Works across all profiles' folders and never deletes ``keep``.
        """
        files = [p for p in self.root.glob("*/*") if p.suffix in AUDIO_SUFFIXES and p != keep]
        total = keep.stat().st_size + sum(p.stat().st_size for p in files)
        if total <= self.max_bytes:
            return

        def order(p: Path) -> tuple[int, float]:
            retired = int(self.meta(p).get("plays", 0)) >= self.max_plays
            return (0 if retired else 1, p.stat().st_mtime)

        for p in sorted(files, key=order):
            if total <= self.max_bytes:
                break
            total -= p.stat().st_size
            p.unlink(missing_ok=True)
            p.with_suffix(".json").unlink(missing_ok=True)

    def stats(self) -> dict:
        """Counts for ``focus tracks``."""
        tracks = self.tracks()
        now = _now()
        retired = sum(1 for t in tracks if self.retired(t))
        cooling = sum(
            1
            for t in tracks
            if not self.retired(t)
            and (last := self.last_played(t)) is not None
            and now - last < self.cooldown
        )
        return {
            "profile": self.dir.name,
            "tracks": len(tracks),
            "ready": len(tracks) - retired - cooling,
            "cooling_down": cooling,
            "retired": retired,
            "plays": sum(self.plays(t) for t in tracks),
            "megabytes": round(sum(t.stat().st_size for t in tracks) / 1024**2, 1),
        }

"""Append-only log of focus sessions, stored as JSON Lines.

One line per session (or per pomodoro block), written when it ends. The file
lives at ``$FOCUS_LOG_PATH``, else ``$XDG_DATA_HOME/focus/sessions.jsonl``, else
``~/.local/share/focus/sessions.jsonl``, so other tools can read it directly.
"""

import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

LYRIA_35_USD_PER_TRACK = 0.08  # Gemini API paid tier, checked 2026-10-09


@dataclass
class SessionRecord:
    """One finished session or pomodoro block."""

    started_at: str  # ISO 8601, local time with UTC offset
    ended_at: str
    profile: str
    kind: str  # "focus" (plain session) | "work" | "break" (pomodoro blocks)
    planned_seconds: int | None  # None for open-ended sessions
    audio_seconds: float  # music actually streamed (pauses don't count)
    outcome: str  # completed | quit | interrupted | ended | error
    engine: (
        str  # realtime | lyria-3.5 | synth  (records before 2026-10-09 say "lyria" for realtime)
    )
    modulation_freq: float
    modulation_depth: float
    fallback_reason: str | None = None
    cycle: int | None = None  # pomodoro block number, 1-based
    cycles: int | None = None  # pomodoro blocks planned
    paid_requests: int | None = None  # Lyria 3.5 tracks generated (billed)


def now_iso() -> str:
    """Current local time as an ISO 8601 string with offset, to the second."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log_path() -> Path:
    """Where the session log lives (see module docstring)."""
    override = os.environ.get("FOCUS_LOG_PATH")
    if override:
        return Path(override).expanduser()
    data_home = os.environ.get("XDG_DATA_HOME") or "~/.local/share"
    return Path(data_home).expanduser() / "focus" / "sessions.jsonl"


def append_record(record: SessionRecord, path: Path | None = None) -> Path:
    """Append one record to the log, creating the file if needed."""
    path = path or log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record)) + "\n")
    return path


def read_records(path: Path | None = None) -> list[dict]:
    """Read every record from the log, skipping lines that don't parse."""
    path = path or log_path()
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def summarize(records: list[dict], days: int = 7, today: date | None = None) -> dict:
    """Roll records up into per-day totals for the last ``days`` days.

    Focus time counts plain sessions and pomodoro work blocks; break blocks are
    reported separately. Days are calendar dates of each session's start, in
    the timezone it was recorded in.
    """
    today = today or date.today()
    first = today - timedelta(days=days - 1)
    per_day: dict[date, dict] = {}
    for offset in range(days):
        d = first + timedelta(days=offset)
        per_day[d] = {
            "date": d.isoformat(),
            "focus_minutes": 0.0,
            "break_minutes": 0.0,
            "sessions": 0,
            "completed_work_blocks": 0,
            "by_profile": defaultdict(float),
            "by_engine": defaultdict(float),
            "paid_tracks": 0,
        }

    for r in records:
        try:
            d = datetime.fromisoformat(r["started_at"]).date()
            minutes = float(r.get("audio_seconds", 0)) / 60.0
        except (KeyError, TypeError, ValueError):
            continue
        day = per_day.get(d)
        if day is None:
            continue
        day["paid_tracks"] += int(r.get("paid_requests") or 0)
        if r.get("kind") == "break":
            day["break_minutes"] += minutes
            continue
        day["focus_minutes"] += minutes
        day["sessions"] += 1
        day["by_profile"][r.get("profile", "?")] += minutes
        engine = r.get("engine", "?")
        day["by_engine"]["realtime" if engine == "lyria" else engine] += minutes  # old label
        if r.get("kind") == "work" and r.get("outcome") == "completed":
            day["completed_work_blocks"] += 1

    rows = []
    for d in sorted(per_day):
        day = per_day[d]
        day["focus_minutes"] = round(day["focus_minutes"], 1)
        day["break_minutes"] = round(day["break_minutes"], 1)
        day["by_profile"] = {k: round(v, 1) for k, v in sorted(day["by_profile"].items())}
        day["by_engine"] = {k: round(v, 1) for k, v in sorted(day["by_engine"].items())}
        day["est_cost_usd"] = round(day["paid_tracks"] * LYRIA_35_USD_PER_TRACK, 2)
        rows.append(day)
    return {
        "from": first.isoformat(),
        "to": today.isoformat(),
        "days": rows,
        "total_focus_minutes": round(sum(r["focus_minutes"] for r in rows), 1),
        "total_sessions": sum(r["sessions"] for r in rows),
        "total_paid_tracks": sum(r["paid_tracks"] for r in rows),
        "total_est_cost_usd": round(sum(r["est_cost_usd"] for r in rows), 2),
    }


def format_minutes(minutes: float) -> str:
    """Render minutes as e.g. '1h 05m' or '25m'."""
    total = int(round(minutes))
    hours, mins = divmod(total, 60)
    return f"{hours}h {mins:02d}m" if hours else f"{mins}m"


def format_summary(summary: dict) -> str:
    """Human-readable table for ``focus log``."""
    lines = [f"\n📒 Focus log, {summary['from']} to {summary['to']}\n"]
    for day in summary["days"]:
        d = date.fromisoformat(day["date"])
        if not day["sessions"] and not day["break_minutes"]:
            lines.append(f"  {day['date']}  {d:%a}   —")
            continue
        parts = [f"{format_minutes(day['focus_minutes'])} focus"]
        if day["break_minutes"]:
            parts.append(f"{format_minutes(day['break_minutes'])} break")
        if day["completed_work_blocks"]:
            parts.append(f"{day['completed_work_blocks']} 🍅")
        if day["paid_tracks"]:
            parts.append(f"{day['paid_tracks']} new tracks ≈ ${day['est_cost_usd']:.2f}")
        profiles = ", ".join(f"{k} {format_minutes(v)}" for k, v in day["by_profile"].items())
        if profiles:
            parts.append(profiles)
        lines.append(f"  {day['date']}  {d:%a}   " + "  ·  ".join(parts))
    lines.append(
        f"\n  Total: {format_minutes(summary['total_focus_minutes'])} focus "
        f"across {summary['total_sessions']} session(s)\n"
    )
    return "\n".join(lines)

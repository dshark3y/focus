"""Tests for pomodoro planning/flow and the session log."""

import json
from datetime import date

import pytest
from click.testing import CliRunner

from focus import cli
from focus.cli import main, parse_pomodoro, pomodoro_blocks
from focus.profiles import BREAK_PROFILE, PROFILES, get_profile
from focus.session_log import (
    SessionRecord,
    append_record,
    format_minutes,
    read_records,
    summarize,
)
from focus.ui.notify import applescript_notification


def _record(started_at, kind="focus", seconds=600.0, outcome="completed", profile="deep-work"):
    return SessionRecord(
        started_at=started_at,
        ended_at=started_at,
        profile=profile,
        kind=kind,
        planned_seconds=None,
        audio_seconds=seconds,
        outcome=outcome,
        engine="lyria",
        modulation_freq=18.0,
        modulation_depth=0.15,
    )


class TestParsePomodoro:
    def test_parses_minutes_to_seconds(self):
        assert parse_pomodoro("50/10") == (3000, 600)
        assert parse_pomodoro(" 25 / 5 ") == (1500, 300)

    @pytest.mark.parametrize("spec", ["50", "50-10", "a/b", "0/5", "25/0"])
    def test_rejects_bad_specs(self, spec):
        with pytest.raises(ValueError):
            parse_pomodoro(spec)


class TestPomodoroBlocks:
    def test_alternates_and_has_no_trailing_break(self):
        assert pomodoro_blocks(60, 30, 3) == [
            ("work", 1, 60),
            ("break", 1, 30),
            ("work", 2, 60),
            ("break", 2, 30),
            ("work", 3, 60),
        ]

    def test_single_cycle_is_one_work_block(self):
        assert pomodoro_blocks(60, 30, 1) == [("work", 1, 60)]


class TestRunPomodoro:
    def _run(self, monkeypatch, outcomes):
        calls, notes = [], []
        monkeypatch.setattr("focus.ui.notify.notify", lambda t, m: notes.append(m))

        def fake_run(profile, seconds, **block):
            calls.append((profile.name, block["kind"], block["cycle"], seconds))
            return outcomes.pop(0)

        work = get_profile("deep-work")
        cli._run_pomodoro(work, pomodoro_blocks(60, 30, 2), 2, fake_run, notify=True)
        return calls, notes

    def test_plays_work_with_profile_and_breaks_with_break_profile(self, monkeypatch):
        calls, notes = self._run(monkeypatch, ["completed"] * 3)
        assert calls == [
            ("deep-work", "work", 1, 60),
            ("break", "break", 1, 30),
            ("deep-work", "work", 2, 60),
        ]
        assert len(notes) == 3
        assert "Pomodoro done" in notes[-1]

    def test_stops_when_a_block_is_quit(self, monkeypatch):
        calls, notes = self._run(monkeypatch, ["completed", "quit"])
        assert len(calls) == 2
        assert len(notes) == 1  # only the end-of-work-block notification

    def test_break_profile_has_no_entrainment_and_is_not_listed(self):
        assert BREAK_PROFILE.modulation_depth == 0.0
        assert "break" not in PROFILES


class TestPomodoroCliValidation:
    def test_rejects_pomodoro_with_duration(self):
        result = CliRunner().invoke(
            main, ["start", "--mock", "--pomodoro", "25/5", "--duration", "600"]
        )
        assert result.exit_code == 1
        assert "--pomodoro" in result.output

    def test_rejects_malformed_pomodoro(self):
        result = CliRunner().invoke(main, ["start", "--mock", "--pomodoro", "25"])
        assert result.exit_code == 1


class TestSessionLog:
    def test_append_and_read_round_trip(self, tmp_path):
        path = tmp_path / "nested" / "sessions.jsonl"
        append_record(_record("2026-10-09T09:00:00+00:00"), path)
        append_record(_record("2026-10-09T10:00:00+00:00", kind="break"), path)
        path.write_text(path.read_text() + "not json\n")
        records = read_records(path)
        assert [r["kind"] for r in records] == ["focus", "break"]

    def test_summary_separates_focus_and_break_and_counts_blocks(self, tmp_path):
        path = tmp_path / "s.jsonl"
        append_record(_record("2026-10-09T09:00:00+00:00", kind="work", seconds=1500), path)
        append_record(_record("2026-10-09T09:25:00+00:00", kind="break", seconds=300), path)
        append_record(
            _record("2026-10-09T09:30:00+00:00", kind="work", seconds=600, outcome="quit"), path
        )
        append_record(_record("2026-10-08T20:00:00+00:00", profile="light-study"), path)
        append_record(_record("2026-09-01T20:00:00+00:00"), path)  # outside window

        summary = summarize(read_records(path), days=2, today=date(2026, 10, 9))
        yesterday, today = summary["days"]
        assert today["focus_minutes"] == 35.0
        assert today["break_minutes"] == 5.0
        assert today["completed_work_blocks"] == 1
        assert today["sessions"] == 2
        assert yesterday["by_profile"] == {"light-study": 10.0}
        assert summary["total_focus_minutes"] == 45.0
        assert today["by_engine"] == {"realtime": 35.0}  # old "lyria" label mapped

    def test_summary_counts_paid_tracks_and_cost(self, tmp_path):
        path = tmp_path / "s.jsonl"
        rec = _record("2026-10-09T09:00:00+00:00")
        rec.engine, rec.paid_requests = "lyria-3.5", 3
        append_record(rec, path)
        summary = summarize(read_records(path), days=1, today=date(2026, 10, 9))
        assert summary["days"][0]["paid_tracks"] == 3
        assert summary["days"][0]["est_cost_usd"] == 0.24
        assert summary["total_est_cost_usd"] == 0.24

    def test_format_minutes(self):
        assert format_minutes(25) == "25m"
        assert format_minutes(65) == "1h 05m"

    def test_log_command_json(self, tmp_path, monkeypatch):
        path = tmp_path / "s.jsonl"
        monkeypatch.setenv("FOCUS_LOG_PATH", str(path))
        append_record(_record(f"{date.today().isoformat()}T09:00:00+00:00"), path)
        result = CliRunner().invoke(main, ["log", "--days", "1", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["log_path"] == str(path)
        assert data["total_focus_minutes"] == 10.0

    def test_log_command_handles_missing_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FOCUS_LOG_PATH", str(tmp_path / "missing.jsonl"))
        result = CliRunner().invoke(main, ["log"])
        assert result.exit_code == 0
        assert "Total: 0m focus" in result.output


class TestNotify:
    def test_applescript_escapes_quotes(self):
        script = applescript_notification('Fo"cus', "Break over.")
        assert 'with title "Fo\\"cus"' in script
        assert 'display notification "Break over."' in script


class TestEngineSwitching:
    def test_e_key_requests_switch_and_status_shows_engine(self):
        from focus.ui.transport import PlaybackState, format_status_line

        state = PlaybackState(profile_name="deep-work", engine_label="lyria-3.5")
        state.handle_key(b"e")
        assert state.engine_switch_requested
        assert "lyria-3.5" in format_status_line(state)

    def test_default_engine_from_env(self, monkeypatch):
        monkeypatch.setenv("FOCUS_ENGINE", "lyria-3.5")
        assert cli.default_engine() == "lyria-3.5"
        monkeypatch.setenv("FOCUS_ENGINE", "bogus")
        assert cli.default_engine() == "realtime"

    def test_launcher_e_toggles_engine(self, monkeypatch):
        import io

        from focus.ui import launcher

        keys = iter([b"e", b"\r"])
        monkeypatch.setattr(launcher, "_getch", lambda: next(keys))
        monkeypatch.setattr(launcher.sys, "stdout", io.StringIO())
        name, engine = launcher.run_launcher(engine="realtime")
        assert engine == "lyria-3.5"

    def test_tracks_command(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FOCUS_CACHE_DIR", str(tmp_path))
        result = CliRunner().invoke(main, ["tracks"])
        assert "empty" in result.output
        (tmp_path / "deep-work").mkdir()
        (tmp_path / "deep-work" / "a.wav").write_bytes(b"RIFF0000")
        result = CliRunner().invoke(main, ["tracks"])
        assert "deep-work" in result.output and "1 tracks" in result.output

"""Desktop notifications for pomodoro transitions (macOS only, best-effort)."""

import json
import subprocess
import sys


def applescript_notification(title: str, message: str) -> str:
    """Build the AppleScript for a notification with a chime.

    JSON string literals double as AppleScript ones for the characters that
    matter here (quotes and backslashes are escaped the same way).
    """
    return (
        f"display notification {json.dumps(message, ensure_ascii=False)} "
        f'with title {json.dumps(title, ensure_ascii=False)} sound name "Glass"'
    )


def notify(title: str, message: str) -> None:
    """Show a notification; silently does nothing off macOS or on failure."""
    if sys.platform != "darwin":
        return
    try:
        subprocess.run(
            ["osascript", "-e", applescript_notification(title, message)],
            check=False,
            capture_output=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        pass

"""Command-line interface for Focus music generator."""

import asyncio
import json
import os
import re
import sys
from dataclasses import replace

import click

from focus.dsp.dynamics import LimiterState, apply_limiter
from focus.dsp.entrainment import ModulationState, apply_entrainment, apply_fade_out
from focus.dsp.spatial import ReverbState, apply_reverb, apply_stereo_widening
from focus.generation.lyria_client import (
    LyriaConfig,
    create_client,
    resolve_api_key,
)
from focus.profiles import BREAK_PROFILE, FocusProfile, get_profile, list_profiles
from focus.session_log import (
    SessionRecord,
    append_record,
    format_summary,
    log_path,
    now_iso,
    read_records,
    summarize,
)
from focus.ui.transport import KeyboardController, PlaybackState, StatusLine

# Spectrum visualizer is optional; imported lazily in _run_session so the CLI
# still works if numpy/scipy are missing (guarded there).

# Check for optional dependencies
try:
    import numpy as np
    import sounddevice as sd

    AUDIO_AVAILABLE = True
except (ImportError, OSError):
    np = None
    sd = None
    AUDIO_AVAILABLE = False


@click.group(invoke_without_command=True)
@click.version_option(version="0.1.0")
@click.pass_context
def main(ctx):
    """Focus - Neural entrainment music generator.

    Generate focus-enhancing music using AI (Google Lyria) with
    neural entrainment modulation for improved concentration.

    Run `focus` with no arguments in a terminal to pick a profile interactively.

    Quick Usage:\n
        focus start --profile deep-work \n
        focus start --duration 600  # 10 minute session \n
        focus start --output session.wav \n
    """
    if ctx.invoked_subcommand is not None:
        return

    # Bare invocation: drop into the interactive picker when attached to a
    # terminal; otherwise (pipes, CI) fall back to the usual help text.
    if sys.stdin.isatty() and sys.stdout.isatty():
        from focus.ui.launcher import run_launcher

        choice = run_launcher(engine=default_engine())
        if choice:
            profile, engine = choice
            launch_session(profile=profile, engine=engine)
    else:
        click.echo(ctx.get_help())


ENGINE_CHOICES = ("realtime", "lyria-3.5", "offline")


def default_engine() -> str:
    """Engine to use when none is given: $FOCUS_ENGINE if valid, else realtime."""
    value = os.environ.get("FOCUS_ENGINE", "realtime")
    return value if value in ENGINE_CHOICES else "realtime"


def library_available(profile_name: str) -> bool:
    """Whether offline mode has anything to play (this profile, else any profile)."""
    from focus.generation.track_library import TrackCache

    return bool(TrackCache(profile_name).tracks() or TrackCache(None).tracks())


def engine_available(engine: str, profile_name: str) -> bool:
    """Whether a session could run on ``engine`` right now."""
    if engine == "offline":
        return library_available(profile_name)
    return resolve_api_key() is not None


def next_engine(current: str, profile_name: str) -> str | None:
    """The engine [e] switches to: the next available one in ENGINE_CHOICES order."""
    i = ENGINE_CHOICES.index(current)
    for step in range(1, len(ENGINE_CHOICES)):
        candidate = ENGINE_CHOICES[(i + step) % len(ENGINE_CHOICES)]
        if engine_available(candidate, profile_name):
            return candidate
    return None


@main.command("profiles")
def show_profiles():
    """List available focus profiles."""
    profiles = list_profiles()

    click.echo("\n🎧 Available Focus Profiles\n")
    click.echo("-" * 60)

    for p in profiles:
        click.echo(f"\n  {click.style(p.name, fg='cyan', bold=True)}")
        click.echo(f"  {p.description}")
        click.echo(f"  Modulation: {p.modulation_freq:.0f} Hz @ {p.modulation_depth:.0%} depth")

    click.echo("\n" + "-" * 60)
    click.echo("\nUsage: focus start --profile <name>\n")


@main.command("start")
@click.option(
    "--profile",
    "-p",
    type=str,
    default="deep-work",
    help="Focus profile to use (see 'focus profiles' for list)",
)
@click.option(
    "--frequency",
    "-f",
    type=click.FloatRange(min=0.0, max=100.0, min_open=True),
    default=None,
    help="Override modulation frequency (Hz)",
)
@click.option(
    "--depth",
    "-d",
    type=click.FloatRange(min=0.0, max=1.0),
    default=None,
    help="Override modulation depth (0.0-1.0; 0 turns modulation off)",
)
@click.option(
    "--band",
    type=float,
    default=None,
    help="Override modulated low-band cutoff in Hz (0 = full-spectrum modulation)",
)
@click.option(
    "--prompt",
    type=str,
    default=None,
    help="Custom Lyria prompt (overrides profile)",
)
@click.option(
    "--mock",
    is_flag=True,
    default=False,
    help="Use mock audio generator (no API key needed)",
)
@click.option(
    "--duration",
    type=int,
    default=None,
    help="Session duration in seconds (minimum: 60, default: unlimited)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Save audio to WAV file (in addition to playback)",
)
@click.option(
    "--reverb/--no-reverb",
    is_flag=True,
    default=True,
    help="Enable/disable spatial reverb (default: enabled)",
)
@click.option(
    "--stereo-width",
    type=float,
    default=1.2,
    help="Stereo width enhancement (1.0=original, >1.0=wider)",
)
@click.option(
    "--limiter/--no-limiter",
    is_flag=True,
    default=True,
    help="Enable/disable safety limiter (default: enabled)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Show detailed debug output",
)
@click.option(
    "--track-duration",
    type=int,
    default=9,
    help="Duration of each track before rotation (1-9 minutes, default: 9)",
)
@click.option(
    "--spectrum/--no-spectrum",
    is_flag=True,
    default=True,
    help="Live audio spectrum visualizer (interactive terminal only, default: on)",
)
@click.option(
    "--engine",
    type=click.Choice(ENGINE_CHOICES),
    default="realtime",
    envvar="FOCUS_ENGINE",
    show_default=True,
    help="realtime: endless live stream (Lyria RealTime). lyria-3.5: generated "
    "~2.5 min tracks blended together, reused from a library ($0.08 per new "
    "track). offline: saved tracks only, no network. Default from $FOCUS_ENGINE; "
    "press [e] mid-session to switch",
)
@click.option(
    "--offline",
    "--cached-only",
    "offline",
    is_flag=True,
    default=False,
    help="Same as --engine offline: play saved tracks only, ignoring the play cap "
    "and cooldown (no network, no cost, plays don't count toward the cap)",
)
@click.option(
    "--max-plays",
    type=click.IntRange(min=1),
    default=10,
    show_default=True,
    help="Lyria 3.5: retire a library track after this many plays",
)
@click.option(
    "--cooldown-hours",
    type=click.FloatRange(min=0),
    default=12.0,
    show_default=True,
    help="Lyria 3.5: don't reuse a library track within this many hours of its last play",
)
@click.option(
    "--pomodoro",
    type=str,
    default=None,
    metavar="WORK/BREAK",
    help="Pomodoro cycles in minutes, e.g. 50/10 or 25/5 (breaks play calm music)",
)
@click.option(
    "--cycles",
    type=click.IntRange(min=1),
    default=4,
    show_default=True,
    help="Number of work blocks in a pomodoro run",
)
@click.option(
    "--log/--no-log",
    "log_session",
    default=True,
    help="Record the session in the focus log (see 'focus log', default: on)",
)
@click.option(
    "--notify/--no-notify",
    default=True,
    help="macOS notification at each pomodoro transition (default: on)",
)
def start_session(
    profile: str,
    frequency: float | None,
    depth: float | None,
    band: float | None,
    prompt: str | None,
    mock: bool,
    duration: int | None,
    output: str | None,
    reverb: bool,
    stereo_width: float,
    limiter: bool,
    verbose: bool,
    track_duration: int,
    spectrum: bool,
    pomodoro: str | None,
    cycles: int,
    log_session: bool,
    notify: bool,
    engine: str,
    offline: bool,
    max_plays: int,
    cooldown_hours: float,
):
    """Start a focus music session.

    Examples:

        focus start --profile deep-work

        focus start -p light-study --duration 300

        focus start --frequency 16 --depth 0.3 --mock

        focus start -p deep-work --pomodoro 50/10 --cycles 3

        focus start -p light-study --engine lyria-3.5
    """
    launch_session(
        profile=profile,
        frequency=frequency,
        depth=depth,
        band=band,
        prompt=prompt,
        mock=mock,
        duration=duration,
        output=output,
        reverb=reverb,
        stereo_width=stereo_width,
        limiter=limiter,
        verbose=verbose,
        track_duration=track_duration,
        spectrum=spectrum,
        pomodoro=pomodoro,
        cycles=cycles,
        log_session=log_session,
        notify=notify,
        engine="offline" if offline else engine,
        max_plays=max_plays,
        cooldown_hours=cooldown_hours,
    )


def parse_pomodoro(spec: str) -> tuple[int, int]:
    """Parse 'WORK/BREAK' minutes (e.g. '50/10') into seconds.

    Raises:
        ValueError: if the spec is malformed or either block is under a minute.
    """
    match = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", spec)
    if not match:
        raise ValueError(
            f"--pomodoro must look like WORK/BREAK in minutes, e.g. 50/10 (got {spec!r})"
        )
    work, brk = int(match.group(1)), int(match.group(2))
    if work < 1 or brk < 1:
        raise ValueError("--pomodoro work and break blocks must each be at least 1 minute")
    return work * 60, brk * 60


def pomodoro_blocks(
    work_seconds: int, break_seconds: int, cycles: int
) -> list[tuple[str, int, int]]:
    """Block plan as (kind, cycle, seconds): work, break, ..., work (no trailing break)."""
    blocks: list[tuple[str, int, int]] = []
    for cycle in range(1, cycles + 1):
        blocks.append(("work", cycle, work_seconds))
        if cycle < cycles:
            blocks.append(("break", cycle, break_seconds))
    return blocks


def apply_overrides(
    profile: FocusProfile,
    frequency: float | None = None,
    depth: float | None = None,
    band: float | None = None,
    prompt: str | None = None,
) -> FocusProfile:
    """Return ``profile`` with any CLI overrides applied.

    Compares to None so explicit zeros apply (``--depth 0`` turns modulation
    off), and uses ``replace`` so untouched fields such as the intro/outro
    prompts survive.
    """
    changes: dict = {}
    if frequency is not None:
        changes["modulation_freq"] = frequency
    if depth is not None:
        changes["modulation_depth"] = depth
    if band is not None:
        # 0 (or negative) disables band-limiting -> full-spectrum modulation
        changes["modulation_band_hz"] = None if band <= 0 else band
    if prompt:
        changes["prompt"] = prompt
    return replace(profile, **changes) if changes else profile


def launch_session(
    profile: str,
    frequency: float | None = None,
    depth: float | None = None,
    band: float | None = None,
    prompt: str | None = None,
    mock: bool = False,
    duration: int | None = None,
    output: str | None = None,
    reverb: bool = True,
    stereo_width: float = 1.2,
    limiter: bool = True,
    verbose: bool = False,
    track_duration: int = 9,
    spectrum: bool = True,
    pomodoro: str | None = None,
    cycles: int = 4,
    log_session: bool = True,
    notify: bool = True,
    engine: str = "realtime",
    max_plays: int = 10,
    cooldown_hours: float = 12.0,
):
    """Resolve a profile, apply overrides, and run a session.

    Shared entry point for both the ``start`` command and the bare-``focus``
    interactive launcher.
    """
    if duration is not None and duration < 60:
        click.echo(
            "Error: Duration must be at least 60 seconds to allow for intro/outro phases.",
            err=True,
        )
        sys.exit(1)

    blocks = None
    if pomodoro:
        if duration is not None or output:
            click.echo("Error: --pomodoro can't be combined with --duration or --output.", err=True)
            sys.exit(1)
        try:
            work_seconds, break_seconds = parse_pomodoro(pomodoro)
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
        blocks = pomodoro_blocks(work_seconds, break_seconds, cycles)

    try:
        focus_profile = get_profile(profile)
    except KeyError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if engine == "offline" and not mock and not library_available(profile):
        click.echo(
            "Error: no saved tracks for offline mode yet. Play some sessions with "
            "--engine lyria-3.5 first.",
            err=True,
        )
        sys.exit(1)

    # Fail fast, before the terminal is put into key-reading mode
    if not mock and engine != "offline" and not resolve_api_key():
        click.echo(
            "Error: no API key. Set GOOGLE_API_KEY (or GEMINI_API_KEY), "
            "or use --mock to try it without one.",
            err=True,
        )
        sys.exit(1)

    focus_profile = apply_overrides(focus_profile, frequency, depth, band, prompt)

    click.echo(f"\n🎯 Starting focus session: {click.style(focus_profile.name, bold=True)}")
    click.echo(
        f"   Modulation: {focus_profile.modulation_freq:.0f} Hz @ "
        f"{focus_profile.modulation_depth:.0%}"
    )
    click.echo(
        f"   Effects: Reverb={'ON' if reverb else 'OFF'}, "
        f"Width={stereo_width}, Limiter={'ON' if limiter else 'OFF'}"
    )
    if verbose:
        click.echo(
            f"   BPM: {focus_profile.bpm}, Density: {focus_profile.density}, "
            f"Brightness: {focus_profile.brightness}"
        )
        click.echo(f"   Prompt: {focus_profile.prompt[:60]}...")
    if mock:
        pass
    elif engine == "offline":
        click.echo("   Engine: offline (saved tracks, no network, no cost)")
    elif engine == "lyria-3.5":
        click.echo(
            f"   Engine: Lyria 3.5 (library reuse free, new tracks $0.08; "
            f"cap {max_plays} plays, {cooldown_hours:g}h cooldown)"
        )
    else:
        click.echo("   Engine: realtime (falls back to saved tracks if the connection drops)")
    if duration:
        click.echo(f"   Duration: {duration} seconds")
    if blocks:
        click.echo(
            f"   Pomodoro: {cycles} × {work_seconds // 60} min work, "
            f"{break_seconds // 60} min breaks"
        )
    if output:
        click.echo(f"   Output: {output}")
    if sys.stdin.isatty() and sys.stdout.isatty():
        controls = "[space] pause  [n] next  [↑↓] volume  [?] help  [q] quit"
        if not mock:
            controls = controls.replace("[n] next", "[n] next  [e] switch engine")
        click.echo(f"\n   Controls: {controls}\n")
    else:
        click.echo("\n   Press Ctrl+C to stop\n")

    def run(block_profile: FocusProfile, block_duration: int | None, **block) -> str:
        return asyncio.run(
            _run_session(
                block_profile,
                mock,
                block_duration,
                output,
                reverb=reverb,
                stereo_width=stereo_width,
                limiter=limiter,
                verbose=verbose,
                track_duration=track_duration,
                spectrum=spectrum,
                log_session=log_session and not mock,
                engine=engine,
                max_plays=max_plays,
                cooldown_hours=cooldown_hours,
                **block,
            )
        )

    try:
        if blocks is None:
            run(focus_profile, duration)
        else:
            _run_pomodoro(focus_profile, blocks, cycles, run, notify=notify)
    except KeyboardInterrupt:
        click.echo("\n\n🛑 Session stopped by user")

    if output:
        click.echo(f"💾 Audio saved to: {output}")
    click.echo("👋 Session ended. Stay focused!\n")


def _run_pomodoro(
    work_profile: FocusProfile,
    blocks: list[tuple[str, int, int]],
    cycles: int,
    run,
    notify: bool = True,
) -> None:
    """Play work and break blocks in turn; stop early if a block is quit."""
    from focus.ui.notify import notify as send_notification

    for index, (kind, cycle, seconds) in enumerate(blocks):
        minutes = seconds // 60
        if kind == "work":
            click.echo(f"🍅 Work block {cycle}/{cycles} · {minutes} min")
            block_profile = work_profile
        else:
            click.echo(f"☕ Break · {minutes} min")
            block_profile = BREAK_PROFILE

        outcome = run(block_profile, seconds, kind=kind, cycle=cycle, cycles=cycles)
        if outcome != "completed":
            click.echo(f"   Pomodoro stopped during {kind} block {cycle}/{cycles}.")
            return

        is_last = index == len(blocks) - 1
        if not notify:
            continue
        if is_last:
            send_notification("Focus", f"Pomodoro done: {cycles} work blocks complete")
        elif kind == "work":
            next_minutes = blocks[index + 1][2] // 60
            send_notification(
                "Focus", f"Work block {cycle}/{cycles} done. {next_minutes} min break."
            )
        else:
            send_notification("Focus", f"Break over. Work block {cycle + 1}/{cycles}.")

    click.echo(f"🎉 Pomodoro complete: {cycles} work blocks")


async def _run_session(
    profile: FocusProfile,
    use_mock: bool,
    duration: int | None,
    output_path: str | None = None,
    reverb: bool = True,
    stereo_width: float = 1.2,
    limiter: bool = True,
    verbose: bool = False,
    track_duration: int = 9,
    spectrum: bool = True,
    kind: str = "focus",
    cycle: int | None = None,
    cycles: int | None = None,
    log_session: bool = False,
    engine: str = "realtime",
    max_plays: int = 10,
    cooldown_hours: float = 12.0,
) -> str:
    """Run the audio generation session.

    Returns how it ended: completed (duration reached), quit, interrupted,
    ended (stream stopped on its own) or error.
    """
    try:
        from focus.audio.output import AudioOutput, FileAudioOutput
    except ImportError:
        if not use_mock:
            click.echo("Error: sounddevice not available", err=True)
            return "error"

    sample_rate = 48000

    # Clamp track duration to valid range (1-9 minutes)
    track_duration_seconds = max(60, min(9 * 60, track_duration * 60))

    def build_config(phase: str) -> LyriaConfig:
        """Build a Lyria config for the given musical phase."""
        bpm = profile.bpm or 120
        density = profile.density or 0.5
        brightness = profile.brightness or 0.5
        if phase == "intro" and profile.intro_prompt:
            return LyriaConfig(
                prompt=f"{profile.intro_prompt}, {profile.prompt}",
                bpm=bpm,
                density=max(0.1, density - 0.2),  # Start with lower density
                brightness=brightness,
            )
        if phase == "outro" and profile.outro_prompt:
            return LyriaConfig(
                prompt=f"{profile.outro_prompt}, {profile.prompt}",
                bpm=bpm,
                density=density,
                brightness=brightness,
            )
        return LyriaConfig(prompt=profile.prompt, bpm=bpm, density=density, brightness=brightness)

    # Mutable so [e] can switch engines mid-session
    current_engine = engine
    paid_requests_total = 0
    switch_to: str | None = None  # engine to change to at the next reconnect
    offline_fallback_reason: str | None = None  # why we dropped to saved tracks
    library_ok: bool | None = None  # memo: can offline mode play anything?

    def make_client(phase: str):
        if current_engine in ("lyria-3.5", "offline") and not use_mock:
            from focus.generation.track_client import TrackClient

            return TrackClient(
                build_config(phase),
                profile=profile.name,
                verbose=verbose,
                offline=current_engine == "offline",
                main_prompt=profile.prompt,
                max_plays=max_plays,
                cooldown_hours=cooldown_hours,
            )
        return create_client(
            build_config(phase),
            use_mock=use_mock,
            verbose=verbose,
            session_duration=track_duration_seconds,
        )

    if not AUDIO_AVAILABLE:
        client = make_client("main")
        click.echo("⚠️  sounddevice not available, running in test mode")
        await client.connect()
        chunk_count = 0
        async for chunk in client.generate_stream():
            chunk_count += 1
            click.echo(
                f"  📦 Chunk {chunk_count}: shape={chunk.shape}, "
                f"range=[{chunk.min():.2f}, {chunk.max():.2f}]"
            )
            if chunk_count >= 10:
                break
        await client.stop()
        return "ended"

    current_phase = "intro" if duration and profile.intro_prompt else "main"
    started_at = now_iso()
    outcome = "ended"

    # Connect to generator before touching the terminal, so a connection error
    # can't leave it in cbreak mode with the display half drawn.
    client = make_client(current_phase)
    await client.connect()
    if verbose:
        click.echo("   ✓ Connected to audio generator")

    # Interactive transport controls (only attached to a real terminal)
    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    state = None
    keyboard = None
    status_line = None
    display = None
    spectrum_task = None
    if interactive:
        state = PlaybackState(
            profile_name=(
                f"{profile.name} · {kind} {cycle}/{cycles}" if cycle is not None else profile.name
            ),
            modulation_freq=profile.modulation_freq,
            modulation_depth=profile.modulation_depth,
            status="connecting",
            engine_label="" if use_mock else engine,
        )
        if not verbose:
            # The live status line and -v logging both want the bottom region;
            # when verbose, the logs already convey state, so skip the display.
            if spectrum:
                # The spectrum display owns the bottom rows and draws the same
                # status text as its last row (via format_status_line).
                try:
                    from focus.analysis.realtime import SpectrumAnalyzer
                    from focus.ui.spectrum import SpectrumDisplay

                    analyzer = SpectrumAnalyzer(sample_rate=sample_rate)
                    display = SpectrumDisplay(state, analyzer)
                    display.start()
                    display.render()
                except ImportError:
                    display = None
            if display is None:
                status_line = StatusLine()
                status_line.start()
                status_line.render(state)

        # Redraw immediately on each keypress so pause / volume / help toggles
        # are reflected instantly instead of on the next audio chunk.
        def _on_key():
            if display is not None:
                display.render()
            elif status_line is not None:
                status_line.render(state)

        keyboard = KeyboardController(state, on_change=_on_key)
        try:
            keyboard.start()
        except Exception:
            keyboard.stop()
            if display is not None:
                display.finish()
            if status_line is not None:
                status_line.finish()
            raise

    # Initialize DSP state
    mod_state = ModulationState()
    reverb_state = ReverbState() if reverb else None
    limiter_state = LimiterState(ceiling_linear=0.989) if limiter else None  # -0.1 dBTP

    chunk_count = 0
    total_seconds = 0.0

    # Fade settings (in seconds)
    fade_duration = 5.0
    fade_in_samples_remaining = int(fade_duration * sample_rate)
    fade_out_buffer = []  # Buffer for fade-out when duration is set

    # Phase management for timed sessions (natural musical evolution)
    # Phases: intro -> main -> outro
    intro_duration = 15.0  # seconds for buildup phase
    outro_duration = 30.0  # seconds for wind-down phase
    phase_switched_to_main = current_phase == "main"
    phase_switched_to_outro = False

    if verbose:
        click.echo(f"   🔊 Audio device: {sd.query_devices(sd.default.device[1])['name']}")
        if duration:
            click.echo(
                f"   🎵 Musical phases: intro ({intro_duration}s) → "
                f"main → outro ({outro_duration}s)"
            )

    # Use queue-based output for robust playback
    output = AudioOutput(sample_rate=sample_rate)
    output.start()

    # Optional file output
    file_output = None
    if output_path:
        file_output = FileAudioOutput(filepath=output_path, sample_rate=sample_rate)
        file_output.start()

    session_complete = False
    session_error: Exception | None = None

    try:
        # Drive the spectrum redraw at a fixed frame rate, decoupled from the
        # irregular arrival of audio chunks. Created inside the try so the
        # finally block always cancels it and restores the terminal.
        if display is not None:
            # Feed the visualizer from the playback tap (what's actually heard),
            # now that the output stream exists.
            display.source = output.latest_samples

            async def _spectrum_loop():
                try:
                    while state is None or not state.quit_requested:
                        display.render()
                        await asyncio.sleep(1.0 / display.fps)
                except asyncio.CancelledError:
                    pass

            spectrum_task = asyncio.create_task(_spectrum_loop())
        # Outer loop: each iteration consumes one generator until it ends or an
        # interactive control (pause / next take) asks us to reconnect.
        while not session_complete:
            stream = client.generate_stream()
            reconnect = False

            async for chunk in stream:
                # Live engine lost (e.g. no network) and fell back to the synth:
                # play the saved library instead, if there is one.
                if (
                    not use_mock
                    and current_engine != "offline"
                    and getattr(client, "using_synth", False)
                ):
                    if library_ok is None:
                        library_ok = library_available(profile.name)
                    if library_ok:
                        offline_fallback_reason = getattr(client, "fallback_reason", None)
                        switch_to = "offline"
                        reconnect = True
                        break
                chunk_count += 1
                chunk_seconds = len(chunk) / sample_rate
                total_seconds += chunk_seconds

                if verbose:
                    # Log amplitude to verify signal presence
                    max_amp = np.max(np.abs(chunk))
                    click.echo(
                        f"   📦 Chunk {chunk_count}: {len(chunk)} samples, "
                        f"max_amp={max_amp:.3f}, phase={current_phase}"
                    )

                # Phase transitions for timed sessions
                if duration:
                    # Transition: intro -> main (after intro_duration)
                    if (
                        current_phase == "intro"
                        and total_seconds >= intro_duration
                        and not phase_switched_to_main
                    ):
                        current_phase = "main"
                        phase_switched_to_main = True
                        await client.set_prompt(profile.prompt)
                        if verbose:
                            click.echo("   🎵 Phase transition: intro → main")

                    # Transition: main -> outro (outro_duration before end)
                    time_remaining = duration - total_seconds
                    if (
                        current_phase == "main"
                        and time_remaining <= outro_duration
                        and not phase_switched_to_outro
                    ):
                        if profile.outro_prompt:
                            current_phase = "outro"
                            phase_switched_to_outro = True
                            outro_full_prompt = f"{profile.outro_prompt}, {profile.prompt}"
                            await client.set_prompt(outro_full_prompt)
                            if verbose:
                                click.echo("   🎵 Phase transition: main → outro")

                # Apply neural entrainment
                modulated, mod_state = apply_entrainment(
                    chunk,
                    sample_rate,
                    target_freq=profile.modulation_freq,
                    depth=profile.modulation_depth,
                    state=mod_state,
                    band_cutoff_hz=profile.modulation_band_hz,
                )

                # Apply fade-in to early chunks
                if fade_in_samples_remaining > 0:
                    chunk_samples = len(modulated)
                    if fade_in_samples_remaining >= chunk_samples:
                        # This entire chunk needs fading
                        fade_progress = 1.0 - (
                            fade_in_samples_remaining / (fade_duration * sample_rate)
                        )
                        end_progress = fade_progress + chunk_samples / (fade_duration * sample_rate)
                        t = np.linspace(
                            fade_progress * np.pi / 2,
                            end_progress * np.pi / 2,
                            chunk_samples,
                        )
                        envelope = np.sin(t) ** 2
                        if modulated.ndim == 2:
                            modulated = modulated * envelope[:, np.newaxis]
                        else:
                            modulated = modulated * envelope
                        modulated = modulated.astype(np.float32)
                    else:
                        # Partial fade on this chunk
                        fade_progress = 1.0 - (
                            fade_in_samples_remaining / (fade_duration * sample_rate)
                        )
                        t = np.linspace(
                            fade_progress * np.pi / 2,
                            np.pi / 2,
                            fade_in_samples_remaining,
                        )
                        envelope = np.sin(t) ** 2
                        if modulated.ndim == 2:
                            modulated[:fade_in_samples_remaining] *= envelope[:, np.newaxis]
                        else:
                            modulated[:fade_in_samples_remaining] *= envelope
                        modulated = modulated.astype(np.float32)
                    fade_in_samples_remaining -= chunk_samples

                # --- Phase 3 DSP Chain ---

                # 1. Spatialization (Reverb)
                if reverb:
                    modulated, reverb_state = apply_reverb(
                        modulated, sample_rate, state=reverb_state
                    )

                # 2. Stereo Widening
                if abs(stereo_width - 1.0) > 0.01:
                    modulated = apply_stereo_widening(modulated, width=stereo_width)

                # 4. Dynamics (Limiter)
                if limiter:
                    modulated, limiter_state = apply_limiter(
                        modulated, sample_rate, state=limiter_state
                    )

                # Output gain (volume) is applied inside AudioOutput, post-limiter
                if state is not None:
                    output.set_volume(state.volume)

                # ALWAYS write to real-time output immediately (no buffering delay)
                output.write(modulated)

                # For file output with duration: buffer the last 5 seconds for fade-out
                if file_output:
                    if duration:
                        fade_out_buffer.append(modulated.copy())
                        # Keep only enough buffer for fade-out duration
                        total_buffered = sum(len(c) for c in fade_out_buffer)
                        fade_out_samples = int(fade_duration * sample_rate)
                        while total_buffered > fade_out_samples and len(fade_out_buffer) > 1:
                            old_chunk = fade_out_buffer.pop(0)
                            total_buffered -= len(old_chunk)
                            # Write the old chunk that's no longer in fade zone
                            file_output.write(old_chunk)
                    else:
                        # No duration limit, write immediately to file
                        file_output.write(modulated)

                # Check duration limit
                if duration and total_seconds >= duration:
                    if verbose:
                        click.echo(f"\n   ⏱️  Duration reached ({total_seconds:.1f}s)")
                    # Apply fade-out to file output's buffered chunks
                    if fade_out_buffer and file_output:
                        combined = np.concatenate(fade_out_buffer, axis=0)
                        faded = apply_fade_out(combined, sample_rate, fade_duration)
                        file_output.write(faded)
                        fade_out_buffer.clear()
                    outcome = "completed"
                    session_complete = True
                    break

                # Interactive controls
                if state is not None:
                    state.elapsed_seconds = total_seconds
                    state.buffer_seconds = output.buffer_seconds
                    state.status = (
                        "synth fallback"
                        if getattr(client, "using_synth", False) and not use_mock
                        else "playing"
                    )
                    if status_line is not None:
                        status_line.render(state)
                    if state.quit_requested:
                        outcome = "quit"
                        session_complete = True
                        break
                    resumable = getattr(client, "resumable", False)
                    if state.skip_requested and resumable:
                        # Track engine: blend into the next queued track in place
                        client.skip()
                        state.skip_requested = False
                    if state.engine_switch_requested:
                        state.engine_switch_requested = False
                        target = None if use_mock else next_engine(current_engine, profile.name)
                        if target is not None:
                            switch_to = target
                            reconnect = True
                            break
                    if state.paused or state.skip_requested:
                        reconnect = True
                        break

                # Yield control to event loop to keep UI responsive
                await asyncio.sleep(0)

            # Close the abandoned/finished generator before reconnecting
            try:
                await stream.aclose()
            except Exception:
                pass

            # Swap engines in place ([e], or the automatic drop to offline):
            # same profile and phase, fresh fade-in
            if switch_to is not None and not session_complete:
                target, switch_to = switch_to, None
                if state is not None:
                    state.status = "reconnecting"
                    state.engine_label = target
                    if status_line is not None:
                        status_line.render(state)
                paid_requests_total += getattr(client, "paid_requests", 0) or 0
                await client.stop()
                current_engine = target
                client = make_client(current_phase)
                await client.connect()
                fade_in_samples_remaining = int(fade_duration * sample_rate)
                continue

            # End the session unless an interactive control asked to reconnect
            if session_complete or state is None or not reconnect:
                break

            # Pause: tear the session down (stops burning quota), wait, reconnect

            resumable = getattr(client, "resumable", False)
            if state.paused:
                output.pause()
                if not resumable:
                    await client.stop()
                state.status = "paused"
                if status_line is not None:
                    status_line.render(state)
                while state.paused and not state.quit_requested:
                    await asyncio.sleep(0.15)
                    if status_line is not None:
                        status_line.render(state)
                if state.quit_requested:
                    outcome = "quit"
                    break
                output.resume()
                if resumable:
                    # Keep the queued tracks: carry on from where playback stopped
                    fade_in_samples_remaining = int(fade_duration * sample_rate)
                    continue

            # "Next take": force a fresh generation (same profile/phase)
            state.skip_requested = False
            state.status = "reconnecting"
            if status_line is not None:
                status_line.render(state)
            await client.stop()
            client = make_client(current_phase)
            await client.connect()
            # Fade the new take in to avoid a hard join
            fade_in_samples_remaining = int(fade_duration * sample_rate)

    except asyncio.CancelledError:
        outcome = "interrupted"
    except Exception as e:
        outcome = "error"
        session_error = e
        if verbose:
            import traceback

            traceback.print_exc()
    finally:
        # Restore the terminal before any further output
        if spectrum_task is not None:
            spectrum_task.cancel()
            try:
                await spectrum_task
            except asyncio.CancelledError:
                pass
        if keyboard is not None:
            keyboard.stop()
        if display is not None:
            display.finish()
        if status_line is not None:
            status_line.finish()
        # Flush any remaining buffered audio to file (for Ctrl+C case with duration set)
        if fade_out_buffer and file_output:
            combined = np.concatenate(fade_out_buffer, axis=0)
            faded = apply_fade_out(
                combined, sample_rate, min(fade_duration, len(combined) / sample_rate)
            )
            file_output.write(faded)
        output.flush()  # Flush leftover samples with fade-out
        output.stop()
        if file_output:
            file_output.stop()
        await client.stop()
        # Reported after the terminal is restored so the message stays readable
        if session_error is not None:
            click.echo(f"   ⚠️  Session error: {session_error}", err=True)
        fallback_reason = getattr(client, "fallback_reason", None)
        if offline_fallback_reason:
            click.echo(
                f"   ⚠️  Lost the live stream ({offline_fallback_reason}); "
                "switched to your saved tracks.",
                err=True,
            )
        if log_session and total_seconds > 0:
            record = SessionRecord(
                started_at=started_at,
                ended_at=now_iso(),
                profile=profile.name,
                kind=kind,
                planned_seconds=duration,
                audio_seconds=round(total_seconds, 1),
                outcome=outcome,
                engine=(
                    "synth"
                    if use_mock or getattr(client, "using_synth", False)
                    else getattr(client, "engine_name", "lyria")
                ),
                paid_requests=(
                    paid_requests_total + (getattr(client, "paid_requests", 0) or 0)
                    if current_engine != "realtime" or paid_requests_total
                    else None
                ),
                modulation_freq=profile.modulation_freq,
                modulation_depth=profile.modulation_depth,
                fallback_reason=fallback_reason,
                cycle=cycle,
                cycles=cycles,
            )
            try:
                append_record(record)
            except OSError as e:
                click.echo(f"   ⚠️  Could not write focus log: {e}", err=True)
        if fallback_reason and getattr(client, "using_synth", False) and not use_mock:
            click.echo(
                f"   ⚠️  Lyria was unavailable ({fallback_reason}); "
                "played the fallback synth instead.",
                err=True,
            )
        elif fallback_reason:
            click.echo(
                f"   ⚠️  Lyria 3.5 stopped generating ({fallback_reason}); "
                "switched to your saved tracks.",
                err=True,
            )
        if verbose:
            buffer_info = f", buffer={output.buffer_seconds:.1f}s"
            underrun_msg = (
                f", {output.underrun_count} underruns" if output.underrun_count > 0 else ""
            )
            click.echo(
                f"   ✓ Session ended after {total_seconds:.1f}s "
                f"({chunk_count} chunks{underrun_msg}{buffer_info})"
            )
    return outcome


@main.command("log")
@click.option("--days", type=click.IntRange(min=1), default=7, show_default=True)
@click.option("--json", "as_json", is_flag=True, default=False, help="Machine-readable output")
def show_log(days: int, as_json: bool):
    """Show focus time per day from the session log."""
    summary = summarize(read_records(), days=days)
    if as_json:
        click.echo(json.dumps({"log_path": str(log_path()), **summary}, indent=2))
    else:
        click.echo(format_summary(summary))


@main.command("tracks")
def show_tracks():
    """Show the Lyria 3.5 track library: tracks, plays and what's retired."""
    from focus.generation.track_library import TrackCache, cache_root

    root = cache_root()
    folders = sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []
    click.echo(f"\n🎼 Track library: {root}\n")
    if not folders:
        click.echo("  (empty: tracks appear here after a session with --engine lyria-3.5)\n")
        return
    for folder in folders:
        st = TrackCache(folder.name).stats()
        click.echo(
            f"  {st['profile']:<14} {st['tracks']:>3} tracks · {st['ready']} ready · "
            f"{st['cooling_down']} cooling down · {st['retired']} retired · "
            f"{st['plays']} plays · {st['megabytes']} MB"
        )
    click.echo("")


@main.command("test-audio")
def test_audio():
    """Test audio output with a simple tone."""
    if not AUDIO_AVAILABLE:
        click.echo("Error: sounddevice not available", err=True)
        sys.exit(1)

    click.echo("\n🔊 Testing audio output...")
    click.echo(f"   Default output: {sd.query_devices(sd.default.device[1])['name']}")

    # Generate a 440 Hz test tone
    sample_rate = 48000
    duration = 2.0
    t = np.arange(int(duration * sample_rate)) / sample_rate
    tone = 0.3 * np.sin(2 * np.pi * 440 * t)
    stereo = np.column_stack([tone, tone]).astype(np.float32)

    click.echo("   Playing 440 Hz tone for 2 seconds...")
    try:
        sd.play(stereo, samplerate=sample_rate, blocking=True)
        click.echo("   ✓ Audio test complete!")
    except Exception as e:
        click.echo(f"   ✗ Audio error: {e}", err=True)


@main.command("analyze")
@click.argument("audio_file", type=click.Path(exists=True))
@click.option(
    "--expected-freq",
    "-f",
    type=float,
    default=15.0,
    help="Expected modulation frequency (Hz)",
)
@click.option(
    "--expected-depth",
    "-d",
    type=float,
    default=0.3,
    help="Expected modulation depth",
)
def analyze_audio(audio_file: str, expected_freq: float, expected_depth: float):
    """Analyze an audio file for neural entrainment modulation.

    Verifies that amplitude modulation is present at the expected frequency.
    """
    try:
        import numpy as np
        from scipy.io import wavfile
    except ImportError:
        click.echo("Error: scipy is required for audio analysis", err=True)
        sys.exit(1)

    from focus.analysis.fft import generate_report

    click.echo(f"\n📊 Analyzing: {audio_file}\n")

    sample_rate, audio = wavfile.read(audio_file)

    # Convert to float
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0

    report = generate_report(audio, sample_rate, expected_freq, expected_depth)
    click.echo(report)


if __name__ == "__main__":
    main()

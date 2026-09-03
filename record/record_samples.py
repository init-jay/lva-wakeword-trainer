#!/usr/bin/env python3
"""
Record real voice samples for wake word training.
Creates 16kHz mono WAV files in the repo's data/recordings/samples/.

Lives in its own directory with its own uv environment: it runs on the host for
microphone access and needs only numpy, whereas the training environment pins torch
and tensorflow. Keeping them apart means recording does not need the trainer's stack.

Capture defaults to PyAudio. `--backend ffmpeg` is kept as a fallback for hosts
without PortAudio, but it is not the recommended path: recordings made through it
came out with audible clicks and pops that PyAudio's did not. Both backends run
through the same code path, so that difference is in the capture, not in how it is
driven.

Worth knowing that the level metering did NOT catch this. Peak and SNR describe
level and steady background noise; an impulsive click is brief enough to move
neither, so those takes measured well (SNR 38 dB) while sounding wrong. Listen to a
few clips before committing to a long session - the numbers are necessary, not
sufficient.

PyAudio needs PortAudio:  brew install portaudio && uv sync --extra pyaudio

Usage:
    cd record
    uv run record_samples.py --list-devices
    uv run record_samples.py --wake-word "hey seeree"
    uv run record_samples.py --wake-word "hey seeree" --output-dir ../data/recordings/samples/speaker1
"""

import argparse
import re
import subprocess
import sys
import termios
import time
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2                     # 16-bit
DURATION = 2.0  # seconds of usable audio captured after the cue
WARMUP = 0.6    # seconds discarded after opening the device, before the cue

# Samples belong to the repo, not to this tool's directory, so the default output
# path is anchored to the repo root rather than the working directory. Everything
# the pipeline records or generates lives under data/, which keeps the repo root
# clean and means one ignore rule covers the lot.
REPO_ROOT = Path(__file__).resolve().parent.parent
RECORDINGS_DIR = REPO_ROOT / "data" / "recordings"
SAMPLES_DIR = RECORDINGS_DIR / "samples"

# Unsplit recordings go OUTSIDE data/recordings/samples/. train.py globs that tree
# recursively for positives, so a three-minute raw file left there becomes a
# training positive - and after trimming, only its first 2 s survives, which is a
# few utterances and a lot of silence presented as one example of the wake word.
RAW_DIR = RECORDINGS_DIR / "raw"

FULL_SCALE = 32768.0
# Speech should peak somewhere near -12 dBFS.
#
# Be clear about what this does and does not buy. Absolute level turns out not to
# matter to the model at all: a synthetic sweep from -6 to -34 dBFS detected 6/6 at
# every step with a flat median score of ~0.984. What does matter is SNR, which
# degrades detection below about 15 dB. Recording hot is therefore about margin -
# it keeps SNR high and leaves headroom - not about a level the model needs.
#
# The reason to warn at capture rather than fix it later is that SNR is fixed at
# capture: amplifying a quiet clip raises its noise with it.
LOW_LEVEL_DBFS = -18.0
CLIPPING_DBFS = -0.5
LOW_SNR_DB = 20.0


def raw_dir_for(output_dir: Path) -> Path:
    """Where the unsplit recording for `output_dir` belongs.

    Mirrors the speaker subdirectory under data/recordings/raw/, so
    data/recordings/samples/speaker1 -> data/recordings/raw/speaker1. Outside the
    samples tree entirely, because train.py searches that tree recursively for
    positives.
    """
    try:
        return RAW_DIR / output_dir.resolve().relative_to(SAMPLES_DIR.resolve())
    except ValueError:
        return RAW_DIR


class Terminal:
    """Keeps the tty in the state we found it, take after take.

    A capture backend can leave the terminal non-canonical - ffmpeg puts it in
    cbreak mode for its own keyboard controls, and a killed ffmpeg never restores
    it. The damage outlives the program: ENTER then echoes as ^M and never submits
    a line, in this process and in the shell afterwards, until someone runs
    `stty sane`. Nothing in this file asks for that mode, which is exactly why it
    has to be undone here rather than avoided.

    So the settings are read once at startup and put back after every take, and
    again on the way out. Restoring costs nothing when nothing broke.

    All of it is a no-op off a TTY: piped stdin has no terminal settings, and its
    buffered input is a script's deliberate input rather than stray keystrokes.
    """

    def __init__(self):
        self.saved = None
        if sys.stdin.isatty():
            try:
                self.saved = termios.tcgetattr(sys.stdin)
            except (termios.error, ValueError, OSError):
                pass

    def restore(self):
        """Put the line discipline back, then drop anything typed during the take.

        Order matters: flushing first would leave the queue to refill from whatever
        arrives before the mode is fixed.
        """
        if self.saved is None:
            return
        try:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.saved)
            termios.tcflush(sys.stdin, termios.TCIFLUSH)
        except (termios.error, ValueError, OSError):
            pass                           # not a settable terminal; harmless


def next_index(output_dir: Path, safe_name: str) -> int:
    """One past the highest existing index for this wake word.

    Numbering from a file COUNT breaks as soon as the sequence has gaps: one
    speaker's directory held 56 clips numbered up to 0062 (takes had been deleted), so
    count + 1 pointed at 0057, which already existed - and recording would have
    silently overwritten five good samples before reaching free numbers.
    """
    pattern = re.compile(rf"^{re.escape(safe_name)}_(\d+)\.wav$")
    indices = [int(m.group(1)) for f in output_dir.glob("*.wav")
               if (m := pattern.match(f.name))]
    return max(indices, default=0) + 1


def measure(speech: np.ndarray, noise: np.ndarray):
    """Peak level, noise floor and SNR, all in dB.

    Speech level is the 90th-percentile 10 ms frame rather than the peak, so one
    transient does not stand in for the whole utterance. The noise floor comes from
    the warm-up audio, which is recorded before the cue and is therefore room tone
    by construction - it costs nothing to measure and is the only honest reference
    for how much of the recording is not signal.
    """
    peak = int(np.abs(speech).max()) if speech.size else 0
    peak_dbfs = 20 * np.log10(max(peak, 1) / FULL_SCALE)

    frame = SAMPLE_RATE // 100
    n = len(speech) // frame
    if n < 2:
        return peak_dbfs, None, None, 0
    frames = speech[:n * frame].astype(np.float64).reshape(n, frame)
    speech_rms = float(np.percentile(np.sqrt(np.mean(frames ** 2, axis=1)), 90))

    noise_rms = float(np.sqrt(np.mean(noise.astype(np.float64) ** 2))) if noise.size else 0.0
    snr = 20 * np.log10(speech_rms / noise_rms) if noise_rms > 0 and speech_rms > 0 else None
    noise_dbfs = 20 * np.log10(noise_rms / FULL_SCALE) if noise_rms > 0 else None

    clipped = int((np.abs(speech) >= 32700).sum())
    return peak_dbfs, noise_dbfs, snr, clipped


def list_devices(backend: str = "ffmpeg"):
    """Print audio input devices and their numbers.

    Device numbering is per backend - avfoundation and PortAudio enumerate
    separately - so --device from one is not valid for the other.
    """
    if backend == "pyaudio":
        import pyaudio
        pa = pyaudio.PyAudio()
        try:
            print("Audio input devices (use the number with --device):")
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if info.get("maxInputChannels", 0) > 0:
                    print(f"  {i}: {info['name']}")
        finally:
            pa.terminate()
        return

    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        capture_output=True, text=True,
    ).stderr
    audio = out.split("AVFoundation audio devices:")
    if len(audio) < 2:
        print(out)
        return
    print("Audio input devices (use the number with --device):")
    for line in audio[1].splitlines():
        m = re.search(r"\[(\d+)\]\s+(.*)", line)
        if m:
            print(f"  {m.group(1)}: {m.group(2).strip()}")


def open_stream(device: str) -> subprocess.Popen:
    """Spawn ffmpeg streaming raw 16kHz mono 16-bit PCM to stdout.

    `-nostdin` and a closed stdin are both needed, and not for tidiness: ffmpeg
    reads the terminal for its own keyboard controls, so an inherited stdin lets it
    eat the ENTER meant for the next take's prompt - the take-by-take loop then
    looks frozen. Closing the handle stops it reaching ffmpeg at all; -nostdin stops
    ffmpeg trying to read it.
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
        "-f", "avfoundation", "-i", f":{device}",
        "-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS),
        "-f", "s16le", "-",
    ]
    return subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def read_exact(pipe, nbytes: int):
    """Read exactly nbytes from a pipe, or None if the stream ends.

    A pipe read can come back short, so this loops - a partial read here would
    silently shorten the recording.
    """
    buf = b""
    while len(buf) < nbytes:
        chunk = pipe.read(nbytes - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


MIC_HINT = ("could not open the microphone. If this is the first run, macOS may be "
            "waiting on a microphone permission prompt for your terminal.")


def capture(backend: str, device: str, warmup_s: float, duration_s: float,
            on_chunk=None, cue: str = "", allow_partial: bool = False):
    """Record warm-up audio, then `duration_s` of speech. Returns both as int16.

    One entry point for both backends so the two are genuinely comparable: the
    warm-up handling, the cue timing and the framing are identical, and only the
    capture library differs. Anything that differs in the resulting audio is then
    attributable to the backend rather than to how it was driven.

    `on_chunk` is called with each block as it arrives, for the live level meter.
    Returning False from it stops the capture early.

    With `allow_partial`, Ctrl-C returns the audio captured so far instead of
    propagating. The chunks only exist inside this function, so an interrupt that
    escapes it discards the recording no matter what the caller does - which is why
    a long --continuous session has to be caught here rather than one frame up. An
    interrupt before any speech arrives still propagates: there is nothing to keep,
    and the warm-up is what the noise reference is measured from.
    """
    step = SAMPLE_RATE // 10 * SAMPLE_WIDTH                # 100 ms
    warmup_bytes = int(SAMPLE_RATE * warmup_s) * SAMPLE_WIDTH
    total_bytes = int(SAMPLE_RATE * duration_s) * SAMPLE_WIDTH
    warmup = b""
    chunks, got = [], 0

    if backend == "pyaudio":
        import pyaudio
        pa = pyaudio.PyAudio()
        try:
            stream = pa.open(format=pyaudio.paInt16, channels=CHANNELS,
                             rate=SAMPLE_RATE, input=True,
                             input_device_index=int(device) if device else None,
                             frames_per_buffer=step // SAMPLE_WIDTH)
        except Exception as e:
            pa.terminate()
            raise SystemExit(f"PyAudio {MIC_HINT}\n{e}")

        def read(n):
            return stream.read(n // SAMPLE_WIDTH, exception_on_overflow=False)

        try:
            warmup = read(warmup_bytes)
            if cue:
                print(cue)
            while got < total_bytes:
                chunk = read(min(step, total_bytes - got))
                chunks.append(chunk)
                got += len(chunk)
                if on_chunk and on_chunk(chunk, got) is False:
                    break
        except KeyboardInterrupt:
            if not (allow_partial and chunks):
                raise
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()
    else:
        proc = open_stream(device)
        if proc.stdout is None:
            raise SystemExit("failed to open ffmpeg pipes")
        try:
            warmup = read_exact(proc.stdout, warmup_bytes)
            if warmup is None:
                err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
                raise SystemExit(f"ffmpeg {MIC_HINT}\n{err}")
            if cue:
                print(cue)
            while got < total_bytes:
                chunk = read_exact(proc.stdout, min(step, total_bytes - got))
                if chunk is None:
                    break
                chunks.append(chunk)
                got += len(chunk)
                if on_chunk and on_chunk(chunk, got) is False:
                    break
        except KeyboardInterrupt:
            if not (allow_partial and chunks):
                raise
        finally:
            proc.kill()
            proc.wait()

    return (np.frombuffer(b"".join(chunks), dtype=np.int16),
            np.frombuffer(warmup[len(warmup) // 2:], dtype=np.int16))


def record_sample(filename: str, device: str, backend: str = "ffmpeg"):
    """Record one sample, cueing once the stream is live. Returns its levels.

    The device does not deliver usable audio the instant it opens, so WARMUP
    seconds are read first and the cue lands on an already-running stream - cueing
    before it settles loses the word onset (the "h" in "hey"). That audio is not
    written to the sample, but its second half is kept as a noise-floor reference.
    """
    audio, noise = capture(backend, device, WARMUP, DURATION, cue="SPEAK NOW!")
    if audio.size == 0:
        raise SystemExit("microphone stream ended before any audio arrived")

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())

    return measure(audio, noise)


def segment_utterances(data, noise_rms, min_ms=180.0, max_ms=2000.0,
                       gap_ms=250.0, pad_ms=100.0):
    """Split a continuous recording into one clip per utterance.

    Energy-gated against the measured room tone rather than a fixed level, so it
    adapts to the room instead of needing a threshold tuned by hand. Frames above
    the gate are grouped, runs closer together than `gap_ms` are merged (the pause
    inside "hey ... seeree" must not split the phrase in two), and each run is
    padded by `pad_ms` so the word onset and release survive.

    Runs outside [min_ms, max_ms] are dropped and reported: a very short one is a
    click or a breath, a very long one is two utterances run together or a stretch
    of noise. Neither makes a usable positive, and a bad clip is worse than a
    missing one - the corpus is only ~90 clips, so each is weighted heavily.

    Returns (segments, rejects) as lists of (start, end, padded_ms, voiced_ms).
    Both durations are carried because they answer different questions: the padded
    one is the clip you get, the voiced one is what the accept/reject test used -
    and reporting a rejection against the padded length mislabels a short fragment
    at the end of a recording as "too long".
    """
    frame = SAMPLE_RATE // 100                      # 10 ms
    n = len(data) // frame
    if n < 2:
        return [], []

    frames = data[:n * frame].astype(np.float64).reshape(n, frame)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))

    # 12 dB above room tone, but never below 5% of the loudest frame - that floor
    # keeps a very quiet room from turning breathing into an "utterance".
    gate = max(noise_rms * (10 ** (12 / 20)), rms.max() * 0.05)
    voiced = np.flatnonzero(rms > gate)
    if voiced.size == 0:
        return [], []

    merge = int(gap_ms / 10)
    runs, current = [], [voiced[0], voiced[0]]
    for i in voiced[1:]:
        if i - current[1] <= merge:
            current[1] = i
        else:
            runs.append(current)
            current = [i, i]
    runs.append(current)

    pad = int(SAMPLE_RATE * pad_ms / 1000)
    segments, rejects = [], []
    for a, b in runs:
        # Judge the run by its own length, not the padded clip's. Padding adds
        # 2 * pad_ms, which would carry a 30 ms click past a 180 ms floor and let
        # every click through as an "utterance".
        voiced_ms = (b + 1 - a) * 10
        start = max(0, a * frame - pad)
        end = min(len(data), (b + 1) * frame + pad)
        duration = (end - start) / SAMPLE_RATE * 1000
        (segments if min_ms <= voiced_ms <= max_ms else rejects).append(
            (start, end, duration, voiced_ms))
    return segments, rejects


def record_continuous(device: str, seconds: float, backend: str = "ffmpeg"):
    """Record one long block, returning (audio, room-tone reference).

    The whole block is held in memory - three minutes is under 6 MB - so
    segmentation can use statistics over the entire recording rather than having
    to decide about each utterance as it arrives.
    """
    print(f"RECORDING for {seconds:.0f}s - say the wake word, pause, repeat.")
    print("Ctrl-C stops early and keeps what has been recorded.\n")

    def meter(chunk, got):
        block = np.frombuffer(chunk, dtype=np.int16)
        level = 20 * np.log10(max(int(np.abs(block).max()), 1) / FULL_SCALE)
        bar = "#" * max(0, int((level + 60) / 3))
        print(f"\r  {got / SAMPLE_WIDTH / SAMPLE_RATE:5.1f}s  "
              f"{level:>6.1f} dBFS  {bar:<20}", end="", flush=True)

    audio, noise = capture(backend, device, WARMUP, seconds,
                           on_chunk=meter, allow_partial=True)
    print()
    if len(audio) < int(SAMPLE_RATE * seconds):
        print(f"  stopped early - keeping {len(audio) / SAMPLE_RATE:.1f}s")
    return audio, noise


def level_report(peak_dbfs, noise_dbfs, snr, clipped):
    """One line of feedback, plus any warning worth acting on before the next take."""
    parts = [f"peak {peak_dbfs:>6.1f} dBFS"]
    if noise_dbfs is not None:
        parts.append(f"noise {noise_dbfs:>6.1f} dBFS")
    if snr is not None:
        parts.append(f"SNR {snr:>4.0f} dB")
    print("    " + "   ".join(parts))

    if clipped:
        print(f"    CLIPPING: {clipped} sample(s) at full scale - lower the input gain.")
    elif peak_dbfs > CLIPPING_DBFS:
        print("    Very close to full scale - lower the input gain.")
    elif peak_dbfs < LOW_LEVEL_DBFS:
        print(f"    LOW LEVEL: aim for about -12 dBFS. Raise the input gain or move "
              f"closer;\n    this cannot be fixed later, since amplifying the clip "
              f"raises its noise too.")
    if snr is not None and snr < LOW_SNR_DB:
        print(f"    NOISY: only {snr:.0f} dB above the room. Quieter room, or closer mic.")


def write_segments(audio, noise, output_dir: Path, safe_name: str, args):
    """Cut a continuous recording into clips and write them. Returns their levels.

    With args.dry_run nothing is written - the split is only reported. That matters
    for --resegment: the clips from a first pass are already on disk, so a second
    pass with different parameters would write a duplicate of every one of them.
    """
    noise_rms = float(np.sqrt(np.mean(noise.astype(np.float64) ** 2))) if noise.size else 0.0
    segments, rejects = segment_utterances(
        audio, noise_rms, min_ms=args.min_ms, max_ms=args.max_ms, gap_ms=args.gap_ms)

    print(f"\nFound {len(segments)} utterance(s) in "
          f"{len(audio) / SAMPLE_RATE:.0f}s of audio")
    if rejects:
        short = sum(1 for *_, voiced in rejects if voiced < args.min_ms)
        print(f"  skipped {len(rejects)}: {short} too short (clicks, breaths, or an "
              f"utterance clipped by the end of the recording), "
              f"{len(rejects) - short} too long (merged or noise)")
        for start, _, _, voiced in rejects:
            why = "short" if voiced < args.min_ms else "long"
            print(f"    at {start / SAMPLE_RATE:>6.1f}s  {voiced:>5.0f}ms voiced  ({why})")

    index = next_index(output_dir, safe_name)
    session = []
    for start, end, duration, _ in segments:
        path = output_dir / f"{safe_name}_{index:04d}.wav"
        clip = audio[start:end]
        levels = measure(clip, noise)
        session.append(levels)

        if args.dry_run:
            print(f"  [dry run] {start / SAMPLE_RATE:>6.1f}s  {duration:>5.0f}ms  "
                  f"peak {levels[0]:>6.1f} dBFS")
            index += 1
            continue

        if path.exists():
            raise SystemExit(f"Refusing to overwrite {path}")
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(clip.tobytes())
        print(f"  {path.name}  {duration:>5.0f}ms  peak {levels[0]:>6.1f} dBFS")
        if levels[3] or levels[0] > CLIPPING_DBFS or levels[0] < LOW_LEVEL_DBFS:
            level_report(*levels)
        index += 1
    return session


def session_summary(session):
    """Report levels across a whole session.

    A per-sample warning is easy to shrug off; the same warning across a whole
    session is a setup problem, and much cheaper to fix now than after discovering
    it in a trained model.
    """
    if not session:
        return
    peaks = np.array([s[0] for s in session])
    snrs = np.array([s[2] for s in session if s[2] is not None])
    print(f"\nThis session: {len(session)} clip(s), "
          f"median peak {np.median(peaks):.1f} dBFS", end="")
    print(f", median SNR {np.median(snrs):.0f} dB" if snrs.size else "")

    low = int((peaks < LOW_LEVEL_DBFS).sum())
    if low > len(session) // 2:
        print(f"  {low}/{len(session)} samples were below {LOW_LEVEL_DBFS:.0f} dBFS. "
              "Raise the input gain before recording more -")
        print("  every sample in the set shares whatever level you record it at.")


def main():
    parser = argparse.ArgumentParser(description="Record voice samples for wake word training")
    parser.add_argument("--wake-word", default="hey seeree", help="Wake word you're recording")
    parser.add_argument("--raw-dir", default=None,
                        help="Where unsplit --continuous recordings go (default: "
                             "data/recordings/raw/, mirroring the speaker "
                             "subdirectory). Kept out of data/recordings/samples/ so "
                             "the trainer never picks a raw file up as a positive.")
    parser.add_argument("--output-dir", default=str(SAMPLES_DIR),
                        help="Output directory (default: %(default)s). Use a "
                             "per-speaker subdirectory when several people record.")
    parser.add_argument("--backend", choices=["pyaudio", "ffmpeg"], default="pyaudio",
                        help="Capture library (default: %(default)s). ffmpeg is kept "
                             "as a fallback for hosts without PortAudio, but its "
                             "recordings carried audible clicks - see the note at the "
                             "top of this file.")
    parser.add_argument("--device", default="0",
                        help="Input device number. Numbering is per backend - run "
                             "--list-devices with the same --backend.")
    parser.add_argument("--continuous", type=float, metavar="SECONDS",
                        help="Record one block of this length and split it into one "
                             "clip per utterance, instead of one ENTER per take. "
                             "Say the wake word, pause ~1s, repeat.")
    parser.add_argument("--resegment", metavar="WAV",
                        help="Re-split a raw recording saved by --continuous, without "
                             "recording again. Use after adjusting --min-ms/--gap-ms.")
    parser.add_argument("--keep-raw", action="store_true", default=True,
                        help="Keep the unsplit recording so it can be re-segmented "
                             "(default: on)")
    parser.add_argument("--dry-run", action="store_true",
                        help="With --resegment, report the split without writing "
                             "anything. Use it to tune --gap-ms/--min-ms before "
                             "committing, since re-segmenting otherwise duplicates "
                             "clips already written from the same recording.")
    parser.add_argument("--min-ms", type=float, default=180.0,
                        help="Shortest run kept as an utterance (default: %(default)s)")
    parser.add_argument("--max-ms", type=float, default=2000.0,
                        help="Longest run kept as an utterance (default: %(default)s)")
    parser.add_argument("--gap-ms", type=float, default=250.0,
                        help="Silence shorter than this does not split an utterance "
                             "(default: %(default)s)")
    parser.add_argument("--list-devices", action="store_true",
                        help="List microphones and exit")
    args = parser.parse_args()

    if args.list_devices:
        list_devices(args.backend)
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = args.wake_word.replace(" ", "_").lower()

    print("=" * 50)
    print(f"Voice Sample Recorder - \"{args.wake_word}\"")
    print("=" * 50)
    print()
    print("Record at least 20-50 samples for best results.")
    print("Vary your tone, speed, distance from mic, etc.")
    print()
    print("  - Press ENTER to start recording")
    print(f"  - Capture goes through {args.backend}; --list-devices shows microphones")
    print(f"  - Wait for \"SPEAK NOW!\", then say \"{args.wake_word}\" naturally")
    print("  - Recording lasts 2 seconds; silence is trimmed at training time")
    print("  - Levels are reported after each take; aim for a peak near -12 dBFS")
    print("  - Press 'q' + ENTER to quit")
    print()

    existing = len(list(output_dir.glob("*.wav")))
    index = next_index(output_dir, safe_name)
    print(f"Existing samples: {existing} (next file: {safe_name}_{index:04d}.wav)")
    print()

    if args.resegment:
        with wave.open(args.resegment, "rb") as wf:
            if wf.getframerate() != SAMPLE_RATE or wf.getnchannels() != CHANNELS:
                raise SystemExit(f"{args.resegment} must be {SAMPLE_RATE}Hz mono")
            audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        # No warm-up to draw on, so take the quietest second as the room reference.
        frame = SAMPLE_RATE
        blocks = [audio[i:i + frame] for i in range(0, len(audio) - frame, frame)]
        noise = min(blocks, key=lambda b: np.abs(b).mean()) if blocks else audio[:0]
        session = write_segments(audio, noise, output_dir, safe_name, args)
        session_summary(session)
        return

    if args.continuous:
        audio, noise = record_continuous(args.device, args.continuous, args.backend)
        if args.keep_raw:
            raw_dir = Path(args.raw_dir) if args.raw_dir else raw_dir_for(output_dir)
            raw_dir.mkdir(parents=True, exist_ok=True)
            raw = raw_dir / f"{safe_name}_raw_{int(time.time())}.wav"
            with wave.open(str(raw), "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(SAMPLE_WIDTH)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio.tobytes())
            print(f"  raw recording kept at {raw} "
                  f"(re-split with --resegment if needed)")
        session = write_segments(audio, noise, output_dir, safe_name, args)
        session_summary(session)
        return

    session = []
    term = Terminal()
    try:
        while True:
            term.restore()                 # undo any mangling by the last take
            user_input = input(f"[Sample {existing + 1}] Press ENTER to record "
                               f"(q or Ctrl-C to quit): ")

            if user_input.lower() == 'q':
                break

            print("Get ready...", end=" ", flush=True)
            time.sleep(0.5)

            path = output_dir / f"{safe_name}_{index:04d}.wav"
            if path.exists():                  # never clobber an existing take
                raise SystemExit(f"Refusing to overwrite {path}")
            levels = record_sample(str(path), args.device, args.backend)

            print(f"Saved: {path}")
            level_report(*levels)
            session.append(levels)
            existing += 1
            index += 1
            print()
    except (KeyboardInterrupt, EOFError):
        print()                            # end the half-written prompt line
    finally:
        term.restore()                     # never hand back a broken terminal

    print(f"\nDone! {existing} total samples in {output_dir}/")

    session_summary(session)


if __name__ == "__main__":
    main()

"""What the positive clips SAY, and how fast - shared by both trainers.

Both of these are tuned values with runs behind them, which is the whole reason they
live here rather than being written out twice. A second copy in the microWakeWord
path would drift from this one silently, and the drift would look like a model
difference.
"""

# Speed coverage of the positives, widened at the top for run 9.
#
# The measured failure: a synthetic sweep of the run 4 model detected 6/6 up to
# 1.25x and then fell off a cliff - 3/6 at 1.40x, 2/6 at 1.60x. Training rendered
# nothing above 1.3x, so the model fails just outside the range it was shown, and
# is fine below it (0.55x still gave 6/6). The asymmetry says widen the top only.
#
# It matches a real failure too: four of the five held-out clips run 4 missed were
# the fast ones, and the shortest (300 ms) was shorter than every clip it detected.
PLAIN_SPEEDS = (0.7, 1.6)

# Batched TTS needs every clip in a request to share one voice AND one speed, so
# plain speeds are drawn from a grid rather than continuously. 0.05 steps gives 19
# values across the range - fine enough that the corpus is barely distinguishable
# from a continuous draw, coarse enough that (voice, speed) buckets hold ~9 clips
# at the default sample count, which is a usable batch.
PLAIN_SPEED_STEP = 0.05
PLAIN_SPEED_GRID = [round(PLAIN_SPEEDS[0] + i * PLAIN_SPEED_STEP, 2)
                    for i in range(int((PLAIN_SPEEDS[1] - PLAIN_SPEEDS[0])
                                       / PLAIN_SPEED_STEP) + 1)]


def plain_positive_texts(wake_word: str) -> list:
    """The phrase-alone renderings, as text handed to the TTS.

    PUNCTUATION LEAVES A TAIL, AND THE TAIL MOVES THE ALIGNMENT. Run 14 shipped
    `...` and `!!` in this list and the alignment peak went 160 -> 200 ms with the
    firing floor 80 -> 160 ms, putting median latency at 160 ms against a 120 ms
    gate. create_fixed_size_clip aligns the END OF THE ARRAY with the end of the
    window (see trim_silence), so anything trailing the phrase - a drawn-out
    ending, a breath - displaces the phrase earlier in the window, and the model
    learns to wait longer before firing.

    Trailing material surviving trim_silence, vs the plain rendering, median over
    8 voices: `...` +95 ms, `!!` +55 ms, `?` +20 ms, `!` +15 ms, `.` +15 ms,
    `,` +10 ms. The two heavy ones are gone. It is strongly voice-dependent -
    am_liam and bf_emma add 120-170 ms to EVERY punctuated variant while af_sarah
    adds nothing - so this is a property of the corpus as a whole, not of one mark.

    `wake_word` appears twice on purpose. The pre-run-14 list held three
    plain-equivalent entries (`wake_word`, `.lower()` which was the same string,
    and `.title()` which renders identically), and that is why its alignment was
    tight. Weighting plain back up keeps the mean tail near +10 ms, against +30 ms
    for run 14's list. Prosody diversity is worth less than alignment: the spread
    here is 0.09-0.44 in embedding distance, while a different voice is 0.70.

    The alignment argument is openWakeWord's, but the tail is a property of the
    RENDERING, not of the frontend, so it applies to microWakeWord too - only the
    window it displaces the phrase within differs (1500 ms rather than 2000 ms).
    """
    return [
        wake_word,
        wake_word,
        f"{wake_word}!",
        f"{wake_word}?",
        f"{wake_word},",
        f"{wake_word}.",
    ]

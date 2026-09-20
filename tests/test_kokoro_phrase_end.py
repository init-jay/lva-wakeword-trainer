"""Guards for phrase_end_sample (re-exported by train/corpus/kokoro.py from
tts_protocol.audio).

The contract it broke before: the wake-word end of a Kokoro clip is cut at
this function's answer, and it was verified-by-word-match rather than
positional because a tokenisation mismatch (a normalisation rule splitting a
word) used to cut at the wrong place silently - "a wrong cut here is what
broke the alignment last time" (audio.py docstring). The tests pin the
match, the mismatch-to-None, and the run-on case where words follow the
wake word and must not move the cut.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.corpus.kokoro import phrase_end_sample  # noqa: E402

TS = 16000  # SR; phrase_end_sample's default


def _t(word, start, end):
    return {"word": word, "start_time": start, "end_time": end}


def test_matching_timestamps_return_end_of_last_phrase_word():
    # Punctuation and case are stripped before the match, so a timestamped
    # "Seeree." still lines up with the wake word's word.
    ts = [_t("Hey", 0.1, 0.4), _t("Seeree.", 0.4, 0.9)]
    assert phrase_end_sample(ts, "hey seeree", TS) == int(0.9 * TS) == 14400


def test_words_not_matching_the_phrase_return_none():
    # The guard: a timestamp list that does not contain the phrase words
    # must not cut at all rather than at the wrong sample.
    ts = [_t("hello", 0.1, 0.4), _t("there", 0.4, 0.9)]
    assert phrase_end_sample(ts, "hey seeree", TS) is None


def test_no_timestamps_returns_none():
    assert phrase_end_sample([], "hey seeree", TS) is None


def test_run_on_words_after_the_phrase_do_not_move_the_cut():
    # The run-on samples this cut exists for: words after the wake word are
    # part of the same utterance and must not be counted as the phrase's
    # tail. The cut stays at the end of the last phrase word.
    ts = [_t("hey", 0.0, 0.2), _t("seeree", 0.2, 0.5), _t("play", 0.5, 0.9)]
    assert phrase_end_sample(ts, "hey seeree", TS) == int(0.5 * TS) == 8000


def main():
    import _runner
    _runner.run(sys.modules[__name__])


if __name__ == "__main__":
    main()

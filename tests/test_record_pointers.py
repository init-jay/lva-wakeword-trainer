"""Pointers to the measurements must resolve from the tree that carries them.

SPEED.md was split, not deleted. `docs/SPEED.md` is the pipeline-efficiency half -
route choice, stage costs, the avenues closed by measurement - and it belongs in
this tree, because it is what anyone tuning the pipeline for a different wake word
needs. The campaign's model results (per-speaker scores, sweep verdicts, the
staged candidate) moved to branch `train/hey_seeree`, where the rest of that
record lives.

So a citation of `SPEED.md` now has exactly two correct forms: `docs/SPEED.md`,
which resolves here, or `record/SPEED.md`, where `record/` means that branch (the
shorthand is defined in CLAUDE.md). A bare `SPEED.md` is a dead pointer: the root
file is gone, and the reader has no way to know which half they were being sent to.

That is invisible in review - a bare "SPEED.md" reads like a citation, and every
other form of it also reads like a citation - hence this test rather than a note.
Comments and docstrings wrap at ~79 columns, so the qualifier and the branch name
often land on different lines; the check judges the citation plus the two lines
after it, which is what a reader's eye does. On a tree that still carries the root
file (the record branch), bare citations are correct, so nothing is asserted -
and that fact is printed rather than passed quietly.
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

RECORD_BRANCH = "train/hey_seeree"
RECORD_FILE = "SPEED.md"
GUIDANCE_FILE = "docs/SPEED.md"

# Not source: generated corpus/model trees and the record's own prose, where the
# bare name is the subject rather than a pointer.
SKIP_DIRS = {".git", "__pycache__", ".venv", "data", "output", "logs", "node_modules"}
SKIP_NAMES = {"uv.lock", "RECORD.md"}

# A provenance tag is <commit>-d<digest> / <commit>-<digest>-<hash> (src/provenance.py).
MODEL_TAG = re.compile(r"\b[0-9a-f]{7}-(?:d[0-9a-f]{6,8}|[0-9a-f]{8})\b")


def _lines(path):
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except (UnicodeDecodeError, OSError):
        return []                     # binary or unreadable: no prose to check


def _citations(root):
    """(relpath, lineno, window) for every line naming the measurements file."""
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in SKIP_NAMES or path == Path(__file__):
            continue
        rel = path.relative_to(root)
        if set(rel.parts[:-1]) & SKIP_DIRS:
            continue
        lines = _lines(path)
        for i, line in enumerate(lines):
            if RECORD_FILE in line:
                yield str(rel), i + 1, "\n".join(lines[i:i + 3])


def test_every_citation_of_the_measurements_resolves():
    bad = []
    for rel, line_no, window in _citations(REPO_ROOT):
        if GUIDANCE_FILE in window or f"record/{RECORD_FILE}" in window \
                or RECORD_BRANCH in window:
            continue
        bad.append(f"{rel}:{line_no}: {window.splitlines()[0].strip()}")
    assert not bad, (
        f"{len(bad)} citation(s) of {RECORD_FILE} carry no prefix, and the root file "
        f"is gone - write `{GUIDANCE_FILE}` for the efficiency half or "
        f"`record/{RECORD_FILE}` for the campaign results (branch {RECORD_BRANCH}):\n"
        + "\n".join(f"    {b}" for b in bad[:20]))


def test_both_halves_are_where_the_pointers_say():
    """A pointer that resolves is only useful if it resolves to the right half."""
    assert (REPO_ROOT / GUIDANCE_FILE).is_file(), f"{GUIDANCE_FILE} is missing"
    if (REPO_ROOT / RECORD_FILE).exists():
        print(f"  {RECORD_FILE} is in this tree (a record branch); the split "
              f"checks do not apply here")
        return
    assert not (REPO_ROOT / RECORD_FILE).exists(), (
        f"root {RECORD_FILE} is back: it is the campaign record, and it belongs on "
        f"{RECORD_BRANCH}")


def test_the_guidance_doc_is_efficiency_and_not_a_scorecard():
    """`docs/SPEED.md` must stay the transferable half.

    The seam this guards is the one the split was drawn at: a timings doc that
    accumulates "model X scored Y" lines has silently become the record again, and
    the next wake word's owner inherits numbers from a campaign that is not theirs.
    Route costs and stage costs are allowed to name the corpus they were measured
    on - a timing without its conditions is useless - so this checks the two forms
    that are only ever results: a model's provenance tag, and a scorecard citation.
    """
    doc = (REPO_ROOT / GUIDANCE_FILE).read_text()
    tags = MODEL_TAG.findall(doc)
    assert not tags, f"model provenance tags in {GUIDANCE_FILE}: {tags}"
    assert "scorecards.jsonl" not in doc and "det@FA" not in doc, (
        f"{GUIDANCE_FILE} cites scorecard rows; those are one word's results and "
        f"belong on {RECORD_BRANCH}")


def test_the_record_shorthand_is_defined_where_a_reader_would_look():
    """`record/` is only useful if its meaning is one hop away, in the file that
    tells an agent where things live."""
    claude = (REPO_ROOT / "CLAUDE.md").read_text()
    assert f"record/{RECORD_FILE}" in claude, (
        "CLAUDE.md no longer defines the `record/` shorthand the comments use")
    assert RECORD_BRANCH in claude and GUIDANCE_FILE in claude, (
        f"CLAUDE.md must name both halves: {GUIDANCE_FILE} and branch {RECORD_BRANCH}")


def main():
    import _runner
    _runner.run(sys.modules[__name__])


if __name__ == "__main__":
    main()

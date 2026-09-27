"""The record boundary: mechanism must not depend on one word's campaign.

SPEED.md was split, not deleted. `docs/SPEED.md` is the pipeline-efficiency half -
route choice, stage costs, the avenues closed by measurement - and it lives in this
tree because it is what anyone tuning the pipeline for a different wake word needs.
The campaign's model results (per-speaker scores, sweep verdicts, the staged
candidate) live on branch `train/hey_seeree`.

The question that decides which half a citation belongs to is not "is this number
big or small" but **whose fact is it**. And that question has a sharp edge this file
guards: a comment in a trainer, an image, the Makefile or a skill that cites the
record branch is a mechanism note that resolves on exactly one fork and nowhere else.
Someone cloning this repo to train their own word cannot follow it, cannot check it,
and cannot tell whether it is still true. So:

  * code, build files and skills cite `docs/SPEED.md` or nothing;
  * only the prose that deliberately signposts the split - CLAUDE.md, README.md,
    ARCHITECTURE.md and docs/ - may name the record branch;
  * `docs/SPEED.md` stays the efficiency half: a model provenance tag or a
    scorecard citation in it means the record crept back into the generic tree.

Static on purpose. There is no CI here, the suite runs on the host in seconds, and
every failure mode this covers is invisible in review: an unprefixed `SPEED.md`
reads like a citation, and so does a citation of a branch the reader's clone does
not have. Comments and docstrings wrap at ~79 columns, so the qualifier and the
branch name often land on different lines; the checks judge three-line windows,
which is what a reader's eye does.
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

RECORD_BRANCH = "train/hey_seeree"
RECORD_FILE = "SPEED.md"
GUIDANCE_FILE = "docs/SPEED.md"

# The prose whose JOB is to say where the record went. Nothing else may name it.
SIGNPOST_DOCS = {"CLAUDE.md", "README.md", "ARCHITECTURE.md"}
SIGNPOST_DIRS = {"docs"}

# Generated trees and the record's own branch prose: not source to police.
SKIP_DIRS = {".git", "__pycache__", ".venv", "data", "output", "logs", "node_modules"}
SKIP_NAMES = {"uv.lock", "RECORD.md"}

# A provenance tag is <commit>-d<digest> / <commit>-<digest>-<hash> (src/provenance.py).
MODEL_TAG = re.compile(r"\b[0-9a-f]{7}-(?:d[0-9a-f]{6,8}|[0-9a-f]{8})\b")


def _tracked_text_files():
    for path in sorted(REPO_ROOT.rglob("*")):
        if not path.is_file() or path.name in SKIP_NAMES or path == Path(__file__):
            continue
        rel = path.relative_to(REPO_ROOT)
        if set(rel.parts[:-1]) & SKIP_DIRS:
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            continue                     # binary: no prose to check
        yield rel, lines


def _is_signpost(rel):
    return rel.name in SIGNPOST_DOCS or (len(rel.parts) > 1 and rel.parts[0] in SIGNPOST_DIRS)


def _windows(lines, needle):
    """(lineno, three-line window) for each line containing `needle`.

    The window exists because a citation can wrap: 'the timings are in docs/SPEED.md'
    and 'branch train/hey_seeree' land on different lines more often than not, and a
    per-line rule would flag correct text.
    """
    for i, line in enumerate(lines):
        if needle in line:
            yield i + 1, "\n".join(lines[i:i + 3])


def test_only_the_signposts_name_the_record_branch():
    bad = []
    for rel, lines in _tracked_text_files():
        if _is_signpost(rel):
            continue
        for line_no, window in _windows(lines, RECORD_BRANCH):
            bad.append(f"{rel}:{line_no}: {window.splitlines()[0].strip()}")
    assert not bad, (
        f"{len(bad)} mention(s) of branch {RECORD_BRANCH} outside the signposts "
        f"({sorted(SIGNPOST_DOCS)}, {'/'.join(sorted(SIGNPOST_DIRS))}/). Mechanism "
        f"must not depend on the record existing - a reader cloning this repo for "
        f"their own word has no such branch:\n"
        + "\n".join(f"    {b}" for b in bad[:20]))


def test_every_citation_of_the_timings_resolves_here():
    """Any mention of the measurements file must be the in-tree half."""
    bad = []
    for rel, lines in _tracked_text_files():
        for line_no, window in _windows(lines, RECORD_FILE):
            if GUIDANCE_FILE in window:
                continue
            if _is_signpost(rel) and (RECORD_BRANCH in window or RECORD_FILE in window):
                continue                 # the signposts' job is naming both halves
            bad.append(f"{rel}:{line_no}: {window.splitlines()[0].strip()}")
    assert not bad, (
        f"{len(bad)} citation(s) of {RECORD_FILE} do not resolve in this tree - "
        f"route and stage costs belong in `{GUIDANCE_FILE}` (which is here), and the "
        f"campaign results belong on {RECORD_BRANCH} (which mechanism must not cite):\n"
        + "\n".join(f"    {b}" for b in bad[:20]))


def test_both_halves_are_where_the_pointers_say():
    assert (REPO_ROOT / GUIDANCE_FILE).is_file(), f"{GUIDANCE_FILE} is missing"
    assert not (REPO_ROOT / RECORD_FILE).exists(), (
        f"root {RECORD_FILE} is back: those are one word's model results, and the "
        f"efficiency half already lives in {GUIDANCE_FILE}")


def test_the_guidance_doc_is_efficiency_and_not_a_scorecard():
    """The seam the split was drawn at, checked on content.

    Corpus conditions may be named - a timing without its conditions is useless -
    but the two forms that are only ever results are not: a model's provenance tag,
    and a scorecard citation.
    """
    doc = (REPO_ROOT / GUIDANCE_FILE).read_text()
    tags = MODEL_TAG.findall(doc)
    assert not tags, f"model provenance tags in {GUIDANCE_FILE}: {tags}"
    assert "scorecards.jsonl" not in doc and "det@FA" not in doc, (
        f"{GUIDANCE_FILE} cites scorecard rows; those are one word's results and "
        f"belong on {RECORD_BRANCH}")


def main():
    import _runner
    _runner.run(sys.modules[__name__])


if __name__ == "__main__":
    main()

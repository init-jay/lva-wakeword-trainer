"""Mechanism must not depend on one word's campaign.

`docs/SPEED.md` holds the pipeline's timings. Model results belong in no file here:
they are a property of whichever word was trained, not of the pipeline. These checks
exist because the failure is invisible in review - a citation of a branch nobody else
has reads exactly like a citation.
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RECORD_BRANCH = "train/hey_seeree"      # the branch this tree must never reference
RECORD_FILE = "SPEED.md"
GUIDANCE = REPO_ROOT / "docs" / RECORD_FILE
SKIP_DIRS = {".git", "__pycache__", ".venv", "data", "output", "logs", "node_modules"}
SKIP_NAMES = {"uv.lock", "RECORD.md"}
# A model's provenance tag: <commit>-d<digest> or <commit>-<digest>-<hash>.
MODEL_TAG = re.compile(r"\b[0-9a-f]{7}-(?:d[0-9a-f]{6,8}|[0-9a-f]{8})\b")


def _files():
    for path in sorted(REPO_ROOT.rglob("*")):
        if not path.is_file() or path.name in SKIP_NAMES or path == Path(__file__):
            continue
        rel = path.relative_to(REPO_ROOT)
        if set(rel.parts[:-1]) & SKIP_DIRS:
            continue
        try:
            yield rel, path.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            pass                          # binary: nothing to read


def test_no_file_references_the_record_branch():
    hits = [f"{rel}:{i}: {ln.strip()}" for rel, lines in _files()
            for i, ln in enumerate(lines, 1) if RECORD_BRANCH in ln]
    assert not hits, (
        f"{len(hits)} reference(s) to branch {RECORD_BRANCH}. Someone cloning this "
        f"repo to train their own word has no such branch, so mechanism text cannot "
        f"depend on it:\n    " + "\n    ".join(hits[:20]))


def test_every_timings_citation_names_the_in_tree_doc():
    # Three-line window: a citation wraps, and a per-line rule flags correct prose.
    hits = []
    for rel, lines in _files():
        for i, ln in enumerate(lines):
            if RECORD_FILE in ln and GUIDANCE.name not in "\n".join(lines[i:i + 3]):
                hits.append(f"{rel}:{i + 1}: {ln.strip()}")
    assert not hits, (
        f"{len(hits)} citation(s) of {RECORD_FILE} that do not resolve here - the "
        f"timings live in docs/{RECORD_FILE}:\n    " + "\n    ".join(hits[:20]))


def test_the_timings_doc_is_timings_and_not_a_scorecard():
    doc = GUIDANCE.read_text()
    tags = MODEL_TAG.findall(doc)
    assert not tags, f"model provenance tags in docs/{RECORD_FILE}: {tags}"
    assert "scorecards.jsonl" not in doc and "det@FA" not in doc, (
        f"docs/{RECORD_FILE} cites model results; those are one word's, not the "
        f"pipeline's")


def test_the_root_record_file_stays_gone():
    assert not (REPO_ROOT / RECORD_FILE).exists(), (
        f"root {RECORD_FILE} is back, and it is the record: docs/{RECORD_FILE} is "
        f"the half this repo keeps")


def main():
    import _runner
    _runner.run(sys.modules[__name__])


if __name__ == "__main__":
    main()

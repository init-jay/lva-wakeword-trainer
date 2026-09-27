"""Pointers to the record must resolve from a tree that has no record.

`main` is the pipeline; the hey_seeree measurements live on branch
`train/hey_seeree` (see CLAUDE.md, "Where the measurements live"). Deleting
SPEED.md from this tree is the easy half - the hard half is the ~20 comments
that cite it as a *locatable source*: 'record/SPEED.md "Piper fleet"',
"the result is in SPEED.md", "record that in SPEED.md and stop". Those are
instructions to go look, and after the deletion they point at a file that is
not here. CLAUDE.md:141 says plainly that the improvement.md/bug.md citations are
provenance labels rather than links to keep alive, and that stays true of them:
naming where a finding came from does not require a resolvable path. A citation
that tells the reader to go and read something does.

So the rule this enforces is narrow and mechanical: in a tree WITHOUT SPEED.md,
every mention of it carries the `record/` prefix (defined in CLAUDE.md) or names
the branch, so the reader knows to look on `train/hey_seeree` instead of hunting
for a file that was never meant to be here.

Why a static test and not a comment: the pointer set is written by every future
PR that measures something on its own wake word and moves the record out. A bare
"SPEED.md" typed into a new comment is invisible in review - the reviewer sees a
citation, not a dangling one - and nothing else in the suite would notice.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Where the record lives when it is not in this tree.
RECORD_BRANCH = "train/hey_seeree"
RECORD_FILE = "SPEED.md"

# Not source: generated corpus/model trees, the data dirs, and the record's own
# prose (RECORD.md is the record branch's README, where the name is the subject).
SKIP_DIRS = {".git", "__pycache__", ".venv", "data", "output", "logs", "node_modules"}
SKIP_NAMES = {"uv.lock", "RECORD.md"}


def _mentions(path):
    """(line_no, line) for every line naming the record file, or [] if unreadable."""
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []                     # binary or unreadable: no prose to check
    return [(i, ln) for i, ln in enumerate(text.splitlines(), 1) if RECORD_FILE in ln]


def test_every_record_pointer_says_where_the_record_is():
    if (REPO_ROOT / RECORD_FILE).exists():
        # A tree that carries the record (the record branch itself, or main before
        # it moved) needs no qualifier: the bare name resolves. Assert nothing,
        # loudly enough that a run here is not mistaken for a pass.
        print(f"  {RECORD_FILE} is in this tree, so bare citations resolve; "
              f"qualification not required")
        return
    bad = []
    for path in sorted(REPO_ROOT.rglob("*")):
        if not path.is_file() or path.parent.name in SKIP_DIRS or any(
                p in SKIP_DIRS for p in path.relative_to(REPO_ROOT).parts[:-1]):
            continue
        if path.name in SKIP_NAMES or path == Path(__file__):
            continue
        for line_no, line in _mentions(path):
            # Comments and docstrings wrap at ~79 columns, so "SPEED.md on branch"
            # and "train/hey_seeree" land on different lines more often than on the
            # same one. The rule is semantic - a reader must be able to find the
            # record - so judge it over the citation and the two lines after it.
            window = "\n".join(line.splitlines()[0:1] + _tail(path, line_no, 2))
            qualified = f"record/{RECORD_FILE}" in window or RECORD_BRANCH in window
            if not qualified:
                bad.append(f"{path.relative_to(REPO_ROOT)}:{line_no}: {line.strip()}")
    assert not bad, (
        f"{len(bad)} citation(s) of {RECORD_FILE} do not say where the record is - "
        f"the file is not in this tree, so a bare name is a dead pointer. Prefix it "
        f"`record/` (CLAUDE.md defines the shorthand) or name branch "
        f"{RECORD_BRANCH}:\n" + "\n".join(f"    {b}" for b in bad[:20]))


def test_the_qualified_pointers_name_a_branch_that_exists():
    """`record/` is only useful if it means something. Check the definition is in
    CLAUDE.md, so a reader who greps the shorthand finds its meaning in one hop."""
    claude = (REPO_ROOT / "CLAUDE.md").read_text()
    assert f"record/{RECORD_FILE}" in claude or "record/`" in claude, (
        "CLAUDE.md no longer defines the `record/` shorthand that the comments use")
    assert RECORD_BRANCH in claude, f"CLAUDE.md does not name branch {RECORD_BRANCH}"


def _tail(path, line_no, n):
    """The n lines after line_no, for judging a citation that wraps."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (UnicodeDecodeError, OSError):
        return []
    return lines[line_no:line_no + n]


def main():
    import _runner
    _runner.run(sys.modules[__name__])


if __name__ == "__main__":
    main()

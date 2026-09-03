"""Give container-written output back to the host user.

THE TRAINERS RUN AS ROOT AND THE HOST DOES NOT. Every file either trainer writes
under output/ therefore lands root-owned on the host, and the first thing that
touches it afterwards fails:

    cp: cannot create regular file
        'output/hey_seeree/oww/hey_seeree_<tag>.onnx': Permission denied

That is the tagging copy in run-oww-training.sh, on the host, immediately after a
successful run - the most annoying possible moment. The same wall is hit collecting
the microWakeWord artifacts, scp'ing a model off the box, or deleting an old one,
and it is the same root-ownership problem that made `mv` refuse on the ambient sets.

THE OWNER IS TAKEN FROM A REFERENCE PATH, NOT FROM output/ ITSELF. The obvious
implementation reads the uid off output/ and applies it downward, on the reasoning
that a bind mount carries its host owner. That is true only when the human created
the directory: DOCKER CREATES A MISSING BIND-MOUNT SOURCE AS ROOT. So on a fresh
checkout output/ is root-owned too, the function reads uid 0, compares it against
its own euid 0, concludes "already mine" and does nothing - which is exactly the
case it exists to fix, and it silently did nothing at all.

So the reference is a directory that can only have come from the human: the
git-tracked source trees compose mounts read-only. Those exist in the clone before
any container runs, so their owner is whoever cloned the repo.

Shared by both trainers rather than duplicated, because the failure is identical on
both sides and was found the expensive way twice.
"""

import os
from pathlib import Path

# Mounted, git-tracked, and therefore created by the clone rather than by Docker.
# Ordered by how certain that is.
DEFAULT_REFERENCES = ("train", "wordlists", "eval", "scripts")


def target_owner(work_dir, references=DEFAULT_REFERENCES):
    """(uid, gid) of the human who owns the checkout, or None if it cannot be told.

    Skips anything root-owned: root is either Docker's doing or a genuinely
    root-run host, and in both cases it tells us nothing about who to hand back to.
    """
    for name in references:
        try:
            info = (Path(work_dir) / name).stat()
        except OSError:
            continue
        if info.st_uid != 0:
            return info.st_uid, info.st_gid
    return None


def hand_back(root, work_dir=None, references=DEFAULT_REFERENCES):
    """chown `root` and everything under it to whoever owns the checkout.

    Best-effort by design: a permissions problem here must not fail a run that has
    already produced a model. Returns the number of paths changed.
    """
    root = Path(root)
    work_dir = Path(work_dir) if work_dir else root.parent

    owner = target_owner(work_dir, references)
    if owner is None:
        return 0                      # nothing root-owned to fix, or nobody to fix it to
    uid, gid = owner
    if uid == os.geteuid():
        return 0                      # running as the human already - the host case

    changed = 0
    # root ITSELF as well as its contents: a root-owned output/ stops the host
    # creating the tagged copy inside it, which is the failure that started this.
    for path in [root, *root.rglob("*")]:
        try:
            if path.stat().st_uid != uid:
                os.chown(path, uid, gid)
                changed += 1
        except OSError:
            continue                  # keep going; report what did work

    if changed:
        print(f"  handed {changed} path(s) under {root} back to uid {uid}:{gid}")
    return changed

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

THE DESIRED OWNER IS NOT GUESSED OR PASSED IN. output/ is a bind mount, so the
directory itself already carries the host user's uid/gid. Read it from there and
apply it downward - which means no UID has to be threaded through compose, and it
stays correct if the repo moves to a different user.

Shared by both trainers rather than duplicated, because the failure is identical on
both sides and was found the expensive way once already.
"""

import os
from pathlib import Path


def hand_back(root):
    """chown everything under `root` to whoever owns `root` itself.

    Best-effort by design: a permissions problem here must not fail a run that has
    already produced a model. Returns the number of paths changed.
    """
    root = Path(root)
    try:
        info = root.stat()
    except OSError as exc:                                           # noqa: BLE001
        print(f"  NOTE: cannot stat {root}: {exc}")
        return 0

    uid, gid = info.st_uid, info.st_gid
    if uid == os.geteuid():
        return 0                      # already ours - the host-side case

    changed = 0
    for path in root.rglob("*"):
        try:
            if path.stat().st_uid != uid:
                os.chown(path, uid, gid)
                changed += 1
        except OSError:
            continue                  # keep going; report what did work

    if changed:
        print(f"  handed {changed} file(s) under {root} back to uid {uid}")
    return changed

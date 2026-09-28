#!/usr/bin/env python3
"""The single state surface: where the pipeline stands, from one command.

An agent arriving cold used to need ~eight separate probes (which words have
recipes, which venvs are built, is external data present, recordings per
speaker, last run tag, run history, what is staged) - and an agent that skips
one proceeds on a wrong assumption, the expensive failure mode in a repo where
the next step can cost 35 minutes. This reads all of them in one pass.

READ-ONLY. It reads the append-only run ledger (output/<word>/runs.jsonl),
.last_run_tag and deploy/scorecards.jsonl; it never writes any of them.
The ledger is append-only and this is not an appender.

Per word and per target; per speaker for recordings. Nothing is pooled across
words or speakers - the invariant that an average hides the failing voice
applies to inputs as much as to scores.

    python3 src/scripts/status.py          # human-readable
    python3 src/scripts/status.py --json   # agent-readable

Stdlib only on purpose: this must run on a cold clone before any venv exists.
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# The venvs the pipeline can need. tests/.venv is the suite's own env
# (improvements.md item 2); the rest are the per-route environments.
VENVS = [
    "tests/.venv/bin/python",
    "src/train/train-applesilicon/.venv/bin/python",
    "src/train/train-mww-applesilicon/.venv/bin/python",
    "src/eval/.venv/bin/python",
    "src/record/.venv/bin/python",
    "src/preflight/.venv/bin/python",
]

EXTERNAL_FIX = "./src/scripts/download-external-data.sh [all|oww|mww]"
TARGETS = ("oww", "mww")


def _count_wav(tree: Path) -> int:
    if not tree.is_dir():
        return 0
    return sum(1 for p in tree.rglob("*.wav") if p.is_file())


def _dir_tree_size(tree: Path):
    """(file count, bytes) without a subprocess; stat-walk, seconds at 45 GB."""
    if not tree.is_dir():
        return 0, 0
    n, b = 0, 0
    for dirpath, _dirnames, filenames in os.walk(tree):
        for name in filenames:
            p = Path(dirpath) / name
            try:
                n += 1
                b += p.stat().st_size
            except OSError:
                pass
    return n, b


def _speakers(samples: Path):
    """Per-speaker clip counts under a samples/ or holdout/ tree.

    A speaker is a direct subdirectory; the documented single-speaker form
    (clips directly in the tree) is reported under "(root)". _runon dirs are
    kept separate, never merged into the plain count.
    """
    out = {}
    if not samples.is_dir():
        return out
    for p in sorted(samples.iterdir()):
        if p.is_dir():
            out[p.name] = _count_wav(p)
    loose = sum(1 for p in samples.glob("*.wav") if p.is_file())
    if loose:
        out["(root)"] = loose
    return out


def _last_ledger_row(word: Path):
    ledger = word / "runs.jsonl"
    if not ledger.is_file():
        return None, 0
    rows = 0
    last = None
    with ledger.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows += 1
            last = json.loads(line)
    if last is None:
        return None, rows
    slim = {k: last.get(k) for k in
            ("tag", "target", "seed", "recorded_utc", "corpus_id")}
    wall = last.get("wall_time")
    if isinstance(wall, dict):
        slim["wall_time_s"] = round(sum(v for v in wall.values()
                                        if isinstance(v, (int, float))), 1)
    gates = (last.get("eval_block") or {}).get("gates")
    if gates:
        slim["gates_failed"] = [g.get("check") for g in gates
                                if isinstance(g, dict) and not g.get("pass")]
    return slim, rows


def collect() -> dict:
    words = set()
    recipes = sorted(p.stem for p in (REPO_ROOT / "recipes").glob("*.yaml"))
    words |= set(recipes)
    for base in ("data/recordings", "output"):
        root = REPO_ROOT / base
        if root.is_dir():
            words |= {p.name for p in root.iterdir() if p.is_dir()}

    state = {
        "recipes": recipes,
        "venvs": {v: (REPO_ROOT / v).exists() for v in VENVS},
        "external": None,
        "recordings": {},
        "runs": {},
        "deploy": {},
    }

    ext = REPO_ROOT / "data" / "external"
    if ext.is_dir():
        entries = {}
        for p in sorted(ext.iterdir()):
            if p.name.startswith("."):
                continue
            n, b = _dir_tree_size(p) if p.is_dir() else (1, p.stat().st_size)
            entries[p.name] = {"files": n, "bytes": b}
        n, b = _dir_tree_size(ext)
        state["external"] = {
            "present": True,
            "files": n,
            "bytes": b,
            "entries": entries,
        }
    else:
        state["external"] = {
            "present": False,
            "fix": EXTERNAL_FIX,
        }

    for word in sorted(words):
        slug = word.replace(" ", "_").lower()

        rec = REPO_ROOT / "data" / "recordings" / slug
        state["recordings"][word] = {
            "samples": _speakers(rec / "samples"),
            "holdout": _speakers(rec / "holdout"),
        }

        out = REPO_ROOT / "output" / slug
        last, rows = _last_ledger_row(out)
        per_target = {}
        for t in TARGETS:
            tag_file = out / t / ".last_run_tag"
            per_target[t] = tag_file.read_text().strip() if tag_file.is_file() else None
        state["runs"][word] = {
            "last_run_tag": per_target,
            "ledger_rows": rows,
            "last_ledger_row": last,
        }

    scorecards = REPO_ROOT / "deploy" / "scorecards.jsonl"
    if scorecards.is_file():
        by_key = {}
        with scorecards.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                by_key.setdefault(f"{r.get('target')}:{r.get('satellite')}", []).append(
                    {k: r.get(k) for k in ("id", "tag", "threshold", "status",
                                           "measured_utc")})
        state["deploy"] = {"scorecard_rows": sum(len(v) for v in by_key.values()),
                           "by_target": by_key}
    else:
        state["deploy"] = {"scorecard_rows": 0, "by_target": {}}
    staged = []
    deploy = REPO_ROOT / "deploy"
    if deploy.is_dir():
        for p in sorted(deploy.rglob("*")):
            if p.is_file() and p.parent != deploy and p.parent.name != "__pycache__":
                staged.append(str(p.relative_to(REPO_ROOT)))
    state["deploy"]["staged_files"] = staged
    return state


def _gb(n: float) -> str:
    return f"{n / 1e9:.1f} GB"


def render(s: dict) -> str:
    lines = []
    lines.append("recipes: " + (", ".join(s["recipes"]) or "(none)"))

    lines.append("\nvenvs:")
    for v, ok in s["venvs"].items():
        lines.append(f"  {'built   ' if ok else 'ABSENT  '} {v}")

    e = s["external"]
    if e.get("present"):
        lines.append(f"\nexternal data: present ({e['files']} files, "
                     f"{_gb(e['bytes'])})")
        for name, ent in e["entries"].items():
            lines.append(f"  {name}: {ent['files']} files, {_gb(ent['bytes'])}")
    else:
        lines.append(f"\nexternal data: ABSENT - {e['fix']}")

    lines.append("\nrecordings (per speaker; runon kept separate):")
    for word, r in s["recordings"].items():
        lines.append(f"  [{word}]")
        for kind in ("samples", "holdout"):
            sp = r[kind]
            if not sp:
                lines.append(f"    {kind}: (none)")
            else:
                total = sum(sp.values())
                per = " ".join(f"{k} {v}" for k, v in sp.items())
                lines.append(f"    {kind}: {total} ({per})")

    lines.append("\nruns:")
    for word, r in s["runs"].items():
        tags = r["last_run_tag"]
        lines.append(f"  [{word}] last run tag: "
                     + ", ".join(f"{t}={tags[t] or '(none)'}" for t in TARGETS))
        last = r["last_ledger_row"]
        if last:
            gate = f"  gates failed: {last['gates_failed']}" if last.get("gates_failed") else "  gates: all pass"
            wt = f"  wall {last['wall_time_s']}s" if "wall_time_s" in last else ""
            lines.append(f"    last ledger row: {last['target']} "
                         f"{last['tag']} seed={last.get('seed')} "
                         f"{last.get('recorded_utc', '')} {wt} {gate}")
        elif r["ledger_rows"]:
            lines.append(f"    last ledger row: (unparseable, {r['ledger_rows']} rows)")
        else:
            lines.append("    last ledger row: (no runs.jsonl)")

    d = s["deploy"]
    lines.append(f"\ndeploy: {d['scorecard_rows']} scorecard row(s); "
                 f"{len(d.get('staged_files', []))} staged file(s)")
    for key, rows in d.get("by_target", {}).items():
        for r in rows:
            lines.append(f"  {key}: {r['id']} tag={r.get('tag')} "
                         f"thr={r.get('threshold')} status={r.get('status')}")
    for f in d.get("staged_files", []):
        lines.append(f"  {f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", action="store_true",
                    help="machine-readable; the agent path")
    args = ap.parse_args()
    state = collect()
    if args.json:
        json.dump(state, sys.stdout, indent=2, sort_keys=True)
        print()
    else:
        print(render(state))


if __name__ == "__main__":
    main()

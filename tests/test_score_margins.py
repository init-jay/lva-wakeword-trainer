"""Guards for tools/score_margins.py's artifact resolution and threshold choice.

WHY THESE EXIST. On 2026-09-24 an arm of the balance sweep was scored against the
wrong model and the verdict came out backwards twice: first "the flat arm produced
models that fire on all 298 clips", then "balancing is a big win on microWakeWord".
Both readings came from paths that resolved to a different run's weights, silently,
because of how mww runs lay out on disk: every manifest is named `hey_seeree.json`
and every model inside the run dir is named `stream_state_internal_quant.tflite`, so
one wrong path component still opens *a* model and prints numbers that belong to
neither the model you meant nor the arm you are judging.

The fix is two things, and both are asserted here: the resolved artifact and its md5
are echoed next to every reading, and a manifest naming a `.tflite` that is not there
is a hard error instead of a shrug.
"""

import hashlib
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import score_margins as sm  # noqa: E402


def _pair(root, manifest_body, tflite_bytes=b"weights-A", name="hey_seeree"):
    """A run dir laid out the way the mww trainer lays one out."""
    root = Path(root)
    (root / f"{name}.json").write_text(json.dumps(manifest_body))
    artifact = root / "stream_state_internal_quant.tflite"
    if tflite_bytes is not None:
        artifact.write_bytes(tflite_bytes)
    return root / f"{name}.json", artifact


def test_a_manifest_resolves_to_the_tflite_it_names():
    with tempfile.TemporaryDirectory() as d:
        manifest, artifact = _pair(d, {"model": "stream_state_internal_quant.tflite"})
        path, digest = sm.artifact_of(manifest)
        assert path == artifact, f"must follow the manifest, not the argument: {path}"
        assert digest == hashlib.md5(artifact.read_bytes()).hexdigest()[:8], digest


def test_a_manifest_naming_a_missing_tflite_is_a_hard_error():
    # The mispairing case: a manifest from run A whose sibling was deleted, or a path
    # typed from run B. Silently scoring *something* is what produced the backwards
    # verdict, so this must exit non-zero and name what it looked for.
    with tempfile.TemporaryDirectory() as d:
        manifest, _ = _pair(d, {"model": "stream_state_internal_quant.tflite"},
                            tflite_bytes=None)
        try:
            sm.artifact_of(manifest)
        except SystemExit as exc:
            msg = str(exc)
            assert "stream_state_internal_quant.tflite" in msg, \
                f"must name the artifact it could not find: {msg}"
            assert str(Path(d)) in msg, f"must name the directory it looked in: {msg}"
        else:
            raise AssertionError("a manifest naming a missing model must not resolve")


def test_a_manifest_with_no_model_key_is_a_hard_error():
    with tempfile.TemporaryDirectory() as d:
        manifest, _ = _pair(d, {"feature_step_size": 10}, tflite_bytes=None)
        try:
            sm.artifact_of(manifest)
        except SystemExit as exc:
            assert "names no model" in str(exc), str(exc)
        else:
            raise AssertionError("a manifest with no `model` key must not resolve")


def test_two_runs_same_basename_still_tell_apart():
    # Both files are named stream_state_internal_quant.tflite, which is exactly why
    # basename is not an identity. The digest is.
    with tempfile.TemporaryDirectory() as d:
        a = Path(d) / "runA"
        b = Path(d) / "runB"
        a.mkdir()
        b.mkdir()
        ma, _ = _pair(a, {"model": "stream_state_internal_quant.tflite"}, b"A")
        mb, _ = _pair(b, {"model": "stream_state_internal_quant.tflite"}, b"B")
        pa, da = sm.artifact_of(ma)
        pb, db = sm.artifact_of(mb)
        assert pa.name == pb.name, "the collision this guards is the basename"
        assert da != db, f"two different models must not share a digest: {da}"
        assert str(pa.parent) != str(pb.parent)


def test_a_bare_model_path_resolves_to_itself():
    with tempfile.TemporaryDirectory() as d:
        _, artifact = _pair(d, {"model": "stream_state_internal_quant.tflite"})
        path, digest = sm.artifact_of(artifact)
        assert path == artifact and len(digest) == 8


def test_operating_threshold_takes_the_lowest_one_within_budget():
    # The budget is a count of adversarial fires, not a rate: 5/298 is what FA<2%
    # means at this clip count. Ascending grid, so the first pass is the loosest
    # threshold that still buys detection - never a fixed 0.5 (CLAUDE.md).
    grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    adv = [0.95, 0.85, 0.65, 0.55, 0.50, 0.35, 0.20]
    # fires(t) = count(p >= t): 7 at 0.2, 5 at 0.4, 3 at 0.6, 2 at 0.7, 1 at 0.9, 0 at 1.0
    assert sm.operating_threshold(adv, 0, grid=grid) == 1.0
    assert sm.operating_threshold(adv, 1, grid=grid) == 0.9
    assert sm.operating_threshold(adv, 2, grid=grid) == 0.7, "first threshold at <=2 fires"
    assert sm.operating_threshold(adv, 5, grid=grid) == 0.4
    assert sm.operating_threshold([0.50], 0, grid=[0.50]) is None, \
        "a peak exactly AT the threshold counts as a fire"
    assert sm.operating_threshold([0.50], 1, grid=[0.50]) == 0.50


def test_an_unreachable_budget_reports_none_instead_of_a_lying_number():
    # A model that fires on every adversarial clip at every threshold has no
    # operating point. Returning 0.6 anyway is what let a "matched-FA" reading be
    # quoted for a model that cannot meet the FA budget; report() turns None into a
    # BEST-AVAILABLE / NOT AT BUDGET line instead.
    adv = [0.99] * 20
    assert sm.operating_threshold(adv, 1, grid=[0.1, 0.5, 0.9]) is None


def main():
    import _runner
    _runner.run(sys.modules[__name__])


if __name__ == "__main__":
    main()

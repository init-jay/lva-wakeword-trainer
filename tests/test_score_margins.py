"""Guards for tools/score_margins.py's artifact resolution, threshold and budget choice.

WHY THESE EXIST. An mww run dir lays out so that one wrong path component still opens
*a* model and prints numbers: every manifest is named `<word>.json` and every model
inside the run dir is named `stream_state_internal_quant.tflite`. Score a manifest from
one run against another run's weights and the report is silently about neither - and a
verdict built on it can come out backwards rather than merely noisy.

The fix is three things, and all are asserted here: the resolved artifact and its md5
are echoed next to every reading; a manifest naming a `.tflite` that is not there is a
hard error instead of a shrug; and a model inside the shared corpus tree is refused
outright, because those bytes belong to whichever run converted last.
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
    # The budget is a COUNT of adversarial fires, not a rate; default_budget() turns the
    # rate into one. Ascending grid, so the first pass is the loosest threshold that still
    # buys detection - never a fixed threshold (CLAUDE.md).
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


def test_default_budget_stays_strictly_inside_the_constraint():
    # Derived, not hardcoded: a count that is right for one adversarial set's size is
    # wrong for the next one, and a budget that lands exactly ON the constraint reads as
    # inside it when it is not. ceil(n * c) - 1 is the largest count strictly inside c.
    assert sm.default_budget(298) == 5, sm.default_budget(298)
    # 6/300 is exactly 2%, which is not strictly inside 2% - so 5, not 6.
    assert sm.default_budget(300) == 5, sm.default_budget(300)
    assert sm.default_budget(1000) == 19, sm.default_budget(1000)
    # A set so small that even one fire breaches the constraint gets a zero budget
    # rather than a negative one: nothing may fire.
    assert sm.default_budget(50) == 0, sm.default_budget(50)
    assert sm.default_budget(0) == 0
    for n in (10, 137, 298, 366, 1000, 4096):
        b = sm.default_budget(n)
        assert b >= 0 and b / n < sm.ADV_FA_CONSTRAINT, (n, b)


def test_the_curve_table_has_a_column_for_every_holdout_speaker():
    # The table exists to show what the FA budget costs the speaker with the fewest
    # clips, so its columns come from the holdout, not from a pair of names in the
    # source. A third speaker the author's holdout did not have must still get a column
    # - dropping it is how the pooled reading survives as the only one.
    peaks = {"adult_a": [("a0", 0.9), ("a1", 0.1)],
             "adult_b": [("b0", 0.7)],
             "child": [("c0", 0.05), ("c1", 0.02), ("c2", 0.4)]}
    speakers = list(peaks)
    head = sm.curve_header(speakers)
    for s in speakers:
        assert s in head, f"{s} has no column in: {head}"
    cells = sm.curve_cells(peaks, speakers, 0.3)
    assert cells.count("/") == len(speakers), cells
    # k/n per speaker, in order, at that threshold (adult_a's 0.1 peak is below it)
    assert cells.split() == ["1/2", "1/1", "1/3"], cells


def test_a_corpus_tree_model_is_refused():
    # data/corpus/<word>/mww/<corpus-id>/tflite_stream_state_internal_quant/ holds ONE
    # copy of the weights per corpus: every run built against that corpus writes its
    # weights there, so the bytes belong to whichever run converted last, not to the
    # model being asked about. Several "different" runs then score as one file, and an
    # arm judged against another arm's weights reads backwards - which is why this path
    # is refused rather than warned about.
    with tempfile.TemporaryDirectory() as d:
        root, saved = Path(d), sm.REPO_ROOT
        try:
            sm.REPO_ROOT = root
            shared = (root / "data" / "corpus" / "hey_seeree" / "mww" / "c574e978" /
                      "tflite_stream_state_internal_quant" /
                      "stream_state_internal_quant.tflite")
            shared.parent.mkdir(parents=True)
            shared.write_bytes(b"last-run-wins")
            try:
                sm.artifact_of(shared)
            except SystemExit as exc:
                msg = str(exc)
                assert "corpus tree" in msg, f"must say why it refused: {msg}"
                assert "output/" in msg, f"must point at the run dir: {msg}"
            else:
                raise AssertionError("a model inside data/corpus/ must not be scored")

            # The guard must not catch the thing it exists to force: the per-run copy.
            run = (root / "output" / "hey_seeree" / "mww" / "c574e978-h5835241" /
                   "stream_state_internal_quant.tflite")
            run.parent.mkdir(parents=True)
            run.write_bytes(b"per-run")
            path, digest = sm.artifact_of(run)
            assert path == run, f"the run-dir copy must still resolve: {path}"
            assert digest == hashlib.md5(b"per-run").hexdigest()[:8], digest
        finally:
            sm.REPO_ROOT = saved


def _fields(line, speakers):
    """The line cut at the width spec's column boundaries, indent included.

    Cut BY POSITION rather than split on whitespace: right-aligned fields ABUT when a
    speaker name is as wide as its column, so split() merges two columns into one token
    and reports a misalignment that is not there. The 4-space indent both lines start
    with is not a column and is skipped.
    """
    widths = [sm.CURVE_THR_W, sm.CURVE_FA_W, sm.CURVE_POOLED_W]
    widths += [sm.curve_widths(speakers)] * len(speakers)
    fields, pos = [], 4
    for w in widths:
        fields.append(line[pos:pos + w])
        pos += w
    assert pos == len(line), f"the width spec does not account for the whole line: {line!r}"
    return fields


def assert_columns_flush(head, row, speakers):
    """Every field ends at its column boundary, on both sides.

    Right-aligned content that stops short of the boundary is the misalignment: it means
    that side formatted the column narrower than the shared spec, so everything to its
    right reads under the wrong label.
    """
    for i, (h, r) in enumerate(zip(_fields(head, speakers), _fields(row, speakers))):
        assert h == h.rstrip(), f"header column {i} ends short of its boundary: {head!r}"
        assert r == r.rstrip(), f"row column {i} ends short of its boundary: {row!r}"


def test_the_curve_header_and_body_share_one_width_spec():
    """Header and body are rendered by the SAME width spec, pinned by column offset.

    The header used to render 'advFA'/'pooled' one and two characters narrower than the
    rows rendered them, so every column after the threshold read two characters to the
    left of its label - 'pooled' sat over the advFA numbers and the first speaker's label
    over the pooled ones. Nothing could see it: sm.curve_header was tested, the rows were
    an f-string inside report(), and the two were only ever printed together.
    """
    peaks = {"a": [("a0", 0.9)],
             "averylongspeakername": [("b0", 0.8), ("b1", 0.1)]}
    speakers = list(peaks)
    head = sm.curve_header(speakers).partition("   (each")[0]   # drop the legend
    row = sm.curve_row(0.5, 12, 298, 40, 51, peaks, speakers)
    assert_columns_flush(head, row, speakers)
    # ... and the cells still carry the same information the old inline f-string did.
    assert row.split() == ["0.50", "12/298", "78%", "1/1", "1/2"], row


def test_the_column_width_spec_follows_the_labels():
    """A long name widens the columns, it does not push them out of alignment.

    This is the case where a speaker name is WIDER than the default column, so the label
    and its cell abut the neighbouring column with no padding between them.
    """
    wide = {"averyveryverylongspeakername": [("a0", 0.9)]}
    speakers = list(wide)
    head = sm.curve_header(speakers).partition("   (each")[0]
    row = sm.curve_row(0.4, 1, 10, 5, 5, wide, speakers)
    assert sm.curve_widths(speakers) > sm.CURVE_SPEAKER_W
    assert_columns_flush(head, row, speakers)


def test_an_empty_adversarial_set_refuses_naming_the_directory():
    """The refusal is pinned at the helper, not at report(): no audio in the test tree.

    A name mismatch loads an EMPTY set rather than failing - a clip whose name misses
    CATEGORY_RE is filed under '(uncategorised)' and is not adversarial - so the guard is
    the only thing between that and a ZeroDivisionError printed eight lines after a
    header whose budget was derived from n=0. That was the observed failure mode.
    """
    try:
        sm.require_measurable([], [0.5], "data/negatives", ["pos_dir"])
    except SystemExit as e:
        assert "data/negatives" in str(e), e
        assert "extend" in str(e) and "hey_other" in str(e), e
        assert "ZeroDivision" not in str(e), e
    else:
        raise AssertionError("an empty adversarial set must refuse, not divide by zero")


def test_empty_positives_refuse_naming_the_directories():
    try:
        sm.require_measurable([0.5], [], "neg_dir",
                              ["data/recordings/holdout/jay"])
    except SystemExit as e:
        assert "data/recordings/holdout/jay" in str(e), e
    else:
        raise AssertionError("an empty positive set must refuse")


def main():
    import _runner
    _runner.run(sys.modules[__name__])


if __name__ == "__main__":
    main()

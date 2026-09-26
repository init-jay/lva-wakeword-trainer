"""Patch microwakeword's test.py to reset the streaming model between clips.

tflite_streaming_model_roc() streams the ambient and positive clips through ONE
Model with persistent state, so the "fresh session" cold-start context (zero
ring-buffer history) happens once per whole stream instead of once per clip.
The deployment runtime (pymicro_wakeword / LVA) and this repo's eval harness
both reset per clip, so the in-run ROC measures a condition deployment never
runs: a model that fires on every cold start scores near-chance faph here and
is only caught at the gate (data/lva-mww-cause/report.md - ~1 in 5 of the
existing models carries the transient; failed run 2e907ac scored AUC 0.177
in-run yet 292/298 adversarial at the 0.5 gate).

The reset is a FRESH Model PER CLIP, not an interpreter reset call, because
there is no interpreter reset call:
- ai-edge-litert 2.2.0's Interpreter exposes only reset_all_variables();
  there is no .reset() and no .reset_internal_state() (verified against the
  installed package).
- reset_all_variables() does not reset the streaming state: this repo's
  eval backend already measured "an interpreter that has
  reset_all_variables() called and its inputs re-zeroed still scores 1.00000
  the first time and 0.99608 every time after" (src/eval/src/backends.py,
  module docstring). Measured again on the pinned model here: the ring
  buffers live in the StatefulPartitionedCall custom op (tensors
  streaming/stream_*/states_1, dtype object - not reachable through the
  tensor API), and the call zeros the int8 weight variables, so a model
  reset this way outputs 0.0 forever, fresh or warmed.
- Both deployment (pymicro_wakeword.MicroWakeWord.reset(): "Need to reload
  model to reset intermediary results") and this repo's eval backend
  (backends.py MicroWakeWordBackend.start()) therefore reset by reloading
  the model. Constructing a new Model - new Interpreter, same .tflite
  bytes - is that reload; its cold-start state is exactly the deployment
  fresh-session state.

Applied to both loops (ambient and positive), so the ROC's faph integral and
cutoff selection reflect the per-clip cold starts of deployment. One more
hunk, in compute_false_accepts_per_hour (the only caller is this function):
the per-track cooldown started at ignore_slices_after_accept (25 slices =
750 ms at stride 3, 10 ms steps), silently skipping the first 750 ms of
every ambient clip - exactly where the cold-start transient lives (report:
peak within ~150 ms of session start). With per-clip resets, each clip IS a
fresh session and its cold-start region must count as a false accept, so the
cooldown now starts at 0; the post-detection 25-slice cooldown is kept.
"""
import sys

path = sys.argv[1]
with open(path) as f:
    content = f.read()

INIT_OLD = '''\
    stride = config["stride"]
    model = Model(
        os.path.join(config["train_dir"], folder, tflite_model_name), stride=stride
    )'''

INIT_NEW = '''\
    stride = config["stride"]
    # Per-clip reset, matching deployment: the streaming state can only be
    # reset by reloading the interpreter (see the module docstring of this
    # patch for the measurements), so each clip below constructs a fresh
    # Model from the same .tflite bytes. Without this the fresh-session
    # cold start happens once per whole stream, and a model that fires on
    # every cold start scores near-chance faph here while firing on every
    # clip at the gate.
    model_path = os.path.join(config["train_dir"], folder, tflite_model_name)'''

AMBIENT_OLD = '''\
    for spectrogram_track in test_ambient_fingerprints:
        streaming_probabilities = model.predict_spectrogram(spectrogram_track)'''

AMBIENT_NEW = '''\
    for spectrogram_track in test_ambient_fingerprints:
        # Fresh Model per clip: the deployment fresh-session cold start.
        model = Model(model_path, stride=stride)
        streaming_probabilities = model.predict_spectrogram(spectrogram_track)'''

POSITIVE_OLD = '''\
        if test_ground_truth[i]:
            # Only test positive samples
            streaming_probabilities = model.predict_spectrogram(test_fingerprints[i])'''

POSITIVE_NEW = '''\
        if test_ground_truth[i]:
            # Only test positive samples
            # Fresh Model per clip: the deployment fresh-session cold start.
            model = Model(model_path, stride=stride)
            streaming_probabilities = model.predict_spectrogram(test_fingerprints[i])'''

COOLDOWN_OLD = '''\
        cooldown_at_cutoffs = np.ones(cutoffs_count) * ignore_slices_after_accept'''

COOLDOWN_NEW = '''\
        # A fresh track is a fresh session, not a post-detection cooldown:
        # with the per-clip Model reset below, every clip starts with zero
        # ring-buffer history, so start the cooldown at 0 and let the
        # cold-start region count as a false accept. Starting it at
        # ignore_slices_after_accept skips the first 25 slices of every clip
        # (750 ms at stride 3, 10 ms steps) - exactly the window the learned
        # cold-start transient lives in (data/lva-mww-cause/report.md: peak
        # within ~150 ms of session start) - and would hide the very failure
        # the reset exists to surface. The post-detection cooldown (the
        # assignment after a counted accept) is unchanged.
        cooldown_at_cutoffs = np.zeros(cutoffs_count)'''

# The applied-state check comes FIRST: once applied, all four `old` anchors
# are gone and a bare anchor check would misreport an applied patch as
# "target not found" on every setup re-run.
if INIT_NEW in content and AMBIENT_NEW in content and POSITIVE_NEW in content and COOLDOWN_NEW in content:
    print("Already patched:", path)
    sys.exit(0)

missing = []
if INIT_OLD not in content:
    missing.append("Model init")
if AMBIENT_OLD not in content:
    missing.append("ambient loop")
if POSITIVE_OLD not in content:
    missing.append("positive loop")
if COOLDOWN_OLD not in content:
    missing.append("faph cooldown init")
if missing:
    print(
        "WARNING: patch target not found in {} ({}). If the model was "
        "applied by hand or a different patch version, verify both loops "
        "construct a fresh Model before each clip and the per-track faph "
        "cooldown starts at 0.".format(
            path, " and ".join(missing)
        )
    )
    sys.exit(0)

content = content.replace(INIT_OLD, INIT_NEW)
content = content.replace(AMBIENT_OLD, AMBIENT_NEW)
content = content.replace(POSITIVE_OLD, POSITIVE_NEW)
content = content.replace(COOLDOWN_OLD, COOLDOWN_NEW)
with open(path, 'w') as f:
    f.write(content)

print("Patched:", path)

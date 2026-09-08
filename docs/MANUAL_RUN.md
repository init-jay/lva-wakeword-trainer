# Running the pipeline manually

The pipeline is designed to be driven by an agent — see `.claude/skills/`. This is
the same four steps done by hand.

Steps 1 and 4 need a microphone and run on your machine; steps 2 and 3 run in
Docker. Replace `"hey seeree"` with your wake word throughout. For what happens
inside each step, see [../ARCHITECTURE.md](../ARCHITECTURE.md).

**0 · Wordlists and data.** The phrases are the one part that does not transfer
between wake words — `extend`, `running` and `hey_other` are built from your phrase's
own consonants and vowels. Copy `wordlists/hey_seeree.yaml`, rewrite those three, then
fetch the third-party corpora (~43 GB for both trainers; `oww` or `mww` alone is less):

```bash
./scripts/download-external-data.sh            # all | oww | mww
```

**1 · Record.** 20–50 clips per speaker, aiming for a peak near −12 dBFS. Record the
holdout in the same session — it is evaluated against and never trained on.

```bash
cd record
uv run record_samples.py --list-devices
uv run record_samples.py --wake-word "hey seeree" --device 0 --speaker speaker1
uv run record_samples.py --wake-word "hey seeree" --device 0 --holdout --speaker speaker1
uv run python check_alignment.py ../data/recordings/samples/speaker1
```

More than one speaker matters more than more clips from one: a model that reads 97%
for an adult has measured 24% for a child.

**2 · Train.** One script per target; they are independent and can run separately.

```bash
./scripts/run-oww-training.sh "hey seeree"     # server — one command, builds its own corpus
./scripts/run-mww-training.sh "hey seeree"     # ESP32  — chains corpus, features, train, manifest
```

Both name their output after the code *and* the audio that produced it, so two runs
from the same commit with different recordings are told apart. `SKIP_BUILD=1` reuses
the image; see each script's header for the other escapes.

If a run dies *after* the corpus is built — a CUDA OOM at the feature array is the
usual way — resume without paying for the TTS again:

```bash
SKIP_CORPUS=1 ./scripts/run-oww-training.sh "hey seeree"
SKIP_CORPUS=1 ./scripts/run-mww-training.sh "hey seeree"
```

For resuming, not for tuning. Training flags still apply; the ones that shape the
corpus are inert, because the clips already exist.

**3 · Eval.** Runs on whichever machine you are sitting at. Generate the adversarial
corpus once, then score:

```bash
docker compose up -d kokoro
docker compose run --rm eval python -m eval.generate_negatives \
    --url http://kokoro:8880/v1/audio/speech
docker compose stop kokoro

# the four gates, one model
docker compose run --rm eval python -m eval.eval_model \
    --model output/hey_seeree/oww/<tag>.onnx

# is this run better than the last one?
docker compose run --rm eval python -m eval.compare_models \
    --models output/hey_seeree/oww/<new>.onnx output/hey_seeree/oww/<previous-best>.onnx

# the two targets side by side: server vs ESP32
docker compose run --rm eval python -m eval.compare_models --models \
    output/hey_seeree/oww/<tag>.onnx \
    output/hey_seeree/mww/<tag>.json

# one model, swept, to pick a deployment operating point
docker compose run --rm eval python -m eval.compare_models --sweep \
    --models output/hey_seeree/mww/<tag>.json
```

Comparing the two targets is legitimate *only* through matched false accepts. An
openWakeWord score is a raw probability and a microWakeWord score is a sliding-window
average of an int8 output — they share no threshold scale, so the fixed-0.5 row is
meaningless across backends as well as across runs. "Detection at the operating point
that admits N adversarial false accepts" is the same question asked of both.

Pass the microWakeWord **`.json`**, not its `.tflite`: the runtime reads
`probability_cutoff` and `sliding_window_size` from the manifest, so scoring the JSON
puts the manifest under test too.

**Read the per-speaker table, not just the totals.** A pooled number is an average
over speakers, and an average is what hides the person the model does not work for.
Compare models at matched false-accept counts — never at a fixed threshold, where two
runs of an identical configuration have measured 77% and 67%.

**4 · Preflight.** Scores your room, your mic and you actually speaking — variables no
corpus contains. Pass the `.tflite` or the microWakeWord `.json`; the `.onnx` is
refused because it is not what the device runs.

```bash
cd preflight
uv run test_model.py --model ../output/hey_seeree/mww/<tag>.json
```

If peaks sit just under your threshold, the operating point is wrong for the room —
not the model.

### Picking an overlay for the training step

Nothing in the base compose file requires a GPU, so evaluation and recording work
anywhere. Training is the one step that cares, and it picks a set of images:

```bash
# NVIDIA box. Will not match an AMD card under ROCm - `driver: nvidia` is a CUDA
# reservation and has no vendor-neutral spelling.
export COMPOSE_FILE=docker-compose.yml:docker-compose.cuda.yml

# Everything else: Apple Silicon, AMD, or a CPU-only Linux box. Multi-arch images,
# native on arm64, no emulation.
export COMPOSE_FILE=docker-compose.yml:docker-compose.cpu.yml
```

Set one per shell, then use the scripts in `scripts/` unchanged — they read
`COMPOSE_FILE` like every other compose command.

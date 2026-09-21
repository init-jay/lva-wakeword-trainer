# The one entry point: the commands a human actually types in this repo, each
# annotated with what it costs - minutes, GB, or which TTS servers it needs.
#
# Thin wrappers only: the scripts keep their logic and their comments
# (improvement.md P2.5), and CLAUDE.md carries the ordering between steps.
# The costs below are the Apple-Silicon host numbers from SPEED.md where
# measured; "needs TTS" means the uv engines in tts-service/engines/, in
# another terminal, per the run-script headers.

WAKE ?= hey seeree
PY_OWW := train-applesilicon/.venv/bin/python
PY_MWW := train-mww-applesilicon/.venv/bin/python

.DEFAULT_GOAL := help

.PHONY: help smoke-oww smoke-mww test fleet eval corpus-oww corpus-mww

help:
	@echo "make help           this list"
	@echo "make smoke-oww      end-to-end check of the oww pipeline; a few minutes to ~15 min on a Mac"
	@echo "                    (corpus REUSED, no TTS; the cost is the forced feature recompute, ~12 min)"
	@echo "make smoke-mww      end-to-end check of the mww pipeline; ~5 min on a Mac, no TTS, corpus+features REUSED"
	@echo "make test           the test suite (45 tests); seconds, no data needed"
	@echo "make fleet N=<n>    start N Piper TTS instances and print PIPER_URLS for the corpus runs;"
	@echo "                    needs data/external (download-external-data.sh), the processes stay up until killed"
	@echo "make eval           builds the eval Docker image; scoring NEEDS Docker on a Mac (the host path,"
	@echo "                    improvement.md P2.4, is not built yet) plus the Kokoro engine on 8900"
	@echo "make corpus-oww     a FULL oww run: the oww corpus is generated inside the trainer, so there is"
	@echo "                    no corpus-only step; ~35-90 min on a Mac (TTS dominates), needs Kokoro on 8900"
	@echo "make corpus-mww     the standalone mww corpus stage; ~5-10 min of TTS (faster with a fleet),"
	@echo "                    needs Piper on 8898 and, for the default 0.3 mix, Kokoro on 8900"
	@echo ""
	@echo "WAKE overrides the wake word (default: $(WAKE))."

# The full pipeline at minified size: reuse the corpus, recompute the features,
# train 200 steps, do the real tflite conversion. The model lands in a smoke
# directory; the canonical model and .last_run_tag are untouched (train/oww/
# train.py --smoke).
smoke-oww:
	SMOKE=1 ./scripts/run-oww-training-applesilicon.sh "$(WAKE)"

# Same, for the mww side: the corpus goes through its --skip path and the
# pre-built features are reused, so nothing needs a TTS server at all.
smoke-mww:
	SMOKE=1 ./scripts/run-mww-training-applesilicon.sh "$(WAKE)"

# No venv in this repo carries pytest (tests/_runner.py), so the suite is plain
# `python tests/test_<x>.py` runs; stop at the first failing file.
test:
	@for t in tests/test_*.py; do $(PY_OWW) "$$t" || exit 1; done

# scripts/start-tts-fleet.sh prints the comma-joined list PIPER_URLS wants;
# the instances keep running in the background until killed.
fleet:
	@test -n "$(N)" || { echo "usage: make fleet N=<number of Piper instances>"; exit 1; }
	./scripts/start-tts-fleet.sh $(N)

# The eval-models skill's path is the Docker image; there is no host invocation
# yet (P2.4), so this builds it and points at the scoring commands.
eval:
	@echo "eval scoring needs the Docker image on a Mac (no host path yet - P2.4)."
	cd eval && docker compose build
	@echo "Score: cd eval && docker compose run --rm eval python -m eval.eval_model --model output/$(WAKE)/oww/<model>.onnx"

# oww has no standalone corpus module: generation is the first stage of
# train/oww/train.py, so "the real corpus stage" here is a full run (TTS
# generation, features, 50k steps, conversion). Reuse instead: --skip-corpus.
corpus-oww:
	./scripts/run-oww-training-applesilicon.sh "$(WAKE)"

# The mww corpus IS standalone. PIPER_URLS makes a fleet of them (make fleet
# N=<n> prints the list); the default is the single in-process engine on 8898.
corpus-mww:
	@if [ -n "$(PIPER_URLS)" ]; then PURLS="--piper-url $(PIPER_URLS)"; else PURLS="--piper-url tcp://127.0.0.1:8898"; fi; \
	$(PY_MWW) -m train.mww.corpus --wake-word "$(WAKE)" $${PURLS} --piper-speakers 12 \
	    --kokoro-url tcp://127.0.0.1:8900 --kokoro-fraction 0.3

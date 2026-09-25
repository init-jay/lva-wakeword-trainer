# The one entry point: the commands a human actually types in this repo, each
# annotated with what it costs - minutes, GB, or which TTS servers it needs.
#
# Thin wrappers only: the scripts keep their logic and their comments
# (improvement.md P2.5), and CLAUDE.md carries the ordering between steps.
# The costs below are the Apple-Silicon host numbers from SPEED.md where
# measured; "needs TTS" means the uv engines in src/tts-service/engines/, in
# another terminal, per the run-script headers.

WAKE ?= hey seeree
PY_OWW := src/train/train-applesilicon/.venv/bin/python
PY_MWW := src/train/train-mww-applesilicon/.venv/bin/python

.DEFAULT_GOAL := help

.PHONY: help smoke-oww smoke-mww test fleet eval eval-docker corpus-oww corpus-mww render-voice-holdout

help:
	@echo "make help           this list"
	@echo "make smoke-oww      end-to-end check of the oww pipeline; a few minutes to ~15 min on a Mac"
	@echo "                    (corpus REUSED, no TTS; the cost is the forced feature recompute, ~12 min)"
	@echo "make smoke-mww      end-to-end check of the mww pipeline; ~5 min on a Mac, no TTS, corpus+features REUSED"
	@echo "make test           the test suite (52 tests); seconds, no data needed"
	@echo "make fleet N=<n>    start N Piper TTS instances and print PIPER_URLS for the corpus runs;"
	@echo "                    needs data/external (download-external-data.sh), the processes stay up until killed"
	@echo "make eval           scoring on the HOST env (the Mac default, P2.4): sets it up if"
	@echo "                    missing, then src/eval/.venv/bin/python src/eval/src/eval_model.py --model ..."
	@echo "                    no Docker, no TTS - scoring reads the rendered corpus (Kokoro on 8900"
	@echo "                    is for generation only)"
	@echo "make eval-docker    the container route - right on a CUDA box or an eval-only machine"
	@echo "make corpus-oww     a FULL oww run: the oww corpus is generated inside the trainer, so there is"
	@echo "                    no corpus-only step; ~35-90 min on a Mac (TTS dominates), needs Kokoro on 8900"
	@echo "make corpus-mww     the standalone mww corpus stage; ~5-10 min of TTS (faster with a fleet),"
	@echo "                    needs Piper on 8898 and, for the default 0.3 mix, Kokoro on 8900"
	@echo "make render-voice-holdout  the voice-holdout synthetic RANKING set (improvement.md P1.2): a"
	@echo "                    few seconds to ~1 min of Kokoro, into data/corpus/eval/voice_holdout_tts;"
	@echo "                    the existing corpora and the real-speaker gates stay untouched"
	@echo ""
	@echo "WAKE overrides the wake word (default: $(WAKE))."

# The full pipeline at minified size: reuse the corpus, recompute the features,
# train 200 steps, do the real tflite conversion. The model lands in a smoke
# directory; the canonical model and .last_run_tag are untouched (train/oww/
# train.py --smoke).
smoke-oww:
	SMOKE=1 ./src/scripts/run-oww-training-applesilicon.sh "$(WAKE)"

# Same, for the mww side: the corpus goes through its --skip path and the
# pre-built features are reused, so nothing needs a TTS server at all.
smoke-mww:
	SMOKE=1 ./src/scripts/run-mww-training-applesilicon.sh "$(WAKE)"

# No venv in this repo carries pytest (tests/_runner.py), so the suite is plain
# `python tests/test_<x>.py` runs; stop at the first failing file.
test:
	@for t in tests/test_*.py; do $(PY_OWW) "$$t" || exit 1; done

# src/scripts/start-tts-fleet.sh prints the comma-joined list PIPER_URLS wants;
# the instances keep running in the background until killed.
fleet:
	@test -n "$(N)" || { echo "usage: make fleet N=<number of Piper instances>"; exit 1; }
	./src/scripts/start-tts-fleet.sh $(N)

# The Mac default (P2.4): the host uv env in src/eval/, no Docker and no TTS -
# scoring reads the already-rendered negatives; only corpus generation speaks to
# Kokoro. The setup script runs only when the venv is missing; it is idempotent.
# The model directories are keyed hey seeree -> hey_seeree; $(subst  ,_,...) cannot
# do that - Make trims the leading space out of subst's first argument - so the
# conversion runs in shell.
eval:
	@test -d src/eval/.venv || ./src/scripts/setup-eval-host.sh
	@WAKE_DIR=$$(echo "$(WAKE)" | tr ' ' '_') ; \
	echo "Score:   src/eval/.venv/bin/python src/eval/src/eval_model.py --model output/$$WAKE_DIR/oww/<model>.onnx"
	@echo "Compare: src/eval/.venv/bin/python src/eval/src/compare_models.py --models <new> <previous-best>"

# The container route: the right one on a CUDA box or an eval-only machine, and
# on a Mac the alternative to the host env. It pins the same deployment-runtime
# wheels as src/eval/pyproject.toml (they must stay equal), so the numbers check
# against each other.
eval-docker:
	cd src/eval && docker compose build
	@WAKE_DIR=$$(echo "$(WAKE)" | tr ' ' '_') ; \
	echo "Score: cd src/eval && docker compose run --rm eval python -m eval.eval_model --model output/$$WAKE_DIR/oww/<model>.onnx"

# oww has no standalone corpus module: generation is the first stage of
# src/train/oww/train.py, so "the real corpus stage" here is a full run (TTS
# generation, features, 50k steps, conversion). Reuse instead: --skip-corpus.
corpus-oww:
	./src/scripts/run-oww-training-applesilicon.sh "$(WAKE)"

# The mww corpus IS standalone. PIPER_URLS makes a fleet of them (make fleet
# N=<n> prints the list); the default is the single in-process engine on 8898.
corpus-mww:
	@if [ -n "$(PIPER_URLS)" ]; then PURLS="--piper-url $(PIPER_URLS)"; else PURLS="--piper-url tcp://127.0.0.1:8898"; fi; \
	PYTHONPATH=src $(PY_MWW) -m train.mww.corpus --wake-word "$(WAKE)" $${PURLS} --piper-speakers 12 \
	    --kokoro-url tcp://127.0.0.1:8900 --kokoro-fraction 0.3

# The voice-holdout synthetic ranking set (improvement.md P1.2): its own output
# directory (data/corpus/eval/voice_holdout_tts) from the voices
# src/wordlists/voice_holdout.yaml holds out of every corpus build, at speeds
# inside the 0.7-1.3 training range. The live Kokoro catalog is probed and a
# stale list fails loudly; the existing eval corpora are not touched.
render-voice-holdout:
	@test -d src/eval/.venv || ./src/scripts/setup-eval-host.sh
	src/eval/.venv/bin/python src/eval/src/generate_positives.py \
		--wake-word "$(WAKE)" \
		--voice-holdout \
		--out data/corpus/eval/voice_holdout_tts

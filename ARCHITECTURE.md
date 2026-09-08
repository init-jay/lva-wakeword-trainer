# High Level Pipeline Design

```mermaid
flowchart TB

    subgraph SRC ["1 · Record and check real speech — host, needs a mic"]
        direction TB
        REC["Record real speech"]
        CHK["Check real data"]
        REC --> CHK
        CHK --> TRAINDATA[("Training data")]
        CHK --> HOLDDATA[("Holdout data<br/>NEVER trained on")]
    end

    subgraph RUN ["2 · Training run"]
        direction TB
        GEN["Generate corpus<br/>real + synthetic"]
        subgraph TRN ["Train model"]
            direction LR
            OWW["openWakeWord<br/>server"]
            MWW["microWakeWord<br/>ESP32 / edge"]
        end
        GEN --> OWW
        GEN --> MWW
    end

    TRAINDATA --> RUN

    subgraph EVALBOX ["3 · Eval model"]
        direction TB
        EVC["Generate eval<br/>synthetic corpus"]
        EVALSET[("Eval dataset")]
        EVAL["Evaluate trained models<br/>at matched precision, per speaker"]
        EVC --> EVALSET
        EVALSET --> EVAL
    end

    HOLDDATA -->|"merge"| EVALSET
    RUN -->|"trained models"| EVALBOX

    EVALBOX -->|"not better — change ONE thing"| RUN
    EVAL -->|"better"| PRE["4 · Preflight<br/>live mic"]
    PRE --> SHIP(["Deploy"])

    classDef store fill:#eef4ff,stroke:#4a6fa5,color:#12263f
    classDef guard fill:#fff4e6,stroke:#c47f00,color:#3d2800
    class TRAINDATA,HOLDDATA,EVALSET store
    class EVAL,PRE guard
```

The two things this shape is built around: **the holdout leaves step 1 and goes
straight into the eval dataset**, never through training; and **evaluation feeds back
into the training run**, because that loop is the pipeline — seventeen runs of it so
far.

## File layout

```
.
├── docs/
│   └── MANUAL_RUN.md            the four steps by hand, if not using an agent
├── .claude/skills/               agent instructions, one per pipeline step
│   ├── record-samples/SKILL.md   1 · how to get usable recordings, and verify them
│   ├── write-wordlists/SKILL.md  2+3 · the phrases, the one part that does not transfer
│   └── eval-models/SKILL.md      3 · how to read a scorecard without misreading it
├── wordlists/                    per-wake-word phrase lists
│   ├── __init__.py               loader + the checks that keep eval and train disjoint
│   └── hey_seeree.yaml           one file per wake word
├── record/                       1 · record and check
│   ├── record_samples.py
│   ├── check_alignment.py
│   ├── pyproject.toml            its own uv env - kept together
│   └── uv.lock
├── train/                        2 · training run
│   ├── provenance.py             run tag: the commit AND the audio it trained on
│   ├── corpus/                   shared by both trainers
│   │   ├── augment.py
│   │   ├── kokoro.py             the Kokoro TTS client (both trainers)
│   │   ├── kokoro_mlx.py         the in-process mlx:// backend for it
│   │   ├── negatives.py
│   │   ├── positives.py
│   │   ├── piper.py
│   │   └── real.py
│   ├── oww/
│   │   ├── train.py
│   │   └── onnx2tflite.py
│   └── mww/
│       ├── corpus.py
│       ├── features.py
│       ├── config.py
│       ├── train.py
│       └── manifest.py
├── eval/                         3 · eval model
│   ├── paths.py                  holdout vs samples - a safety property, not a convention
│   ├── generate_negatives.py     builds the eval corpus
│   ├── generate_positives.py     builds the eval corpus
│   ├── backends.py
│   ├── eval_model.py
│   ├── compare_models.py
│   └── check_model_alignment.py
├── preflight/                    4 · preflight
│   ├── test_model.py             live mic, the deployment runtime
│   ├── pyproject.toml            its own uv env - host, like record/
│   └── uv.lock
├── deploy/                       the current deployment candidate, staged out of
│   └── README.md                 output/ with its measured status - one at a time
├── tools/                        not in the diagram - one-off measurement
│   ├── audit_voices.py
│   ├── bench_tts.py
│   ├── tf_probe.py
│   └── measure_voice_f0.py
├── patches/
├── docker/
│   ├── Dockerfile.oww.cuda       trains on an NVIDIA GPU, linux/amd64
│   ├── Dockerfile.mww.cuda       trains on an NVIDIA GPU, linux/amd64
│   ├── Dockerfile.oww.cpu        same trainer, no GPU - multi-arch, native on arm64
│   ├── Dockerfile.mww.cpu        same, on python:3.12-slim - the TF image is amd64-only
│   ├── Dockerfile.piper          CUDA base, but runs CPU-only by choice
│   ├── Dockerfile.kokoro         CPU by default; CUDA via docker-compose.cuda.yml
│   ├── Dockerfile.eval           CPU only, native on Apple Silicon
│   └── requirements.txt          shared by BOTH oww trainers, cuda and cpu
├── scripts/
│   ├── download-external-data.sh  -> data/external/  [all|oww|mww]
│   ├── run-oww-training.sh        2 · one command, corpus built by the run
│   ├── run-mww-training.sh        2 · four stages, corpus built separately
│   ├── start-kokoro-host.sh       TTS outside Docker - 3.7x the container on arm64
│   ├── start-piper-host.sh        same, for Piper - 2.4x the container on arm64
│   ├── setup-applesilicon-trainer.sh   what Dockerfile.oww.cpu does, on the host
│   ├── run-oww-training-applesilicon.sh  2 · same trainer, no container
│   ├── setup-mww-applesilicon-trainer.sh what Dockerfile.mww.cpu does, on the host
│   └── run-mww-training-applesilicon.sh  2 · four stages, no container
├── .dockerignore                 keeps data/ (~43 GB) out of every build context
├── docker-compose.yml            no GPU required, so eval and record run anywhere
├── docker-compose.cuda.yml       overlay: NVIDIA devices - kokoro, trainers, piper
├── docker-compose.cpu.yml        overlay: CPU trainers - Apple Silicon, or any non-NVIDIA box
├── docker-compose.mps.yml        overlay: Metal - permanently empty, see apple-port.md
├── SPEED.md                      measured timings - the evidence for the README's route calls
├── train-applesilicon/           host uv env for the oww trainer - apple-port.md phase 1b
└── train-mww-applesilicon/       host uv env for the mww trainer - apple-port.md phase 3
```

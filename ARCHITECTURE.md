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
├── .claude/skills/               agent instructions, one per pipeline step
│   ├── record-samples/SKILL.md   1 · how to get usable recordings, and verify them
│   └── eval-models/SKILL.md      3 · how to read a scorecard without misreading it
├── record/                       1 · record and check
│   ├── record_samples.py
│   ├── check_alignment.py
│   ├── pyproject.toml            its own uv env - kept together
│   └── uv.lock
├── train/                        2 · training run
│   ├── corpus/                   shared by both trainers
│   │   ├── augment.py
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
│   └── test_model.py
├── tools/                        not in the diagram - one-off measurement
│   ├── audit_voices.py
│   ├── bench_tts.py
│   └── measure_voice_f0.py
├── patches/
├── docker/
│   ├── Dockerfile.oww
│   ├── Dockerfile.mww
│   ├── Dockerfile.piper
│   ├── Dockerfile.kokoro
│   └── Dockerfile.eval
└── scripts/
    ├── download-external-data.sh
    └── run-training.sh
```

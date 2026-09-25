"""Make the repo root importable for pytest (if one is ever installed).

The modules under test bootstrap `sys.path` themselves in various shapes -
src/train/corpus/__init__.py inserts src/tts-service/tts_protocol/ for the protocol
client, src/train/corpus/manifest.py inserts the root for `from train.provenance
import ...`, and the eval tools are invoked from an image where eval/src is
mounted AS the `eval` package - but a bare `import train.mww.config`,
`import wordlists` or `import eval.*` still needs the repo ROOT on the path,
and pytest does not add the rootdir for you. Each test file repeats the
insertion so the same file also runs plain (`python tests/test_x.py`), where
this file is never loaded.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

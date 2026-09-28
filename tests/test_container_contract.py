"""The container contract: what the images and mounts must agree on.

WHY THIS FILE EXISTS. Moving the per-word data to `recipes/` at the repo root
broke two things that NO test could see, and both were found by reading files
rather than by running anything:

* all four trainer images failed their own build-time import check, because
  `src/train/corpus/negatives.py` gained a module-level `import recipe` and no
  Dockerfile copied the loader (only `src/train/` and `src/tts-service/`);
* every compose file mounted the loader and nothing mounted the data, so a
  container would import cleanly and then fail at the first `load()` with
  "no recipe at /app/recipes/<word>.yaml".

Neither is reachable from a host-side test run - there is no CI here, and
`make test` does not start Docker. So the checks below are STATIC: they parse
the compose YAML and the Dockerfiles and assert the invariants the runtime
depends on. They cost milliseconds and they would have caught both.

The invariant, in one sentence: the loader finds its data at
`Path(__file__).parents[2] / "recipes"` - i.e. `<repo-root>/recipes`, computed
from where the LOADER sits, never from the cwd - so any mount of the loader at
`<root>/src/recipe` obliges a mount of the data at `<root>/recipes`, and any
image that imports `train.corpus` at build time must COPY the loader.
"""

import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import recipe  # noqa: E402

COMPOSE_FILES = [REPO_ROOT / "docker" / "docker-compose.yml",
                 REPO_ROOT / "src" / "eval" / "docker-compose.yml"]
DOCKERFILES = sorted((REPO_ROOT / "docker").glob("Dockerfile.*")) + \
    [REPO_ROOT / "src" / "eval" / "Dockerfile"]


def _mounts(service):
    """The (source, target) pairs of a service's string-form volumes.

    Only string form is handled: this repo uses no long-syntax volumes, and a
    dict here would mean the compose file changed shape, which the caller
    should notice rather than silently skip.
    """
    out = []
    for entry in service.get("volumes") or []:
        assert isinstance(entry, str), (
            f"long-syntax volume in {entry!r}; _mounts() would skip it and the "
            f"contract would go unverified")
        parts = entry.split(":")
        assert len(parts) >= 2, f"unparsable volume {entry!r}"
        out.append((parts[0], parts[1]))
    return out


def test_data_is_mounted_wherever_the_loader_is():
    """Mount the loader at <root>/src/recipe => the data must sit at <root>/recipes.

    The root is whatever the loader's own parents[2] resolves to inside the
    container, so the check derives it from the loader's mount TARGET rather
    than hardcoding /app: renaming the mount point would otherwise leave this
    green while every run broke.
    """
    checked = 0
    for path in COMPOSE_FILES:
        doc = yaml.safe_load(path.read_text())
        rel = path.relative_to(REPO_ROOT)
        for name, svc in (doc.get("services") or {}).items():
            mounts = _mounts(svc or {})
            targets = {target for _src, target in mounts}
            for _src, target in mounts:
                if not target.endswith("/src/recipe"):
                    continue                       # this service runs no repo module
                root = target[: -len("/src/recipe")]
                assert f"{root}/recipes" in targets, (
                    f"{rel}: service {name!r} mounts the loader at {target} but not "
                    f"the data at {root}/recipes - recipe.RECIPE_DIR resolves there "
                    f"from the loader's own location, so the run would import cleanly "
                    f"and then find no recipe")
                checked += 1
    # Two trainers + the eval harness. Fewer means a service stopped mounting the
    # loader, which is its own story and should not pass unnoticed.
    assert checked == 3, f"expected 3 loader mounts across the compose files, saw {checked}"


def test_loader_path_math_matches_the_mount_point():
    """The host side of the same invariant: RECIPE_DIR is <root>/recipes, and <root>
    is the repo root the tests run from - not the cwd, not the module's own dir."""
    assert recipe.REPO_ROOT == REPO_ROOT, recipe.REPO_ROOT
    assert recipe.RECIPE_DIR == REPO_ROOT / "recipes", recipe.RECIPE_DIR
    assert recipe.path_for("hey seeree").parent == recipe.RECIPE_DIR
    # The data is where the loader says it is, and the loader is not inside it.
    assert recipe.RECIPE_DIR.is_dir(), recipe.RECIPE_DIR
    assert (recipe.RECIPE_DIR / "hey_seeree.yaml").is_file()
    assert not (Path(recipe.__file__).parent / "hey_seeree.yaml").exists(), (
        "a YAML beside the loader: it would be invisible to RECIPE_DIR")


def test_every_copy_source_still_exists():
    """A COPY of a deleted path fails at build time with a confusing message.

    The restructure moved a package out from under src/ and into the root, and
    `COPY src/wordlists/` would have kept building against a directory that no
    longer existed only in the sense that it would have FAILED - loudly, which
    is why this test is about catching it earlier and cheaper than a build.
    """
    for path in DOCKERFILES:
        rel = path.relative_to(REPO_ROOT)
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line.startswith("COPY ") and not line.startswith("ADD "):
                continue
            tokens = [t for t in line.split()[1:] if not t.startswith("--")]
            assert tokens, f"{rel}: {line!r} names no source"
            for src in tokens[:-1]:               # the last token is the destination
                if "$" in src or "*" in src:
                    continue                      # build-arg or glob: not checkable here
                assert (REPO_ROOT / src).exists(), (
                    f"{rel}: {line!r} - {src} does not exist in the repo")


def test_trainer_images_copy_what_they_import():
    """An image that imports `train.corpus` at build time must COPY the loader.

    That is the exact defect: train.corpus.negatives imports recipe at module
    level, the build check imports train.corpus.negatives, and only src/train/
    and src/tts-service/ were copied.
    """
    for path in DOCKERFILES:
        text = path.read_text()
        rel = path.relative_to(REPO_ROOT)
        if "import train.corpus" not in text:
            continue
        assert re.search(r"^COPY\s+src/recipe/", text, re.MULTILINE), (
            f"{rel}: the build imports train.corpus (which imports recipe) but never "
            f"copies src/recipe/ - the check would die with "
            f"ModuleNotFoundError: No module named 'recipe'")


def main():
    import _runner
    _runner.run(sys.modules[__name__])


if __name__ == "__main__":
    main()

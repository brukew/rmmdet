"""Central path resolver for the actreg repo.

Single source of truth for filesystem locations, backed by ``config.yaml`` which
sits next to this file. The repo root is auto-detected as the directory containing
this module, so the repo can be cloned anywhere without editing paths.

Resolution rules (see ``config.yaml`` for the keys):
    1. If the matching UPPERCASE environment variable is set, it wins.
       e.g. ``SAILS_DATA_ROOT=/x`` overrides the ``sails_data_root`` key.
    2. Otherwise the value from ``config.yaml`` is used.
    3. Values that do not start with ``/`` are treated as RELATIVE to the repo
       root and resolved to absolute paths.

Usage
-----
    from paths import PATHS
    PATHS.repo_root                 # -> Path to the repo
    PATHS.rmm_features              # -> Path to shared V-JEPA TAL features
    PATHS["opentad_exps"]           # dict-style access also works
    PATHS.get("vjepa_4class_ckpt")  # None if key missing

Shell scripts can read a single value via:
    python -c "from paths import PATHS; print(PATHS.rmm_features)"
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "paths.py requires PyYAML. Install it (it is included in the pinned "
        "envs under envs/), e.g. `pip install pyyaml`."
    ) from exc

#: Directory containing this file == repo root.
REPO_ROOT = Path(__file__).resolve().parent
CONFIG_FILE = REPO_ROOT / "config.yaml"


class _Paths:
    """Lazily-resolved, attribute- and dict-accessible path registry."""

    def __init__(self, config_file: Path = CONFIG_FILE) -> None:
        self._repo_root = REPO_ROOT
        if not config_file.exists():
            raise FileNotFoundError(f"config.yaml not found at {config_file}")
        with open(config_file, "r") as fh:
            self._raw: dict[str, Any] = yaml.safe_load(fh) or {}

    @property
    def repo_root(self) -> Path:
        return self._repo_root

    def _resolve(self, key: str) -> Path:
        """Resolve a single key applying env override + relative-to-root rules."""
        env_val = os.environ.get(key.upper())
        value = env_val if env_val is not None else self._raw.get(key)
        if value is None:
            raise KeyError(
                f"Unknown path key '{key}'. Known keys: "
                f"{sorted(self._raw) + ['repo_root']}"
            )
        path = Path(str(value))
        if not path.is_absolute():
            path = self._repo_root / path
        return path

    def get(self, key: str, default: Any = None) -> Path | None:
        """Like dict.get; returns ``default`` instead of raising on missing key."""
        if key == "repo_root":
            return self._repo_root
        if os.environ.get(key.upper()) is None and key not in self._raw:
            return default
        return self._resolve(key)

    def __getattr__(self, key: str) -> Path:
        # Only called for attributes not found normally (e.g. config keys).
        if key.startswith("_"):
            raise AttributeError(key)
        return self._resolve(key)

    def __getitem__(self, key: str) -> Path:
        if key == "repo_root":
            return self._repo_root
        return self._resolve(key)

    def keys(self) -> list[str]:
        return ["repo_root", *sorted(self._raw)]

    def as_dict(self) -> dict[str, str]:
        """All keys resolved to absolute path strings (handy for logging)."""
        out = {"repo_root": str(self._repo_root)}
        for k in self._raw:
            out[k] = str(self._resolve(k))
        return out


#: Importable singleton.
PATHS = _Paths()


def _shell_quote(value: str) -> str:
    """Single-quote a value for safe use in a POSIX shell `eval`."""
    return "'" + value.replace("'", "'\\''") + "'"


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Resolve actreg paths from config.yaml (single source of truth)."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--export",
        action="store_true",
        help="Emit `export KEY=VALUE` lines (UPPERCASE keys) for sourcing in shell "
        "scripts, e.g. `eval \"$(python paths.py --export)\"`.",
    )
    group.add_argument(
        "--get",
        metavar="KEY",
        help="Print a single resolved path value (no trailing newline issues; "
        "handy for `X=$(python paths.py --get rmm_features)`).",
    )
    args = parser.parse_args()

    if args.get:
        # Print one resolved value; raises KeyError with the known-keys list if bad.
        print(PATHS[args.get])
    elif args.export:
        # `repo_root` first so REPO_ROOT is always available to scripts.
        resolved = PATHS.as_dict()
        for key, val in resolved.items():
            print(f"export {key.upper()}={_shell_quote(val)}")
    else:
        # `python paths.py` prints the fully-resolved configuration as JSON.
        print(json.dumps(PATHS.as_dict(), indent=2))

#!/usr/bin/env python3
"""Run any repo script on Windows after patching one pickling limitation.

torch DataLoader worker processes on Windows are created via "spawn", which
pickles the Dataset object to send to each worker. This repo's dataset
classes store some attributes as MappingProxyType (a read-only dict view),
which Python's pickle module cannot serialize by default on any platform.
Registering a reducer here only teaches pickle how to serialize that one
built-in type; it does not change any model, data, or training logic.

Usage (replaces "python scripts\\X.py --arg value" with):
    python scripts\\run_with_windows_patch.py scripts\\X.py --arg value

Every argument after the target script path is passed through unchanged.
"""

from __future__ import annotations

import copyreg
import runpy
import sys
from pathlib import Path
from types import MappingProxyType

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    # Must be importable by qualified name in both this process and any
    # spawned DataLoader worker process, so it cannot live in __main__
    # (runpy.run_path(..., run_name="__main__") replaces sys.modules["__main__"]
    # for the target script, which would make a function defined here
    # unreachable under that name once the target script starts).
    sys.path.insert(0, str(SCRIPTS_DIR))

from _windows_pickle_patch import reduce_mappingproxy


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python scripts/run_with_windows_patch.py <target_script.py> [args...]"
        )

    copyreg.pickle(MappingProxyType, reduce_mappingproxy)

    target_script = sys.argv[1]
    # Make the target script see itself as sys.argv[0], as if invoked directly.
    sys.argv = sys.argv[1:]
    runpy.run_path(target_script, run_name="__main__")


if __name__ == "__main__":
    main()

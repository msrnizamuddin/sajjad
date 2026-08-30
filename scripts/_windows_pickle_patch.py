"""Reusable pickle reducer for MappingProxyType, importable by name.

This must live in its own real module (not inside a script's __main__) so
that both the parent process and any spawned torch DataLoader worker
processes on Windows can import it by qualified name when unpickling.
"""

from __future__ import annotations

from types import MappingProxyType


def rebuild_mappingproxy(mapping):
    return MappingProxyType(mapping)


def reduce_mappingproxy(obj):
    return rebuild_mappingproxy, (dict(obj),)

"""``bucket units``: read-only inventory of trace units for external archive/mirror tools.

One entry per (slug, trace_id) found in the per-trace stores (``traces/v1``,
``objects/traces/v1``, ``contexts/v1``, the ``objects/raw/v1/sources`` link). Each carries
``last_event_at`` (event time from the trace record's ``timestamp_end``, falling back to
``timestamp_start``; never file mtime), every local file of the unit (bucket-relative,
with size) and the shared raw blobs the unit's link names. A blob may be owned by several
units; ``raw_blobs`` lists only blob paths, ownership across units is for the caller to
join. Nothing is written.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import paths
from .bucket_archive_ops import _local_unit_files, _unit_stores

UNITS_SCHEMA = "opentraces.bucket.units.v1"
_BLOB_PREFIX = "objects/raw/v1/blobs/"


def _scan_ids(bucket: Path) -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for store in ("traces/v1", "objects/traces/v1", "contexts/v1"):
        root = bucket / store
        if not root.is_dir():
            continue
        for slug_dir in root.iterdir():
            if slug_dir.is_dir():
                found.update((slug_dir.name, t.name) for t in slug_dir.iterdir() if t.is_dir())
    links = bucket / "objects/raw/v1/sources"
    if links.is_dir():
        for slug_dir in links.iterdir():
            if slug_dir.is_dir():
                found.update((slug_dir.name, p.stem) for p in slug_dir.glob("*.json"))
    return found


def _read_json(path: Path) -> dict[str, Any]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def _last_event_at(bucket: Path, slug: str, trace_id: str) -> str | None:
    doc = _read_json(bucket / "traces/v1" / slug / trace_id / "trace.json")
    for key in ("timestamp_end", "timestamp_start"):
        value = doc.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _raw_blobs(bucket: Path, slug: str, trace_id: str) -> list[str]:
    link = _read_json(bucket / _unit_stores(slug, trace_id)[1])
    blob = link.get("blob_path")
    if isinstance(blob, str) and blob.startswith(_BLOB_PREFIX) and ".." not in blob.split("/"):
        return [blob]
    return []


def bucket_units(*, slug: str | None = None, before: str | None = None) -> dict[str, Any]:
    """Units in the local bucket, optionally one slug and/or last_event_at < ``before`` (ISO).

    With ``before``, a unit with no event time is left out (never guessed old).
    """

    bucket = paths.bucket_dir()
    units: list[dict[str, Any]] = []
    for s, t in sorted(_scan_ids(bucket)):
        if slug is not None and s != slug:
            continue
        last = _last_event_at(bucket, s, t)
        if before is not None and (last is None or last >= before):
            continue
        files = []
        for rel in sorted(_local_unit_files(s, t)):
            try:
                size = (bucket / rel).stat().st_size
            except OSError:
                continue
            files.append({"path": rel, "size": size})
        units.append({
            "slug": s,
            "trace_id": t,
            "last_event_at": last,
            "files": files,
            "raw_blobs": _raw_blobs(bucket, s, t),
        })
    return {"units": units, "count": len(units)}

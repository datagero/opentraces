"""Archive ledger, read-only archive fallback and write guard.

A trace unit (``project_slug`` + ``trace_id``) whose files were verified on an
external archive (a NAS mirror of the bucket's relative paths) can be removed
from the local bucket. This module is the one place that knows about it:

* ``archive_bucket_dir()`` / ``archive_root()`` -- where the archive lives and
  whether it is usable right now (smbfs mount carrying the ``.agentic-nas``
  marker). Unconfigured installs have no archive and behave exactly as before.
* the append-only ledger at ``bucket/archive/ledger.jsonl`` -- the record of
  which units are archived (``archive`` lines) and which were brought back
  (``rehydrate`` tombstone lines).
* ``resolve_read`` / ``glob_with_archive`` -- local first, then the archive, but
  ONLY for files the ledger lists. The archive is never listed or probed for
  anything else, so a stray copy on the NAS can never surface as a trace.
* ``assert_not_archive_path`` -- the write guard: the archive is read-only.
* ``skip_if_archived`` / ``allow_rehydrate`` -- the guard the writers that could
  bring an archived unit back consult before they touch disk.
"""

from __future__ import annotations

import contextlib
import contextvars
import fcntl
import json
import os
import re
import subprocess
import time
from collections.abc import Callable, Iterator
from pathlib import Path, PurePosixPath
from typing import Any

from . import paths
from ._time import utc_now_str

ARCHIVE_DIR_ENV = "OT_ARCHIVE_DIR"
NAS_ROOT_ENV = "AGENTIC_NAS_ROOT"
NAS_SHARE = "opentraces"
NAS_MARKER = ".agentic-nas"
REHYDRATE_ENV = "OT_ARCHIVE_REHYDRATE"
LEDGER_SCHEMA = "opentraces.bucket.archive_ledger.v1"
_VALID_TTL_SECONDS = 30.0
_MOUNT_RE = re.compile(r"^(?P<src>.+?) on (?P<mnt>.+?) \((?P<fs>[^,)]+)")


class ArchiveWriteError(RuntimeError):
    """A write was aimed at the read-only archive."""


class ArchiveUnavailableError(RuntimeError):
    """The archive root is not configured or not a verified NAS mount."""


# --- archive root ----------------------------------------------------------

_root_check: Callable[[Path], bool] | None = None
_valid_cache: dict[str, tuple[float, bool]] = {}


def set_root_check(check: Callable[[Path], bool] | None) -> None:
    """Replace the mount check (tests inject one for a temp dir)."""

    global _root_check
    _root_check = check
    _valid_cache.clear()


def reset_caches() -> None:
    _valid_cache.clear()
    _ledger_cache.clear()


def archive_bucket_dir() -> Path | None:
    """Configured archive root (unchecked): ``OT_ARCHIVE_DIR`` or the NAS share."""

    explicit = os.environ.get(ARCHIVE_DIR_ENV)
    if explicit:
        return Path(explicit).expanduser()
    nas_root = os.environ.get(NAS_ROOT_ENV)
    if nas_root:
        return Path(nas_root).expanduser() / NAS_SHARE / "bucket"
    return None


def _mount_table() -> dict[str, str]:
    try:
        text = subprocess.run(
            ["/sbin/mount"], capture_output=True, text=True, timeout=5, check=False
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    table: dict[str, str] = {}
    for line in text.splitlines():
        match = _MOUNT_RE.match(line)
        if match:
            table[match.group("mnt")] = match.group("fs").strip()
    return table


def _marker_present(mount_point: str) -> bool:
    try:
        done = subprocess.run(
            ["/bin/test", "-e", os.path.join(mount_point, NAS_MARKER)], timeout=3, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


def smb_root_ok(path: Path) -> bool:
    """True iff ``path`` sits on a live smbfs mount that carries the NAS marker.

    Mirrors agentic-storage's probe: an unmounted share is a plain folder on the
    boot disk, so the filesystem type and the marker are both required.
    """

    target = os.path.abspath(path)
    best = ""
    fs = ""
    for mount_point, fs_type in _mount_table().items():
        prefix = mount_point.rstrip("/")
        if (target == mount_point or target.startswith(prefix + "/")) and len(mount_point) > len(best):
            best, fs = mount_point, fs_type
    if not best or fs != "smbfs":
        return False
    return _marker_present(best)


def archive_root() -> Path | None:
    """The archive root if configured AND currently usable, else ``None``."""

    root = archive_bucket_dir()
    if root is None:
        return None
    key = str(root)
    now = time.monotonic()
    cached = _valid_cache.get(key)
    if cached is not None and now - cached[0] < _VALID_TTL_SECONDS:
        return root if cached[1] else None
    check = _root_check or smb_root_ok
    try:
        ok = bool(check(root))
    except Exception:  # noqa: BLE001 - a broken check means unusable, never crash a read
        ok = False
    _valid_cache[key] = (now, ok)
    return root if ok else None


# --- write guard -----------------------------------------------------------


def assert_not_archive_path(path: Path | str) -> None:
    """Raise if ``path`` is at or under the configured archive root."""

    root = archive_bucket_dir()
    if root is None:
        return
    target = os.path.abspath(path)
    base = os.path.abspath(root)
    if target == base or target.startswith(base.rstrip(os.sep) + os.sep):
        raise ArchiveWriteError(f"refusing to write under the read-only archive: {target}")


# --- ledger ----------------------------------------------------------------

_ledger_cache: dict[str, tuple[tuple[int, int], dict[tuple[str, str], dict[str, Any]]]] = {}


def ledger_path() -> Path:
    return paths.bucket_dir() / "archive" / "ledger.jsonl"


def iter_ledger_lines() -> Iterator[dict[str, Any]]:
    """Every parseable ledger line, oldest first. Corrupt lines are skipped."""

    try:
        handle = ledger_path().open("r", encoding="utf-8")
    except OSError:
        return
    with handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            try:
                line = json.loads(raw)
            except ValueError:
                continue
            if isinstance(line, dict):
                yield line


def archived_units() -> dict[tuple[str, str], dict[str, Any]]:
    """Live archived units: ``archive`` lines not followed by a ``rehydrate``."""

    path = ledger_path()
    try:
        st = path.stat()
    except OSError:
        return {}
    stamp = (st.st_mtime_ns, st.st_size)
    cached = _ledger_cache.get(str(path))
    if cached is not None and cached[0] == stamp:
        return cached[1]
    live: dict[tuple[str, str], dict[str, Any]] = {}
    for line in iter_ledger_lines():
        key = (str(line.get("slug") or ""), str(line.get("trace_id") or ""))
        if not key[0] or not key[1]:
            continue
        if line.get("op") == "archive":
            live[key] = line
        elif line.get("op") == "rehydrate":
            live.pop(key, None)
    _ledger_cache[str(path)] = (stamp, live)
    return live


def is_archived(project_slug: str, trace_id: str) -> bool:
    return (project_slug, trace_id) in archived_units()


def archived_file_index() -> dict[str, tuple[str, str]]:
    """``bucket-relative posix path`` -> (slug, trace_id) for live archived files."""

    index: dict[str, tuple[str, str]] = {}
    for key, entry in archived_units().items():
        for item in entry.get("files") or []:
            rel = item.get("path") if isinstance(item, dict) else None
            if isinstance(rel, str) and rel:
                index[rel] = key
    return index


def append_ledger(entry: dict[str, Any]) -> None:
    """Append one line atomically (single O_APPEND write, flock, fsync)."""

    path = ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    line = {"schema_version": LEDGER_SCHEMA, **entry}
    data = (json.dumps(line, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    _ledger_cache.pop(str(path), None)


def archive_entry(
    project_slug: str, trace_id: str, files: list[dict[str, Any]], proof_id: str
) -> dict[str, Any]:
    return {
        "op": "archive",
        "slug": project_slug,
        "trace_id": trace_id,
        "files": files,
        "archived_at": utc_now_str(),
        "proof_id": proof_id,
    }


def rehydrate_entry(project_slug: str, trace_id: str) -> dict[str, Any]:
    return {"op": "rehydrate", "slug": project_slug, "trace_id": trace_id, "at": utc_now_str()}


# --- read fallback ---------------------------------------------------------


def bucket_relative(path: Path | str) -> str | None:
    """``path`` relative to the local bucket root as posix, or ``None``."""

    base = os.path.abspath(paths.bucket_dir())
    target = os.path.abspath(path)
    if not target.startswith(base.rstrip(os.sep) + os.sep):
        return None
    return PurePosixPath(*Path(target[len(base) :].lstrip(os.sep)).parts).as_posix()


def resolve_read(path: Path) -> Path:
    """Local path if present, else the archive copy of a ledger-listed file.

    Always returns something safe to open: when neither exists the original
    local path comes back so callers keep their normal missing-file handling.
    """

    path = Path(path)
    if path.exists():
        return path
    if not archived_units():
        return path
    rel = bucket_relative(path)
    if rel is None or rel not in archived_file_index():
        return path
    root = archive_root()
    if root is None:
        return path
    candidate = root / rel
    return candidate if candidate.exists() else path


def exists_with_archive(path: Path) -> bool:
    return resolve_read(path).exists()


def glob_with_archive(root: Path, pattern: str) -> list[Path]:
    """``root.glob(pattern)`` plus ledger-listed archived files matching it.

    Returned paths are local-style (under ``root``); pass them to
    ``resolve_read`` to open. Only fixed-depth patterns (no ``**``) are
    supported. With an empty ledger this is exactly ``sorted(root.glob(..))``.
    """

    found = set(root.glob(pattern)) if root.exists() else set()
    if archived_units():
        depth = len(PurePosixPath(pattern).parts)
        base = bucket_relative(root)
        for rel in archived_file_index():
            if base is not None:
                if base and not rel.startswith(base + "/"):
                    continue
                tail = rel[len(base) + 1 :] if base else rel
            else:
                continue
            pure = PurePosixPath(tail)
            if len(pure.parts) == depth and pure.match(pattern):
                found.add(root / tail)
    return sorted(found)


# --- resurrection guard ----------------------------------------------------

_rehydrate_flag: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "ot_archive_rehydrate", default=False
)


@contextlib.contextmanager
def allow_rehydrate() -> Iterator[None]:
    """Let writers recreate archived units (``--rehydrate``)."""

    token = _rehydrate_flag.set(True)
    try:
        yield
    finally:
        _rehydrate_flag.reset(token)


def rehydrate_allowed() -> bool:
    return _rehydrate_flag.get() or os.environ.get(REHYDRATE_ENV) == "1"


def skip_if_archived(project_slug: str, trace_id: str) -> bool:
    """True when a writer must leave this unit alone (archived, not rehydrating)."""

    return is_archived(project_slug, trace_id) and not rehydrate_allowed()


def has_live_archive_entries() -> bool:
    return bool(archived_units())

"""Lossless packing of superseded TraceRecord objects.

Every re-capture of a growing session writes a new full TraceRecord object
``objects/traces/v1/<project>/<trace>/<digest>.json`` and repoints
``current.json`` at it. The earlier objects stay on disk. They are not
duplicates — each one is the record as an earlier parser version derived it —
so they are kept, but consecutive versions share almost all of their bytes.

``pack_superseded_trace_records`` moves every superseded object of a trace into
one per-trace archive, ``superseded.otpack``, next to ``current.json``.
The current object stays a loose ``.json`` file, so the hot read path
(``current.json`` -> object) is unchanged.

Pack format: a header naming the codec, then a compressed POSIX pax tar (see
``bucket_pack_codecs``). Members are the original ``<digest>.json`` files,
byte-exact, in mtime order so consecutive versions sit next to each other
inside the compression window. Each member carries pax headers
``OPENTRACES.sha256`` (sha256 of the member bytes) and ``OPENTRACES.mtime_ns``.
Codecs are an optional extra: without one, ``bucket pack`` refuses with an
install hint, and nothing on the current-record read path needs a codec.

Safety:

- The new pack is written to a tempfile, fsynced, and re-read with every
  member's bytes hashing to its stored sha256 and the member set equal to the
  merged set BEFORE it replaces the old pack (whose members may exist nowhere
  else). Only then are loose files removed.
- A version is identified by name AND bytes: the same record hash re-written
  with a different envelope (``written_at``) is packed alongside, not refused.
- Re-packing merges the existing pack with newly superseded loose objects, so
  the verb is idempotent. A loose file that is already in the pack with the
  same bytes (a crash after rename, before unlink) is simply removed.
- Interrupted writes leave ``.superseded.otpack.*.tmp`` files that every
  reader ignores and the next pack run removes once they are an hour old.
- The current object and objects younger than ``min_age_seconds`` (default one
  hour) are never packed. ``current.json`` is re-read right before each unlink.

Pack on write (``pack_on_write``): ``write_trace_record`` calls it after every
repoint. Once a trace has ``PACK_ON_WRITE_MIN_LOOSE`` superseded loose objects
(or ``PACK_ON_WRITE_MIN_BYTES`` of them), they go into a NEW small segment
``superseded.<stamp>.otpack`` — the main pack is never rewritten on the capture
path, so the cost is bounded by the segment, not by the trace's history.
``bucket pack`` folds segments back into ``superseded.otpack``. Every reader
treats the main pack and all segments as one member set. No lock is needed:
writers only ever add new files, a segment or loose file is removed only after
a verified pack holding its members is in place, and a member present in two
packs is the same (name, sha256) version.
"""

from __future__ import annotations

import hashlib
import heapq
import io
import json
import os
import re
import tarfile
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import paths
from .bucket_archive import glob_with_archive, resolve_read
from .bucket_layout import trace_records_root
from .bucket_models import TRACE_RECORD_POINTER_SCHEMA
from .bucket_pack_codecs import (
    CODEC_ERRORS,
    PACK_SCHEMA,
    CodecUnavailable,
    open_reader,
    open_writer,
    window_log,
    write_codec,
)

PACK_NAME = "superseded.otpack"
SEGMENT_GLOB = "superseded.*.otpack"
_SEGMENT_NAME = re.compile(r"^superseded\.[0-9]{20}-[0-9a-f]{8}\.otpack$")
PACK_ON_WRITE_MIN_LOOSE = 8
PACK_ON_WRITE_MIN_BYTES = 256 * 1024 * 1024
PACK_ON_WRITE_ENV = "OPENTRACES_PACK_ON_WRITE"
DEFAULT_MIN_AGE_SECONDS = 60 * 60
_TMP_PREFIX = f".{PACK_NAME}."
_TMP_MAX_AGE_SECONDS = 60 * 60
_OBJECT_NAME = re.compile(r"^[0-9a-f]{64}\.json$")
# A packed version whose name collides with a newer loose copy of the same
# record hash (same record, different envelope bytes, e.g. ``written_at``) is
# restored by ``unpack`` as ``<digest>.<sha12>.json``; ``pack`` folds it back.
_VARIANT_NAME = re.compile(r"^([0-9a-f]{64})\.([0-9a-f]{12})\.json$")
_SHA_KEY = "OPENTRACES.sha256"
_MTIME_KEY = "OPENTRACES.mtime_ns"


class PackError(RuntimeError):
    """A pack could not be written or read back losslessly."""


PACK_ERRORS: tuple[type[BaseException], ...] = (
    OSError,
    ValueError,
    EOFError,
    PackError,
    CodecUnavailable,
    tarfile.TarError,
    *CODEC_ERRORS,
)


@dataclass(frozen=True)
class PackMember:
    name: str
    data: bytes
    mtime_ns: int
    mode: int
    sha256: str


def pack_path(trace_dir: Path) -> Path:
    return trace_dir / PACK_NAME


def is_pack_name(file_name: str) -> bool:
    return file_name == PACK_NAME or bool(_SEGMENT_NAME.match(file_name))


def segment_paths(trace_dir: Path) -> list[Path]:
    """Pack-on-write segments of one trace, oldest first (names sort by time)."""

    if not trace_dir.is_dir():
        return []
    return sorted(p for p in trace_dir.glob(SEGMENT_GLOB) if _SEGMENT_NAME.match(p.name) and p.is_file())


def trace_packs(trace_dir: Path) -> list[Path]:
    """Main pack (if any) then every segment: together, the packed member set."""

    main = pack_path(trace_dir)
    return ([main] if main.is_file() else []) + segment_paths(trace_dir)


def _new_segment_path(trace_dir: Path) -> Path:
    return trace_dir / f"superseded.{time.time_ns():020d}-{os.urandom(4).hex()}.otpack"


def _unique(members: Iterator[PackMember]) -> Iterator[PackMember]:
    seen: set[tuple[str, str]] = set()
    for member in members:
        key = (member.name, member.sha256)
        if key not in seen:
            seen.add(key)
            yield member


def _iter_all_members(packs: list[Path]) -> Iterator[PackMember]:
    """Members of several mtime-ordered packs, merged in mtime order, deduped."""

    streams = [iter_pack_members(p) for p in packs]
    if len(streams) == 1:
        return _unique(streams[0])
    return _unique(heapq.merge(*streams, key=lambda m: (m.mtime_ns, m.name)))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _current_object_name(trace_dir: Path) -> str | None:
    try:
        raw = json.loads((trace_dir / "current.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if raw.get("schema_version") != TRACE_RECORD_POINTER_SCHEMA:
        return None
    object_path = raw.get("object_path")
    if not isinstance(object_path, str) or not object_path:
        return None
    return object_path.rsplit("/", 1)[-1]


def iter_pack_members(path: Path) -> Iterator[PackMember]:
    """Stream the members of a pack in stored (mtime) order."""

    with open_reader(path) as raw, tarfile.open(fileobj=raw, mode="r|") as tar:
        for info in tar:
            if not info.isfile():
                continue
            handle = tar.extractfile(info)
            data = handle.read() if handle is not None else b""
            sha = info.pax_headers.get(_SHA_KEY) or _sha256(data)
            mtime_ns = int(info.pax_headers.get(_MTIME_KEY) or int(info.mtime) * 1_000_000_000)
            yield PackMember(info.name, data, mtime_ns, info.mode, sha)


def list_pack(path: Path) -> set[tuple[str, str]]:
    """Return ``{(member name, stored sha256)}`` for a pack.

    A version's identity is its name AND its bytes: the same record hash can be
    stored twice when its envelope bytes differ.
    """

    return {(m.name, m.sha256) for m in iter_pack_members(path)}


def read_packed_object(object_path: Path) -> bytes | None:
    """Return the bytes of ``object_path`` from its trace's pack, if packed."""

    trace_dir = object_path.parent
    candidates = [pack_path(trace_dir), *(
        p for p in glob_with_archive(trace_dir, SEGMENT_GLOB) if _SEGMENT_NAME.match(p.name)
    )]
    found: PackMember | None = None
    for candidate in candidates:
        pack = resolve_read(candidate)
        if not pack.is_file():
            continue
        try:
            for member in iter_pack_members(pack):
                if member.name == object_path.name and (found is None or member.mtime_ns >= found.mtime_ns):
                    found = member
        except CodecUnavailable:
            raise
        except PACK_ERRORS:
            continue
    return found.data if found is not None else None


def iter_trace_record_versions(trace_dir: Path) -> Iterator[tuple[str, bytes, int]]:
    """Yield ``(name, bytes, mtime_ns)`` for every stored version, oldest first.

    Includes packed and loose objects; a name present in both is yielded once.
    """

    items: dict[tuple[str, str], tuple[bytes, int]] = {}
    for pack in trace_packs(trace_dir):
        for member in iter_pack_members(pack):
            items[(member.name, member.sha256)] = (member.data, member.mtime_ns)
    for path in trace_dir.iterdir():
        name = _member_name(path.name)
        if name is not None and path.is_file():
            data = path.read_bytes()
            items[(name, _sha256(data))] = (data, path.stat().st_mtime_ns)
    for (name, _sha), (data, mtime_ns) in sorted(items.items(), key=lambda kv: (kv[1][1], kv[0])):
        yield name, data, mtime_ns


def _member_name(file_name: str) -> str | None:
    """Pack member name for a loose object or restored variant file, else None."""

    if _OBJECT_NAME.match(file_name):
        return file_name
    variant = _VARIANT_NAME.match(file_name)
    return f"{variant.group(1)}.json" if variant else None


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _write_pack(
    target: Path,
    members: Iterator[PackMember],
    wlog: int,
    codec: str | None = None,
    expected: set[tuple[str, str]] | None = None,
) -> set[tuple[str, str]]:
    """Write ``members`` to ``target`` atomically; return ``{(name, sha256)}``.

    The tempfile is re-read and every member's bytes are hashed BEFORE it
    replaces ``target``: on a re-pack the old pack holds the only copy of its
    members, so a corrupt new pack must never overwrite it. ``expected`` (the
    merged member set) must match what was written and read back.
    """

    written: set[tuple[str, str]] = set()
    fd, tmp_name = tempfile.mkstemp(prefix=_TMP_PREFIX, suffix=".tmp", dir=str(target.parent))
    tmp = Path(tmp_name)
    codec = codec or write_codec()
    try:
        with os.fdopen(fd, "wb") as fh:
            with (
                open_writer(fh, codec, wlog) as zfh,
                tarfile.open(fileobj=zfh, mode="w|", format=tarfile.PAX_FORMAT) as tar,
            ):
                for member in members:
                    info = tarfile.TarInfo(member.name)
                    info.size = len(member.data)
                    info.mtime = member.mtime_ns // 1_000_000_000
                    info.mode = member.mode
                    info.pax_headers = {
                        _SHA_KEY: member.sha256,
                        _MTIME_KEY: str(member.mtime_ns),
                    }
                    tar.addfile(info, io.BytesIO(member.data))
                    written.add((member.name, member.sha256))
            fh.flush()
            os.fsync(fh.fileno())
        if expected is not None and written != expected:
            raise PackError(f"{target}: wrote a different member set than expected")
        _verify_pack(tmp, written)
        os.replace(tmp, target)
        _fsync_dir(target.parent)
    finally:
        if tmp.exists():
            tmp.unlink()
    return written


def _verify_pack(path: Path, expected: set[tuple[str, str]]) -> None:
    """Re-read ``path`` and check every member's bytes hash to ``expected``."""

    seen: set[tuple[str, str]] = set()
    for member in iter_pack_members(path):
        actual = _sha256(member.data)
        if actual != member.sha256:
            raise PackError(f"{path}: member {member.name} sha mismatch")
        seen.add((member.name, actual))
    if seen != expected:
        raise PackError(f"{path}: pack contents differ from what was written")


def _sweep_stale_tmp(trace_dir: Path, now: float) -> int:
    removed = 0
    for tmp in trace_dir.glob(f"{_TMP_PREFIX}*.tmp"):
        try:
            if now - tmp.stat().st_mtime >= _TMP_MAX_AGE_SECONDS:
                tmp.unlink()
                removed += 1
        except FileNotFoundError:
            continue
    return removed


def _loose_member(path: Path) -> PackMember:
    data = path.read_bytes()
    st = path.stat()
    name = _member_name(path.name) or path.name
    return PackMember(name, data, st.st_mtime_ns, st.st_mode & 0o7777, _sha256(data))


def pack_trace_dir(
    trace_dir: Path,
    *,
    min_age_seconds: float = DEFAULT_MIN_AGE_SECONDS,
    dry_run: bool = False,
    now: float | None = None,
) -> dict[str, Any]:
    """Pack the superseded objects of one trace directory."""

    now = time.time() if now is None else now
    out: dict[str, Any] = {
        "records_packable": 0,
        "records_packed": 0,
        "skipped_current": 0,
        "skipped_young": 0,
        "loose_bytes_packable": 0,
        "pack_bytes": 0,
        "tmp_removed": 0,
        "pack_written": False,
    }
    current = _current_object_name(trace_dir)
    if current is None:
        # No valid pointer: every object would look superseded. Leave it alone.
        out["skipped_no_pointer"] = 1
        return out
    candidates: list[Path] = []
    for path in sorted(trace_dir.iterdir()):
        if _member_name(path.name) is None or not path.is_file():
            continue
        if path.name == current:
            out["skipped_current"] += 1
            continue
        try:
            st = path.stat()
        except FileNotFoundError:
            continue
        if now - st.st_mtime < min_age_seconds:
            out["skipped_young"] += 1
            continue
        candidates.append(path)
        out["loose_bytes_packable"] += st.st_size
    segments = segment_paths(trace_dir)
    out["records_packable"] = len(candidates)
    out["segments_merged"] = 0
    if dry_run:
        out["segments_mergeable"] = len(segments)
        return out
    out["tmp_removed"] = _sweep_stale_tmp(trace_dir, now)
    if not candidates and not segments:
        return out

    pack = pack_path(trace_dir)
    main_existing = list_pack(pack) if pack.is_file() else set()
    segment_members: set[tuple[str, str]] = set()
    segment_max = 0
    for segment in segments:
        for member in iter_pack_members(segment):
            segment_members.add((member.name, member.sha256))
            segment_max = max(segment_max, len(member.data))
    existing = main_existing | segment_members
    loose: list[tuple[Path, PackMember]] = []
    for path in candidates:
        try:
            loose.append((path, _loose_member(path)))
        except FileNotFoundError:
            # Packed into a segment by a concurrent capture: nothing to do.
            continue
    new: list[PackMember] = []
    queued: set[tuple[str, str]] = set()
    for _path, member in loose:
        key = (member.name, member.sha256)
        # Same name, different bytes (same record re-written with a new
        # envelope) is a distinct version: it is packed alongside, not refused.
        if key not in existing and key not in queued:
            new.append(member)
            queued.add(key)
    if new or segments:
        new.sort(key=lambda m: (m.mtime_ns, m.name))
        sizes = [len(m.data) for m in new] + [segment_max]
        # The packs listed above, not a fresh listing: a segment added by a
        # concurrent capture now is not in ``existing`` and stays for next run.
        listed = ([pack] if main_existing or pack.is_file() else []) + segments
        stream: Iterator[PackMember] = _iter_all_members(listed) if listed else iter(())
        if new:
            stream = _unique(heapq.merge(stream, iter(new), key=lambda m: (m.mtime_ns, m.name)))
        written = _write_pack(
            pack, stream, window_log(max(sizes)), expected=existing | queued
        )
        out["pack_written"] = True
        for segment in segments:
            segment.unlink(missing_ok=True)
            out["segments_merged"] += 1
    else:
        written = existing
    for path, member in loose:
        if (member.name, member.sha256) not in written:
            raise PackError(f"{trace_dir}: {member.name} missing from pack after write")
        if path.name == member.name and _current_object_name(trace_dir) == member.name:
            # Repointed at this object while we were packing; keep it loose.
            continue
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        out["records_packed"] += 1
    _fsync_dir(trace_dir)
    out["pack_bytes"] = pack.stat().st_size if pack.is_file() else 0
    return out


def pack_on_write_enabled() -> bool:
    return os.environ.get(PACK_ON_WRITE_ENV, "1").strip().lower() not in {"0", "false", "no", "off"}


def pack_on_write(
    trace_dir: Path,
    *,
    min_loose: int = PACK_ON_WRITE_MIN_LOOSE,
    min_bytes: int = PACK_ON_WRITE_MIN_BYTES,
) -> dict[str, Any]:
    """Move the superseded loose objects of one trace into a NEW segment.

    Called on the capture path right after ``current.json`` is repointed. Does
    nothing until the trace has ``min_loose`` superseded objects or
    ``min_bytes`` of them, so a trace costs at most that much loose history.
    Never rewrites the main pack. Raises ``PACK_ERRORS`` (incl.
    ``CodecUnavailable``); the caller decides whether to swallow them.
    """

    out: dict[str, Any] = {"records_packed": 0, "segment_written": False, "segment_bytes": 0}
    current = _current_object_name(trace_dir)
    if current is None:
        return out
    candidates: list[Path] = []
    loose_bytes = 0
    for path in sorted(trace_dir.iterdir()):
        if path.name == current or _member_name(path.name) is None or not path.is_file():
            continue
        try:
            loose_bytes += path.stat().st_size
        except FileNotFoundError:
            continue
        candidates.append(path)
    if len(candidates) < min_loose and loose_bytes < min_bytes:
        return out
    loose: list[tuple[Path, PackMember]] = []
    for path in candidates:
        try:
            loose.append((path, _loose_member(path)))
        except FileNotFoundError:
            continue
    if not loose:
        return out
    members = sorted((m for _p, m in loose), key=lambda m: (m.mtime_ns, m.name))
    expected = {(m.name, m.sha256) for m in members}
    segment = _new_segment_path(trace_dir)
    written = _write_pack(
        segment, _unique(iter(members)), window_log(max(len(m.data) for m in members)), expected=expected
    )
    out["segment_written"] = True
    out["segment_bytes"] = segment.stat().st_size
    for path, member in loose:
        if (member.name, member.sha256) not in written:
            raise PackError(f"{trace_dir}: {member.name} missing from segment after write")
        if path.name == member.name and _current_object_name(trace_dir) == member.name:
            continue
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        out["records_packed"] += 1
    _fsync_dir(trace_dir)
    return out


def _trace_dirs(project_slug: str | None) -> list[Path]:
    root = trace_records_root()
    if not root.exists():
        return []
    pattern = f"{project_slug}/*" if project_slug else "*/*"
    return sorted(p for p in root.glob(pattern) if p.is_dir())


def _tree_bytes(root: Path) -> tuple[int, int]:
    """Return ``(object file count, total bytes)`` under ``root``."""

    count = 0
    total = 0
    if not root.exists():
        return 0, 0
    for path in root.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
            if _OBJECT_NAME.match(path.name):
                count += 1
    return count, total


def pack_superseded_trace_records(
    *,
    dry_run: bool = False,
    min_age_seconds: float = DEFAULT_MIN_AGE_SECONDS,
    project_slug: str | None = None,
) -> dict[str, Any]:
    """Pack superseded TraceRecord objects for every trace (or one project).

    Raises ``CodecUnavailable`` (unless ``dry_run``) when no codec is installed.
    """

    codec = None if dry_run else write_codec()
    root = trace_records_root()
    records_before, bytes_before = _tree_bytes(root)
    totals: dict[str, int] = dict.fromkeys(
        ("records_packable", "records_packed", "skipped_current", "skipped_young",
         "loose_bytes_packable", "pack_bytes", "tmp_removed", "pack_written",
         "segments_merged"),
        0,
    )
    errors: list[dict[str, str]] = []
    traces = _trace_dirs(project_slug)
    now = time.time()
    for trace_dir in traces:
        try:
            result = pack_trace_dir(
                trace_dir, min_age_seconds=min_age_seconds, dry_run=dry_run, now=now
            )
        except PACK_ERRORS as exc:
            errors.append({"path": _rel(trace_dir), "detail": str(exc)})
            continue
        for key, value in result.items():
            totals[key] = totals.get(key, 0) + int(value)
    records_after, bytes_after = (
        (records_before, bytes_before) if dry_run else _tree_bytes(root)
    )
    return {
        "schema_version": PACK_SCHEMA,
        "dry_run": dry_run,
        "codec": codec,
        "min_age_seconds": min_age_seconds,
        "traces_scanned": len(traces),
        "packs_written": totals.pop("pack_written", 0),
        **totals,
        "loose_records_before": records_before,
        "loose_records_after": records_after,
        "bytes_before": bytes_before,
        "bytes_after": bytes_after,
        "errors": errors,
        "ok": not errors,
    }


def unpack_trace_dir(trace_dir: Path, *, dry_run: bool = False) -> dict[str, int]:
    """Restore every packed object of one trace as a loose file, byte-exact."""

    out = {"records_restored": 0, "records_present": 0, "packs_removed": 0}
    packs = trace_packs(trace_dir)
    if not packs:
        return out
    members = list(_iter_all_members(packs))
    newest: dict[str, PackMember] = {}
    for member in members:
        if _sha256(member.data) != member.sha256:
            raise PackError(f"{trace_dir}: packed member {member.name} sha mismatch")
        if member.name not in newest or member.mtime_ns >= newest[member.name].mtime_ns:
            newest[member.name] = member
    for member in members:
        target = trace_dir / member.name
        canonical_taken = target.exists() and _sha256(target.read_bytes()) != member.sha256
        if newest[member.name] is not member or canonical_taken:
            target = trace_dir / f"{member.name[:-5]}.{member.sha256[:12]}.json"
        if target.exists():
            if _sha256(target.read_bytes()) != member.sha256:
                raise PackError(f"{target} differs from its packed copy")
            out["records_present"] += 1
            continue
        out["records_restored"] += 1
        if dry_run:
            continue
        fd, tmp_name = tempfile.mkstemp(prefix=f".{member.name}.", suffix=".tmp", dir=str(trace_dir))
        with os.fdopen(fd, "wb") as fh:
            fh.write(member.data)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp_name, member.mode)
        os.utime(tmp_name, ns=(member.mtime_ns, member.mtime_ns))
        os.replace(tmp_name, target)
    if not dry_run:
        _fsync_dir(trace_dir)
        for pack in packs:
            pack.unlink()
            out["packs_removed"] += 1
        _fsync_dir(trace_dir)
    return out


def unpack_superseded_trace_records(
    *,
    dry_run: bool = False,
    project_slug: str | None = None,
) -> dict[str, Any]:
    root = trace_records_root()
    records_before, bytes_before = _tree_bytes(root)
    totals = {"records_restored": 0, "records_present": 0, "packs_removed": 0}
    errors: list[dict[str, str]] = []
    for trace_dir in _trace_dirs(project_slug):
        try:
            result = unpack_trace_dir(trace_dir, dry_run=dry_run)
        except PACK_ERRORS as exc:
            errors.append({"path": _rel(trace_dir), "detail": str(exc)})
            continue
        for key, value in result.items():
            totals[key] += value
    records_after, bytes_after = (
        (records_before, bytes_before) if dry_run else _tree_bytes(root)
    )
    return {
        "schema_version": PACK_SCHEMA,
        "dry_run": dry_run,
        **totals,
        "loose_records_before": records_before,
        "loose_records_after": records_after,
        "bytes_before": bytes_before,
        "bytes_after": bytes_after,
        "errors": errors,
        "ok": not errors,
    }


def verify_trace_packs(
    packs: list[Path] | None = None,
) -> tuple[int, list[dict[str, str]], list[str]]:
    """Check pack integrity; return ``(packs checked, errors, packs skipped)``.

    A pack whose codec is not installed is skipped (reported, not an error):
    ``bucket verify`` must keep working on an install without the extra.

    For each pack: every member decodes, its bytes match the stored sha256,
    its envelope's ``record_hash`` matches its file name, and the trace's
    ``current.json`` still resolves to a loose object (the current record is
    never packed).
    """

    if packs is None:
        packs = all_packs()
    errors: list[dict[str, str]] = []
    skipped: list[str] = []
    for pack in packs:
        rel = _rel(pack)
        try:
            for member in iter_pack_members(pack):
                if _sha256(member.data) != member.sha256:
                    errors.append({"kind": "pack_member_sha", "path": f"{rel}:{member.name}", "detail": "bytes do not match stored sha256"})
                    continue
                try:
                    envelope = json.loads(member.data)
                except ValueError as exc:
                    errors.append({"kind": "pack_member_json", "path": f"{rel}:{member.name}", "detail": str(exc)})
                    continue
                digest = str(envelope.get("record_hash") or "").removeprefix("sha256:")
                if f"{digest}.json" != member.name:
                    errors.append({"kind": "pack_member_hash", "path": f"{rel}:{member.name}", "detail": f"record_hash {digest!r} does not match name"})
        except CodecUnavailable:
            skipped.append(rel)
            continue
        except PACK_ERRORS as exc:
            errors.append({"kind": "pack_unreadable", "path": rel, "detail": str(exc)})
            continue
        current = _current_object_name(pack.parent)
        if current is not None and not (pack.parent / current).is_file():
            errors.append({"kind": "pack_current_missing", "path": rel, "detail": f"current object {current} is not a loose file"})
    return len(packs) - len(skipped), errors, skipped


def all_packs() -> list[Path]:
    root = trace_records_root()
    if not root.exists():
        return []
    return sorted(
        [*root.glob(f"*/*/{PACK_NAME}"),
         *(p for p in root.glob(f"*/*/{SEGMENT_GLOB}") if _SEGMENT_NAME.match(p.name))]
    )


def _rel(path: Path) -> str:
    try:
        return path.relative_to(paths.bucket_dir()).as_posix()
    except ValueError:
        return str(path)

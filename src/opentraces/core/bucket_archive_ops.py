"""``bucket archive`` and ``bucket rehydrate``: move trace units to / from the archive.

The archive copy is made and verified by an external tool (agentic-storage); this
module only trusts it after re-hashing. ``archive`` takes a *proof* listing
verified units with a sha256 per file and, per unit, re-checks every file on the
archive root BEFORE it appends the ledger line and deletes the local files.
Nothing here writes to the archive.

Proof file (JSON)::

    {"schema_version": "agentic-storage.archive-proof.v1",
     "proof_id": "...",
     "units": [{"slug": "...", "trace_id": "...",
                "files": [{"path": "traces/v1/<slug>/<trace>/trace.json",
                           "sha256": "<hex>", "size": 123}]}]}

``path`` is relative to the bucket root and must live in that unit's own stores
(``traces/v1``, ``objects/traces/v1``, ``contexts/v1`` directories and the
``objects/raw/v1/sources`` link), so a proof can never name another unit's or a
shared store's files.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from . import paths
from ._bucket_io import _atomic_write_json
from .bucket_archive import (
    append_ledger,
    archive_entry,
    archive_root,
    archived_units,
    rehydrate_entry,
)
from .bucket_layout import _path_part, bucket_manifest_path

PROOF_SCHEMA = "agentic-storage.archive-proof.v1"
_CHUNK = 1024 * 1024


class ProofError(ValueError):
    """The proof file is unreadable or malformed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def load_proof(path: Path) -> dict[str, Any]:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProofError(f"cannot read proof {path}: {exc}") from exc
    if not isinstance(doc, dict) or doc.get("schema_version") != PROOF_SCHEMA:
        raise ProofError(f"proof schema must be {PROOF_SCHEMA!r}")
    if not isinstance(doc.get("proof_id"), str) or not doc["proof_id"]:
        raise ProofError("proof is missing proof_id")
    if not isinstance(doc.get("units"), list):
        raise ProofError("proof is missing units[]")
    return doc


def _unit_stores(slug: str, trace_id: str) -> tuple[list[str], str]:
    """(directory prefixes, link-file path) a unit's files may live in."""

    ps, pt = _path_part(slug), _path_part(trace_id)
    dirs = [
        f"traces/v1/{ps}/{pt}/",
        f"objects/traces/v1/{ps}/{pt}/",
        f"contexts/v1/{ps}/{pt}/",
    ]
    return dirs, f"objects/raw/v1/sources/{ps}/{pt}.json"


def _path_in_unit(rel: str, slug: str, trace_id: str) -> bool:
    if not rel or rel.startswith("/") or "\\" in rel or ".." in rel.split("/"):
        return False
    dirs, link = _unit_stores(slug, trace_id)
    return rel == link or any(rel.startswith(prefix) for prefix in dirs)


def _local_unit_files(slug: str, trace_id: str) -> set[str]:
    bucket = paths.bucket_dir()
    dirs, link = _unit_stores(slug, trace_id)
    found: set[str] = set()
    for prefix in dirs:
        base = bucket / prefix
        if base.is_dir():
            for item in base.rglob("*"):
                if item.is_file() or item.is_symlink():
                    found.add(item.relative_to(bucket).as_posix())
    if (bucket / link).is_file():
        found.add(link)
    return found


def _check_unit(
    unit: Any, root: Path, ledger_live: dict[tuple[str, str], dict[str, Any]]
) -> tuple[str | None, dict[str, Any] | None]:
    """Return (refusal reason, None) or (None, normalised unit)."""

    if not isinstance(unit, dict):
        return "malformed-unit", None
    slug, trace_id = unit.get("slug"), unit.get("trace_id")
    files = unit.get("files")
    if not isinstance(slug, str) or not slug or not isinstance(trace_id, str) or not trace_id:
        return "malformed-unit", None
    if not isinstance(files, list) or not files:
        return "no-files", None
    if (slug, trace_id) in ledger_live:
        return "already-archived", None
    listed: dict[str, dict[str, Any]] = {}
    for item in files:
        rel = item.get("path") if isinstance(item, dict) else None
        sha = item.get("sha256") if isinstance(item, dict) else None
        if not isinstance(rel, str) or not isinstance(sha, str) or len(sha) != 64:
            return "malformed-file-entry", None
        if not _path_in_unit(rel, slug, trace_id):
            return f"path-outside-unit:{rel}", None
        listed[rel] = {"path": rel, "sha256": sha.lower(), "size": item.get("size")}
    if _local_unit_files(slug, trace_id) - set(listed):
        return "unlisted-local-file", None
    bucket = paths.bucket_dir()
    for rel, item in sorted(listed.items()):
        remote = root / rel
        try:
            if not remote.is_file() or sha256_file(remote) != item["sha256"]:
                return f"archive-mismatch:{rel}", None
            local = bucket / rel
            if local.is_file() and sha256_file(local) != item["sha256"]:
                return f"local-changed-since-proof:{rel}", None
        except OSError as exc:
            return f"io-error:{rel}:{exc.__class__.__name__}", None
    return None, {"slug": slug, "trace_id": trace_id, "files": sorted(listed.values(), key=lambda f: f["path"])}


def _delete_unit_files(slug: str, trace_id: str, files: list[dict[str, Any]]) -> tuple[int, int]:
    bucket = paths.bucket_dir()
    count = freed = 0
    for item in files:
        target = bucket / item["path"]
        try:
            size = target.stat().st_size
            target.unlink()
        except FileNotFoundError:
            continue
        count += 1
        freed += size
    dirs, _ = _unit_stores(slug, trace_id)
    for prefix in dirs:
        base = bucket / prefix.rstrip("/")
        if not base.is_dir():
            continue
        for current, _subdirs, _names in os.walk(base, topdown=False):
            try:
                Path(current).rmdir()
            except OSError:
                pass
        try:
            base.parent.rmdir()
        except OSError:
            pass
    link_parent = (bucket / _unit_stores(slug, trace_id)[1]).parent
    try:
        link_parent.rmdir()
    except OSError:
        pass
    return count, freed


def _set_manifest_archived(units: list[tuple[str, str]], archived: bool) -> None:
    """Flip ``archived`` on the persisted manifest rows (digest-excluded field)."""

    path = bucket_manifest_path()
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not isinstance(doc, dict):
        return
    wanted = set(units)
    changed = False
    for row in doc.get("traces") or []:
        if not isinstance(row, dict) or (row.get("project_slug"), row.get("trace_id")) not in wanted:
            continue
        if archived and row.get("archived") is not True:
            row["archived"] = True
            changed = True
        elif not archived and "archived" in row:
            del row["archived"]
            changed = True
    if changed:
        _atomic_write_json(path, doc)


def archive_units(proof_path: Path, *, apply: bool) -> dict[str, Any]:
    """Dry-run or apply a proof. Per-unit refusal never aborts the other units."""

    proof = load_proof(proof_path)
    root = archive_root()
    result: dict[str, Any] = {
        "ok": True,
        "dry_run": not apply,
        "proof_id": proof["proof_id"],
        "archive_root": str(root) if root else None,
        "units_in_proof": len(proof["units"]),
        "archived": [],
        "skipped": [],
        "files_deleted": 0,
        "bytes_freed": 0,
    }
    if root is None:
        result["ok"] = False
        result["error"] = "archive-root-invalid"
        return result
    live = archived_units()
    done: list[tuple[str, str]] = []
    for unit in proof["units"]:
        reason, norm = _check_unit(unit, root, live)
        if norm is None:
            ident = unit if isinstance(unit, dict) else {}
            result["skipped"].append(
                {"slug": ident.get("slug"), "trace_id": ident.get("trace_id"), "reason": reason}
            )
            continue
        entry = {"slug": norm["slug"], "trace_id": norm["trace_id"], "files": len(norm["files"])}
        if apply:
            append_ledger(archive_entry(norm["slug"], norm["trace_id"], norm["files"], proof["proof_id"]))
            count, freed = _delete_unit_files(norm["slug"], norm["trace_id"], norm["files"])
            result["files_deleted"] += count
            result["bytes_freed"] += freed
            done.append((norm["slug"], norm["trace_id"]))
        result["archived"].append(entry)
    if done:
        _set_manifest_archived(done, True)
    return result


def _copy_verified(src: Path, dest: Path, sha: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".tmp", dir=str(dest.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        shutil.copyfile(src, tmp)
        if sha256_file(tmp) != sha:
            raise OSError(f"sha256 mismatch after copy: {src}")
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            tmp.unlink()


def rehydrate_units(trace_id: str, *, project_slug: str | None = None) -> dict[str, Any]:
    """Copy an archived trace back, verify every file, then tombstone the ledger."""

    live = archived_units()
    matches = [
        (key, entry)
        for key, entry in sorted(live.items())
        if key[1] == trace_id and (project_slug is None or key[0] == project_slug)
    ]
    result: dict[str, Any] = {"ok": True, "trace_id": trace_id, "rehydrated": [], "errors": []}
    if not matches:
        result["ok"] = False
        result["errors"].append({"reason": "not-archived"})
        return result
    root = archive_root()
    if root is None:
        result["ok"] = False
        result["errors"].append({"reason": "archive-root-invalid"})
        return result
    bucket = paths.bucket_dir()
    restored_units: list[tuple[str, str]] = []
    for (slug, tid), entry in matches:
        copied: list[Path] = []
        failure: str | None = None
        for item in entry.get("files") or []:
            rel, sha = item["path"], item["sha256"]
            dest = bucket / rel
            try:
                if dest.is_file():
                    if sha256_file(dest) != sha:
                        failure = f"local-differs:{rel}"
                        break
                    continue
                _copy_verified(root / rel, dest, sha)
                copied.append(dest)
            except OSError as exc:
                failure = f"copy-failed:{rel}:{exc}"
                break
        if failure is not None:
            for path in copied:
                path.unlink(missing_ok=True)
            result["ok"] = False
            result["errors"].append({"slug": slug, "trace_id": tid, "reason": failure})
            continue
        append_ledger(rehydrate_entry(slug, tid))
        restored_units.append((slug, tid))
        result["rehydrated"].append({"slug": slug, "trace_id": tid, "files": len(copied)})
    if restored_units:
        _set_manifest_archived(restored_units, False)
    return result

"""Archive ledger, read fallback, write guard, archive/rehydrate (rm5.13 A3).

Everything runs on temp dirs: the conftest HOME isolation plus the bucket
isolation guard keep the real ``~/.opentraces`` out of reach, and the "NAS" is a
plain temp directory behind an injected root check.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from opentraces_schema import Agent, Step, TraceRecord

from opentraces.core import bucket_archive as ba
from opentraces.core import bucket_archive_ops as ops
from opentraces.core import paths
from opentraces.core.bucket_backend import LocalBucketBackend
from opentraces.core.bucket_context_store import (
    CONTEXT_TREE_BUCKET_SCHEMA,
    read_context_tree_head,
)
from opentraces.core.bucket_envelope import (
    _write_per_trace_envelope,
    project_per_trace_exports,
    trace_v2_summary_by_id,
)
from opentraces.core.bucket_layout import (
    context_tree_head_path,
    context_tree_nodes_path,
    trace_records_root,
    trace_v1_json_path,
)
from opentraces.core.bucket_maintenance import bucket_repair
from opentraces.core.bucket_raw_sources import write_raw_source_artifact
from opentraces.core.bucket_store import (
    bucket_manifest,
    iter_trace_record_objects,
    sync_trace_records_from_local_stores,
)
from opentraces.core.bucket_trace_records import (
    read_trace_record_object,
    trace_record_path,
    write_trace_record,
)
from opentraces.core.trace_corpus import resolve
from tests.core._bucket_isolation import assert_bucket_isolated, bucket_isolation_guard  # noqa: F401

SLUG = "proj"


def _record(trace_id: str, text: str = "hello") -> TraceRecord:
    return TraceRecord(
        trace_id=trace_id,
        session_id=f"session-{trace_id}",
        agent=Agent(name="claude-code", model="anthropic/claude-opus-4-6"),
        task={"description": "archive test"},
        steps=[Step(step_index=1, role="user", content=text)],
    )


def _make_unit(trace_id: str, tmp_path: Path, slug: str = SLUG) -> None:
    """A full unit: record object+pointer, envelope, context head+nodes, raw link."""
    assert_bucket_isolated()
    record = _record(trace_id)
    write_trace_record(record, project_slug=slug, source_layer="bucket", legacy_mirror=False)
    _write_per_trace_envelope(slug, trace_id, record, [], [])
    head = {
        "schema_version": CONTEXT_TREE_BUCKET_SCHEMA,
        "project_slug": slug,
        "trace_id": trace_id,
        "node_count": 0,
        "layer_count": 0,
        "layer_refs": [],
        "capture_methods": [],
        "remote_sync": {"eligible": False},
        "digest": "sha256:" + "0" * 64,
    }
    head_path = context_tree_head_path(slug, trace_id)
    head_path.parent.mkdir(parents=True, exist_ok=True)
    head_path.write_text(json.dumps(head), encoding="utf-8")
    context_tree_nodes_path(slug, trace_id).write_text("", encoding="utf-8")
    source = tmp_path / f"{trace_id}.jsonl"
    source.write_text('{"k": "v"}\n', encoding="utf-8")
    write_raw_source_artifact(
        source, trace_id=trace_id, project_slug=slug, source_kind="test", parser="test"
    )


def _unit_files(trace_id: str, slug: str = SLUG) -> list[str]:
    return sorted(ops._local_unit_files(slug, trace_id))


def _snapshot(trace_id: str, slug: str = SLUG) -> dict[str, bytes]:
    bucket = paths.bucket_dir()
    return {rel: (bucket / rel).read_bytes() for rel in _unit_files(trace_id, slug)}


@pytest.fixture()
def nas(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "nas" / "opentraces" / "bucket"
    root.mkdir(parents=True)
    monkeypatch.setenv(ba.ARCHIVE_DIR_ENV, str(root))
    ba.set_root_check(lambda path: True)
    ba.reset_caches()
    yield root
    ba.set_root_check(None)
    ba.reset_caches()


def _mirror_and_prove(trace_id: str, nas: Path, tmp_path: Path, slug: str = SLUG) -> Path:
    files = []
    for rel, data in _snapshot(trace_id, slug).items():
        dest = nas / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        files.append({"path": rel, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)})
    proof = tmp_path / f"proof-{trace_id}.json"
    proof.write_text(
        json.dumps(
            {
                "schema_version": ops.PROOF_SCHEMA,
                "proof_id": f"proof-{trace_id}",
                "units": [{"slug": slug, "trace_id": trace_id, "files": files}],
            }
        ),
        encoding="utf-8",
    )
    return proof


# --- archive root ------------------------------------------------------------


def test_root_unconfigured_is_none(monkeypatch) -> None:
    monkeypatch.delenv(ba.ARCHIVE_DIR_ENV, raising=False)
    monkeypatch.delenv(ba.NAS_ROOT_ENV, raising=False)
    assert ba.archive_bucket_dir() is None and ba.archive_root() is None


def test_root_from_nas_root_env(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv(ba.ARCHIVE_DIR_ENV, raising=False)
    monkeypatch.setenv(ba.NAS_ROOT_ENV, str(tmp_path))
    assert ba.archive_bucket_dir() == tmp_path / "opentraces" / "bucket"


def test_root_invalid_unless_check_passes(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(ba.ARCHIVE_DIR_ENV, str(tmp_path))
    ba.set_root_check(lambda path: False)
    try:
        assert ba.archive_bucket_dir() == tmp_path and ba.archive_root() is None
    finally:
        ba.set_root_check(None)


def test_smb_root_ok_needs_smbfs_and_marker(monkeypatch, tmp_path) -> None:
    target = tmp_path / "share" / "bucket"
    table = {"/": "apfs", str(tmp_path / "share"): "smbfs"}
    monkeypatch.setattr(ba, "_mount_table", lambda: table)
    monkeypatch.setattr(ba, "_marker_present", lambda mount: True)
    assert ba.smb_root_ok(target) is True
    monkeypatch.setattr(ba, "_marker_present", lambda mount: False)
    assert ba.smb_root_ok(target) is False
    monkeypatch.setattr(ba, "_marker_present", lambda mount: True)
    table[str(tmp_path / "share")] = "apfs"
    assert ba.smb_root_ok(target) is False
    assert ba.smb_root_ok(tmp_path / "elsewhere") is False


# --- ledger ------------------------------------------------------------------


def test_ledger_append_only_with_tombstone() -> None:
    assert ba.archived_units() == {}
    ba.append_ledger(ba.archive_entry("s", "t1", [{"path": "traces/v1/s/t1/trace.json", "sha256": "a" * 64}], "p1"))
    assert ba.is_archived("s", "t1")
    ba.append_ledger(ba.rehydrate_entry("s", "t1"))
    assert not ba.is_archived("s", "t1")
    lines = ba.ledger_path().read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["op"] for line in lines] == ["archive", "rehydrate"]
    ba.append_ledger(ba.archive_entry("s", "t1", [], "p2"))
    assert ba.is_archived("s", "t1")  # re-archive after a tombstone is live again


def test_ledger_skips_corrupt_lines() -> None:
    ba.append_ledger(ba.archive_entry("s", "t1", [], "p1"))
    with ba.ledger_path().open("a", encoding="utf-8") as handle:
        handle.write("not json\n")
    ba.append_ledger(ba.archive_entry("s", "t2", [], "p1"))
    assert set(ba.archived_units()) == {("s", "t1"), ("s", "t2")}


# --- resolve_read / glob ------------------------------------------------------


def test_resolve_read_local_first_then_listed_archive_only(nas, tmp_path) -> None:
    bucket = paths.bucket_dir()
    local = bucket / "traces/v1/s/t1/trace.json"
    remote = nas / "traces/v1/s/t1/trace.json"
    remote.parent.mkdir(parents=True)
    remote.write_text("remote", encoding="utf-8")
    stray = nas / "traces/v1/s/stray/trace.json"
    stray.parent.mkdir(parents=True)
    stray.write_text("stray", encoding="utf-8")
    # Not in the ledger: the archive is not consulted at all.
    assert ba.resolve_read(local) == local
    ba.append_ledger(ba.archive_entry("s", "t1", [{"path": "traces/v1/s/t1/trace.json", "sha256": "a" * 64}], "p"))
    assert ba.resolve_read(local) == remote
    assert ba.resolve_read(bucket / "traces/v1/s/stray/trace.json") == bucket / "traces/v1/s/stray/trace.json"
    local.parent.mkdir(parents=True)
    local.write_text("local", encoding="utf-8")
    assert ba.resolve_read(local) == local


def test_resolve_read_offline_archive_returns_local_path(nas) -> None:
    local = paths.bucket_dir() / "traces/v1/s/t1/trace.json"
    ba.append_ledger(ba.archive_entry("s", "t1", [{"path": "traces/v1/s/t1/trace.json", "sha256": "a" * 64}], "p"))
    ba.set_root_check(lambda path: False)
    assert ba.resolve_read(local) == local


def test_glob_with_archive_unions_ledger_files(nas) -> None:
    root = paths.bucket_dir() / "traces/v1"
    (root / "s/live").mkdir(parents=True)
    (root / "s/live/trace.json").write_text("x", encoding="utf-8")
    ba.append_ledger(
        ba.archive_entry(
            "s",
            "gone",
            [
                {"path": "traces/v1/s/gone/trace.json", "sha256": "a" * 64},
                {"path": "traces/v1/s/gone/trail.jsonl.gz", "sha256": "b" * 64},
            ],
            "p",
        )
    )
    found = ba.glob_with_archive(root, "*/*/trace.json")
    assert [p.relative_to(root).as_posix() for p in found] == ["s/gone/trace.json", "s/live/trace.json"]
    assert ba.glob_with_archive(root, "s/*/trail.jsonl.gz") == [root / "s/gone/trail.jsonl.gz"]


# --- write guard --------------------------------------------------------------


def test_write_guard_blocks_every_bucket_writer(nas) -> None:
    from opentraces.core import _bucket_io as io

    for writer, payload in (
        (io._atomic_write_json, {"a": 1}),
        (io._atomic_write_text, "x"),
        (io._atomic_write_bytes, b"x"),
        (io._atomic_write_gzip, b"x"),
    ):
        with pytest.raises(ba.ArchiveWriteError):
            writer(nas / "traces" / "x", payload)
        assert not (nas / "traces").exists()
    with pytest.raises(ba.ArchiveWriteError):
        io._write_streaming_gzip(nas / "y.gz", ["a"])
    ba.assert_not_archive_path(paths.bucket_dir() / "ok")  # local writes stay allowed


# --- archive / rehydrate ------------------------------------------------------


def test_archive_dry_run_changes_nothing(nas, tmp_path) -> None:
    _make_unit("t1", tmp_path)
    proof = _mirror_and_prove("t1", nas, tmp_path)
    before = _snapshot("t1")
    result = ops.archive_units(proof, apply=False)
    assert result["ok"] and result["dry_run"] and len(result["archived"]) == 1
    assert _snapshot("t1") == before and not ba.ledger_path().exists()


def test_archive_apply_deletes_only_proved_unit(nas, tmp_path) -> None:
    _make_unit("t1", tmp_path)
    _make_unit("t2", tmp_path)
    proof = _mirror_and_prove("t1", nas, tmp_path)
    keep = _snapshot("t2")
    result = ops.archive_units(proof, apply=True)
    assert result["ok"] and result["files_deleted"] == 9 and result["bytes_freed"] > 0
    assert _unit_files("t1") == []
    assert _snapshot("t2") == keep
    entry = ba.archived_units()[(SLUG, "t1")]
    assert entry["proof_id"] == "proof-t1" and len(entry["files"]) == 9
    assert not (paths.bucket_dir() / "traces/v1" / SLUG / "t1").exists()


def test_archive_refuses_when_root_invalid(nas, tmp_path) -> None:
    _make_unit("t1", tmp_path)
    proof = _mirror_and_prove("t1", nas, tmp_path)
    ba.set_root_check(lambda path: False)
    result = ops.archive_units(proof, apply=True)
    assert not result["ok"] and result["error"] == "archive-root-invalid"
    assert _unit_files("t1") and not ba.ledger_path().exists()


@pytest.mark.parametrize(
    "tamper, reason",
    [
        ("archive-corrupt", "archive-mismatch"),
        ("archive-missing", "archive-mismatch"),
        ("local-changed", "local-changed-since-proof"),
        ("extra-local", "unlisted-local-file"),
    ],
)
def test_archive_refuses_unverified_unit(nas, tmp_path, tamper, reason) -> None:
    _make_unit("t1", tmp_path)
    proof = _mirror_and_prove("t1", nas, tmp_path)
    rel = f"traces/v1/{SLUG}/t1/trace.json"
    if tamper == "archive-corrupt":
        (nas / rel).write_bytes(b"bitrot")
    elif tamper == "archive-missing":
        (nas / rel).unlink()
    elif tamper == "local-changed":
        (paths.bucket_dir() / rel).write_bytes(b"newer")
    else:
        (paths.bucket_dir() / f"traces/v1/{SLUG}/t1/late.txt").write_text("late", encoding="utf-8")
    result = ops.archive_units(proof, apply=True)
    assert result["archived"] == [] and result["skipped"][0]["reason"].startswith(reason)
    assert not ba.ledger_path().exists()
    assert (paths.bucket_dir() / rel).exists()


def test_archive_refuses_paths_outside_the_unit(nas, tmp_path) -> None:
    _make_unit("t1", tmp_path)
    _make_unit("t2", tmp_path)
    proof = _mirror_and_prove("t1", nas, tmp_path)
    doc = json.loads(proof.read_text())
    other = f"traces/v1/{SLUG}/t2/trace.json"
    data = (paths.bucket_dir() / other).read_bytes()
    (nas / other).parent.mkdir(parents=True, exist_ok=True)
    (nas / other).write_bytes(data)
    doc["units"][0]["files"].append({"path": other, "sha256": hashlib.sha256(data).hexdigest()})
    proof.write_text(json.dumps(doc))
    result = ops.archive_units(proof, apply=True)
    assert result["skipped"][0]["reason"].startswith("path-outside-unit")
    assert _unit_files("t2") and _unit_files("t1")


def test_archive_ignores_units_not_in_proof_and_is_idempotent(nas, tmp_path) -> None:
    _make_unit("t1", tmp_path)
    proof = _mirror_and_prove("t1", nas, tmp_path)
    ops.archive_units(proof, apply=True)
    again = ops.archive_units(proof, apply=True)
    assert again["skipped"][0]["reason"] == "already-archived" and again["files_deleted"] == 0


def test_bad_proof_is_rejected(tmp_path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": "nope"}))
    with pytest.raises(ops.ProofError):
        ops.load_proof(bad)


def test_rehydrate_restores_bytes_and_tombstones(nas, tmp_path) -> None:
    _make_unit("t1", tmp_path)
    before = _snapshot("t1")
    ops.archive_units(_mirror_and_prove("t1", nas, tmp_path), apply=True)
    result = ops.rehydrate_units("t1")
    assert result["ok"] and result["rehydrated"][0]["files"] == len(before)
    assert _snapshot("t1") == before
    assert not ba.is_archived(SLUG, "t1")
    ops_after = [json.loads(line)["op"] for line in ba.ledger_path().read_text().splitlines()]
    assert ops_after == ["archive", "rehydrate"]


def test_rehydrate_fails_closed_on_corrupt_archive(nas, tmp_path) -> None:
    _make_unit("t1", tmp_path)
    ops.archive_units(_mirror_and_prove("t1", nas, tmp_path), apply=True)
    (nas / f"traces/v1/{SLUG}/t1/trace.json").write_bytes(b"rot")
    result = ops.rehydrate_units("t1")
    assert not result["ok"] and ba.is_archived(SLUG, "t1")
    assert _unit_files("t1") == []  # partial copies were cleaned up


def test_rehydrate_unknown_trace(nas) -> None:
    assert ops.rehydrate_units("missing")["ok"] is False


# --- journey ------------------------------------------------------------------


def test_journey_archive_read_fallback_no_resurrection_rehydrate(nas, tmp_path) -> None:
    _make_unit("t1", tmp_path)
    _make_unit("t2", tmp_path)
    before = _snapshot("t1")
    record_before = read_trace_record_object(trace_record_path(SLUG, "t1")).record.model_dump(mode="json")
    manifest_before = bucket_manifest(write=True, heal=False)
    trace_json_before = LocalBucketBackend().get_trace_json("t1")
    assert all("archived" not in row for row in manifest_before["traces"])

    ops.archive_units(_mirror_and_prove("t1", nas, tmp_path), apply=True)
    assert _unit_files("t1") == []

    # 1. reads through the fallback are byte-identical.
    obj = read_trace_record_object(trace_record_path(SLUG, "t1"))
    assert obj is not None and obj.record.model_dump(mode="json") == record_before
    assert {o.trace_id for o in iter_trace_record_objects()} == {"t1", "t2"}
    assert LocalBucketBackend().get_trace_json("t1") == trace_json_before
    assert ba.resolve_read(trace_v1_json_path(SLUG, "t1")).read_bytes() == before[f"traces/v1/{SLUG}/t1/trace.json"]
    assert read_context_tree_head(SLUG, "t1")["trace_id"] == "t1"
    assert resolve("t1") is not None and resolve("t1").project_slug == SLUG
    assert trace_v2_summary_by_id("t1")["trace_id"] == "t1"

    # 2. nothing may bring it back locally.
    sync_trace_records_from_local_stores()
    project_per_trace_exports(None, project_slug=SLUG, trace_id="t1", record=_record("t1", "different"))
    write_trace_record(_record("t1", "rewritten"), project_slug=SLUG, source_layer="bucket")
    source = tmp_path / "t1.jsonl"
    write_raw_source_artifact(source, trace_id="t1", project_slug=SLUG, source_kind="x", parser="x")
    repair = bucket_repair()
    assert repair["errors"] == []
    assert _unit_files("t1") == []
    assert not (paths.bucket_dir() / "traces/v1" / SLUG / "t1").exists()

    # 3. the manifest keeps the row (marked archived) and the digest is unchanged.
    manifest_after = bucket_manifest(write=True, heal=False)
    rows = {row["trace_id"]: row for row in manifest_after["traces"]}
    assert rows["t1"]["archived"] is True and "archived" not in rows["t2"]
    assert manifest_after["bucket_digest"] == manifest_before["bucket_digest"]
    persisted = json.loads((paths.bucket_dir() / "manifest.json").read_text())
    assert {r["trace_id"]: r.get("archived") for r in persisted["traces"]} == {"t1": True, "t2": None}

    # 4. an offline archive never degrades the persisted manifest.
    on_disk = (paths.bucket_dir() / "manifest.json").read_bytes()
    ba.set_root_check(lambda path: False)
    ba.reset_caches()
    bucket_manifest(write=True, heal=False)
    assert (paths.bucket_dir() / "manifest.json").read_bytes() == on_disk
    ba.set_root_check(lambda path: True)
    ba.reset_caches()

    # 5. rehydrate brings every byte back, and writers work again.
    assert ops.rehydrate_units("t1")["ok"]
    assert _snapshot("t1") == before
    bucket_repair()
    rows = {row["trace_id"]: row for row in bucket_manifest(write=True, heal=False)["traces"]}
    assert "archived" not in rows["t1"]


def test_repair_with_rehydrate_flag_may_rebuild(nas, tmp_path) -> None:
    _make_unit("t1", tmp_path)
    ops.archive_units(_mirror_and_prove("t1", nas, tmp_path), apply=True)
    with ba.allow_rehydrate():
        write_trace_record(_record("t1", "rebuilt"), project_slug=SLUG, source_layer="bucket", legacy_mirror=False)
    assert (trace_records_root() / SLUG / "t1" / "current.json").exists()


def test_cli_archive_and_rehydrate_round_trip(nas, tmp_path) -> None:
    from click.testing import CliRunner

    from opentraces.cli.bucket import bucket_group

    _make_unit("t1", tmp_path)
    proof = _mirror_and_prove("t1", nas, tmp_path)
    runner = CliRunner()
    dry = runner.invoke(bucket_group, ["archive", "--proof", str(proof), "--json"])
    assert dry.exit_code == 0 and json.loads(dry.output)["dry_run"] is True
    assert _unit_files("t1")
    applied = runner.invoke(bucket_group, ["archive", "--proof", str(proof), "--apply", "--json"])
    assert applied.exit_code == 0 and json.loads(applied.output)["dry_run"] is False
    assert _unit_files("t1") == []
    back = runner.invoke(bucket_group, ["rehydrate", "t1", "--json"])
    assert back.exit_code == 0 and _unit_files("t1")
    shutil.rmtree(nas)
    gone = runner.invoke(bucket_group, ["archive", "--proof", str(proof), "--apply", "--json"])
    assert gone.exit_code == 0 and json.loads(gone.output)["skipped"][0]["reason"].startswith("archive-mismatch")
    ba.set_root_check(lambda path: False)
    ba.reset_caches()
    bad = runner.invoke(bucket_group, ["archive", "--proof", str(proof), "--apply", "--json"])
    assert bad.exit_code == 3 and json.loads(bad.output)["status"] == "error"

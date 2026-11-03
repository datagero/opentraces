"""Re-ingesting a session whose trace unit is archived must not resurrect it."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from opentraces.core import bucket_archive as ba
from opentraces.core import bucket_archive_ops as ops
from opentraces.core import paths
from opentraces.core.bucket_store import iter_trace_record_objects
from tests.core._bucket_isolation import assert_bucket_isolated, bucket_isolation_guard  # noqa: F401
from tests.core.test_ingest import _append_turns, _write_jsonl, project_dir  # noqa: F401


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


def _bucket_files() -> dict[str, str]:
    bucket = paths.bucket_dir()
    return {
        p.relative_to(bucket).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in bucket.rglob("*")
        if p.is_file() and p.relative_to(bucket).parts[0] in {"traces", "objects", "contexts"}
    }


def test_reingest_of_live_session_leaves_archived_unit_archived(nas, tmp_path, project_dir) -> None:  # noqa: F811
    from opentraces.core.ingest import ingest_one_session

    assert_bucket_isolated()
    session = "sess-archived"
    jsonl = _write_jsonl(project_dir, session, turns=3)
    result = ingest_one_session(jsonl, project_dir)
    assert result.action == "new"
    objects = iter_trace_record_objects()
    assert len(objects) == 1
    slug, trace_id = objects[0].project_slug, objects[0].trace_id

    before = _bucket_files()
    unit = {rel: sha for rel, sha in before.items() if ops._path_in_unit(rel, slug, trace_id)}
    assert unit
    for rel in unit:
        dest = nas / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((paths.bucket_dir() / rel).read_bytes())
    proof = tmp_path / "proof.json"
    proof.write_text(
        json.dumps(
            {
                "schema_version": ops.PROOF_SCHEMA,
                "proof_id": "ingest-proof",
                "units": [
                    {
                        "slug": slug,
                        "trace_id": trace_id,
                        "files": [{"path": rel, "sha256": sha} for rel, sha in unit.items()],
                    }
                ],
            }
        )
    )
    assert ops.archive_units(proof, apply=True)["files_deleted"] == len(unit)
    assert not any(ops._path_in_unit(rel, slug, trace_id) for rel in _bucket_files())

    # The session is still on disk and grows: the watcher re-ingests it.
    _append_turns(jsonl, session, start=4, count=2)
    again = ingest_one_session(jsonl, project_dir, reparse=True)
    assert again.error is None
    assert not any(ops._path_in_unit(rel, slug, trace_id) for rel in _bucket_files())
    assert ba.is_archived(slug, trace_id)

    # Explicit rehydrate restores the archived bytes exactly.
    assert ops.rehydrate_units(trace_id)["ok"]
    restored = _bucket_files()
    assert {rel: restored[rel] for rel in unit} == unit

"""Stop-hook re-capture throttle: a growing session is re-captured at most once
per interval by the hook; deferred growth is captured by the next sweep."""

from __future__ import annotations

import json
import os
import time

import pytest

from opentraces.core.bucket_trace_records import trace_record_path
from opentraces.core.config import get_project_dir
from opentraces.core.ingest import ingest_one_session
from tests.core.test_ingest import _append_turns, _write_jsonl

SID = "sess-throttle-0001"


@pytest.fixture
def project_dir(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / ".opentraces.json").write_text(json.dumps({
        "marker_version": "2", "project_id": "test-project-0000", "review_policy": "review",
        "push_policy": "manual", "agents": ["claude-code"],
    }))
    return proj


def _pointer(project_dir, trace_id):
    return trace_record_path(get_project_dir(project_dir).name, trace_id)


def test_hook_defers_recent_refresh_and_sweep_catches_up(project_dir) -> None:
    path = _write_jsonl(project_dir, SID, turns=2)
    first = ingest_one_session(path, project_dir, min_recapture_interval_s=300)
    assert first.action == "new"  # a new trace is never deferred
    pointer = _pointer(project_dir, first.trace_id)
    before = pointer.read_bytes()

    _append_turns(path, SID, start=3, count=1)
    deferred = ingest_one_session(path, project_dir, min_recapture_interval_s=300)
    assert deferred.action == "skipped" and deferred.error.startswith("deferred:")
    assert pointer.read_bytes() == before  # no new record written

    swept = ingest_one_session(path, project_dir)  # watcher sweep: no throttle
    assert swept.action == "refreshed"  # growth was not lost: bookmark not advanced
    assert pointer.read_bytes() != before


def test_hook_refreshes_once_interval_has_passed(project_dir) -> None:
    path = _write_jsonl(project_dir, SID, turns=2)
    first = ingest_one_session(path, project_dir, min_recapture_interval_s=300)
    pointer = _pointer(project_dir, first.trace_id)
    old = time.time() - 301
    os.utime(pointer, (old, old))
    _append_turns(path, SID, start=3, count=1)
    again = ingest_one_session(path, project_dir, min_recapture_interval_s=300)
    assert again.action == "refreshed"


def test_zero_interval_and_reparse_never_defer(project_dir) -> None:
    path = _write_jsonl(project_dir, SID, turns=2)
    ingest_one_session(path, project_dir)
    _append_turns(path, SID, start=3, count=1)
    assert ingest_one_session(path, project_dir, min_recapture_interval_s=0).action == "refreshed"
    _append_turns(path, SID, start=4, count=1)
    assert ingest_one_session(path, project_dir, reparse=True, min_recapture_interval_s=300).action == "refreshed"

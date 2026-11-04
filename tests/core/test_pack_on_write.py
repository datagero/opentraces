"""Pack on write: superseded TraceRecord versions never pile up loose.

``write_trace_record`` folds superseded objects into small pack segments once a
trace has ``PACK_ON_WRITE_MIN_LOOSE`` of them; ``bucket pack`` merges segments
into the main pack; ``bucket unpack`` restores every version byte-exact.
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import pytest
from opentraces_schema import Agent, Step, TraceRecord

from opentraces.core import bucket_pack_codecs as codecs
from opentraces.core import bucket_trace_pack as btp
from opentraces.core.bucket_layout import trace_records_root
from opentraces.core.bucket_trace_records import (
    read_trace_record_object,
    trace_record_path,
    write_trace_record,
)
from tests.core._bucket_isolation import assert_bucket_isolated, bucket_isolation_guard  # noqa: F401

N = btp.PACK_ON_WRITE_MIN_LOOSE


@pytest.fixture(autouse=True, params=codecs.WRITE_PREFERENCE)
def codec(request, monkeypatch) -> str:
    if request.param not in codecs.available_codecs():
        pytest.skip(f"codec {request.param} not installed")
    monkeypatch.setattr(codecs, "WRITE_PREFERENCE", (request.param,))
    monkeypatch.delenv(btp.PACK_ON_WRITE_ENV, raising=False)
    return request.param


def _grown(turns: int, trace_id: str = "t1") -> TraceRecord:
    return TraceRecord(
        trace_id=trace_id,
        session_id=f"session-{trace_id}",
        agent=Agent(name="claude-code", model="anthropic/claude-opus-4-6"),
        task={"description": "growing"},
        steps=[Step(step_index=i, role="user", content=f"step {i} " + "abc " * 300) for i in range(1, turns + 1)],
    )


def _capture(turns: range, trace_id: str = "t1") -> tuple[Path, dict[str, str]]:
    """Write one record per turn count; return (trace dir, {name: sha} written)."""
    assert_bucket_isolated()
    written: dict[str, str] = {}
    for n in turns:
        obj = write_trace_record(_grown(n, trace_id), project_slug="proj", source_layer="bucket")
        written[obj.path.name] = hashlib.sha256(obj.path.read_bytes()).hexdigest()
    return trace_records_root() / "proj" / trace_id, written


def _loose(trace_dir: Path) -> list[Path]:
    return [p for p in trace_dir.iterdir() if btp._OBJECT_NAME.match(p.name)]


def _versions(trace_dir: Path) -> dict[str, str]:
    return {name: hashlib.sha256(data).hexdigest() for name, data, _ in btp.iter_trace_record_versions(trace_dir)}


def test_growing_session_keeps_bounded_loose_history() -> None:
    trace_dir, written = _capture(range(1, 3 * N + 2))
    loose = _loose(trace_dir)
    assert len(loose) <= N  # current + fewer than N superseded
    assert btp.segment_paths(trace_dir), "superseded versions went into segments"
    assert not btp.pack_path(trace_dir).exists()  # capture path never writes the main pack
    assert _versions(trace_dir) == written  # every version still readable, byte-exact


def test_current_record_stays_loose_and_readable() -> None:
    _capture(range(1, N + 3))
    current = read_trace_record_object(trace_record_path("proj", "t1"))
    assert current is not None and len(current.record.steps) == N + 2
    assert current.path.is_file()


def test_superseded_object_reads_back_from_a_segment() -> None:
    trace_dir, written = _capture(range(1, N + 3))
    first = next(iter(written))
    assert not (trace_dir / first).exists()
    record = read_trace_record_object(trace_dir / first)
    assert record is not None and len(record.record.steps) == 1


def test_bucket_pack_merges_segments_into_main_pack() -> None:
    trace_dir, written = _capture(range(1, 2 * N + 3))
    assert len(btp.segment_paths(trace_dir)) >= 2
    old = time.time() - 3 * 3600
    for p in _loose(trace_dir):
        os.utime(p, (old, old))
    report = btp.pack_superseded_trace_records()
    assert report["ok"], report
    assert report["segments_merged"] >= 2
    assert btp.segment_paths(trace_dir) == []
    assert btp.pack_path(trace_dir).is_file()
    assert len(_loose(trace_dir)) == 1
    assert _versions(trace_dir) == written
    checked, errors, _skipped = btp.verify_trace_packs()
    assert checked >= 1 and errors == []


def test_unpack_restores_main_pack_and_segments() -> None:
    trace_dir, written = _capture(range(1, N + 3))
    old = time.time() - 3 * 3600
    for p in _loose(trace_dir):
        os.utime(p, (old, old))
    btp.pack_superseded_trace_records()
    _capture(range(N + 3, 2 * N + 5))  # new segment on top of a main pack
    _, written_more = _capture(range(2 * N + 5, 2 * N + 6))
    assert btp.pack_path(trace_dir).is_file() and btp.segment_paths(trace_dir)
    expected = _versions(trace_dir)
    assert set(written) <= set(expected) and set(written_more) <= set(expected)
    report = btp.unpack_superseded_trace_records()
    assert report["ok"], report
    assert btp.trace_packs(trace_dir) == []
    restored = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in _loose(trace_dir)}
    assert set(restored.values()) == set(expected.values())


def test_pack_on_write_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv(btp.PACK_ON_WRITE_ENV, "0")
    trace_dir, _ = _capture(range(1, N + 4))
    assert len(_loose(trace_dir)) == N + 3
    assert btp.trace_packs(trace_dir) == []


def test_pack_failure_never_fails_the_capture(monkeypatch) -> None:
    def boom(*_a, **_k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(btp, "_write_pack", boom)
    trace_dir, written = _capture(range(1, N + 4))
    assert len(_loose(trace_dir)) == N + 3  # nothing removed when the segment failed
    assert {p.name for p in _loose(trace_dir)} == set(written)


def test_concurrent_consolidation_ignores_vanished_loose_file() -> None:
    trace_dir, written = _capture(range(1, 4))
    old = time.time() - 3 * 3600
    for p in _loose(trace_dir):
        os.utime(p, (old, old))
    # A capture packs the loose objects into a segment between listing and read.
    real = btp._loose_member
    calls = {"n": 0}

    def racing(path):
        calls["n"] += 1
        if calls["n"] == 1:
            btp.pack_on_write(trace_dir, min_loose=1)
        return real(path)

    import pytest as _pytest

    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(btp, "_loose_member", racing)
        result = btp.pack_trace_dir(trace_dir, min_age_seconds=0)
    assert _versions(trace_dir) == written
    assert result["records_packed"] <= 2

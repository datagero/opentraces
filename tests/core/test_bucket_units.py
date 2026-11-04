"""``bucket units``: per-(slug, trace) inventory with event time, files and raw blobs."""

from __future__ import annotations

import json

from click.testing import CliRunner

from opentraces.cli.bucket import bucket_group
from opentraces.core import paths
from opentraces.core.bucket_units import bucket_units
from tests.core._bucket_isolation import assert_bucket_isolated, bucket_isolation_guard  # noqa: F401


def _w(rel: str, text: str) -> None:
    p = paths.bucket_dir() / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def _unit(slug: str, tid: str, end: str | None) -> None:
    assert_bucket_isolated()
    doc = {"trace_id": tid, "timestamp_start": "2026-01-01T00:00:00Z"}
    if end:
        doc["timestamp_end"] = end
    _w(f"traces/v1/{slug}/{tid}/trace.json", json.dumps(doc))
    _w(f"contexts/v1/{slug}/{tid}/head.json", "{}")
    blob = f"objects/raw/v1/blobs/ab/{tid}.jsonl"
    _w(blob, "raw")
    _w(f"objects/raw/v1/sources/{slug}/{tid}.json", json.dumps({"blob_path": blob}))


def test_units_event_time_files_and_blobs():
    _unit("p", "t1", "2026-03-01T00:00:00Z")
    _unit("p", "t2", None)
    _unit("q", "t3", "2026-06-01T00:00:00Z")
    out = bucket_units()
    assert out["count"] == 3
    u = {x["trace_id"]: x for x in out["units"]}
    assert u["t1"]["last_event_at"] == "2026-03-01T00:00:00Z"
    assert u["t2"]["last_event_at"] == "2026-01-01T00:00:00Z"  # falls back to start, not mtime
    assert u["t1"]["raw_blobs"] == ["objects/raw/v1/blobs/ab/t1.jsonl"]
    assert {f["path"] for f in u["t1"]["files"]} == {
        "traces/v1/p/t1/trace.json", "contexts/v1/p/t1/head.json", "objects/raw/v1/sources/p/t1.json"}
    assert [x["trace_id"] for x in bucket_units(slug="q")["units"]] == ["t3"]
    assert [x["trace_id"] for x in bucket_units(before="2026-04-01T00:00:00Z")["units"]] == ["t1", "t2"]


def test_units_cli_json():
    _unit("p", "t1", "2026-03-01T00:00:00Z")
    r = CliRunner().invoke(bucket_group, ["units", "--json", "--slug", "p"])
    assert r.exit_code == 0, r.output
    doc = json.loads(r.output)
    assert doc["schema_version"] == "opentraces.bucket.units.v1" and doc["count"] == 1

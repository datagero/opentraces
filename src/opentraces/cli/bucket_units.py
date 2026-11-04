"""``bucket units``: JSON inventory of trace units (slug, trace, event time, files, raw blobs)."""

from __future__ import annotations

import click

from ._envelope import envelope
from ._help import OpentracesCommand
from ._options import dump_json as _dump_json


@click.command("units", cls=OpentracesCommand, hidden=True)
@click.option("--slug", default=None, metavar="SLUG", help="Only this project slug.")
@click.option("--before", default=None, metavar="ISO", help="Only units whose last_event_at is before this ISO time.")
@click.option("--json", "as_json", is_flag=True, help="Emit structured JSON.")
def bucket_units_cmd(slug: str | None, before: str | None, as_json: bool) -> None:
    """List trace units with event time, files and owned raw blobs (read-only)."""
    from ..core.bucket_units import UNITS_SCHEMA, bucket_units

    result = bucket_units(slug=slug, before=before)
    if as_json:
        click.echo(_dump_json(envelope(UNITS_SCHEMA, **result)))
        return
    click.echo(f"{result['count']} unit(s)")
    for u in result["units"]:
        click.echo(f"  {u['slug']}/{u['trace_id']}  {u['last_event_at'] or '-'}  files={len(u['files'])}")

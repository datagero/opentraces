"""``bucket archive`` / ``bucket rehydrate``: move trace units to and from the archive."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from ._envelope import envelope
from ._help import OpentracesCommand
from ._options import dump_json as _dump_json


def _emit(schema: str, result: dict, as_json: bool, title: str) -> None:
    ok = bool(result.get("ok", True))
    fields = {k: v for k, v in result.items() if k not in {"schema_version", "status"}}
    payload = envelope(schema, status="ok" if ok else "error", **fields)
    if as_json:
        click.echo(_dump_json(payload))
    else:
        click.echo(f"{title}:")
        for key, value in result.items():
            if key not in {"schema_version", "archived", "skipped", "rehydrated", "errors"}:
                click.echo(f"  {key}: {value}")
        for label in ("archived", "rehydrated"):
            if label in result:
                click.echo(f"  {label}: {len(result[label])} unit(s)")
        for item in result.get("skipped") or []:
            click.echo(f"  skipped {item.get('slug')}/{item.get('trace_id')}: {item.get('reason')}")
        for item in result.get("errors") or []:
            click.echo(f"  error: {item}", err=True)
    if not ok:
        sys.exit(3)


@click.command("archive", cls=OpentracesCommand, hidden=True)
@click.option("--proof", "proof", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Verified-units proof written by the archive tool (sha256 per file).")
@click.option("--dry-run", "mode", flag_value="dry-run", default=True, help="Check every unit, change nothing (default).")
@click.option("--apply", "mode", flag_value="apply", help="Append the ledger, then delete the local files.")
@click.option("--json", "as_json", is_flag=True, help="Emit structured JSON.")
def bucket_archive_cmd(proof: Path, mode: str, as_json: bool) -> None:
    """Remove local copies of trace units that are verified on the archive.

    Each unit in the proof is re-hashed on the archive root first; a unit that
    is not in the proof, whose archive copy differs, whose local files changed
    since the proof, or that has local files the proof does not list, is
    skipped and reported. Refuses everything when the archive root is not a
    live NAS mount.
    """
    from ..core.bucket_archive_ops import ProofError, archive_units

    try:
        result = archive_units(proof, apply=(mode == "apply"))
    except ProofError as exc:
        raise click.ClickException(str(exc)) from exc
    _emit("opentraces.bucket.archive.v1", result, as_json, f"Bucket archive ({mode})")


@click.command("rehydrate", cls=OpentracesCommand, hidden=True)
@click.argument("trace_id")
@click.option("--project", "project_slug", metavar="SLUG", default=None, help="Disambiguate by project slug.")
@click.option("--json", "as_json", is_flag=True, help="Emit structured JSON.")
def bucket_rehydrate_cmd(trace_id: str, project_slug: str | None, as_json: bool) -> None:
    """Copy an archived trace back from the archive and clear its ledger entry."""
    from ..core.bucket_archive_ops import rehydrate_units

    result = rehydrate_units(trace_id, project_slug=project_slug)
    _emit("opentraces.bucket.rehydrate.v1", result, as_json, "Bucket rehydrate")

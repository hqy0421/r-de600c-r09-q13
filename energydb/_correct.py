"""Corrections of published actuals: the explicit, audited revision entry point.

A correction re-publishes an already-published actual reading with a new
value, a mandatory business reason (stored as the row's ``annotation``) and
an author (``changed_by``), stamped with a fresh ``knowledge_time``. Storage
is append-only (timedb ``series_values``), so the two read contracts fall
out of the existing knowledge-window selectors:

* **Latest results include the correction.** Default reads collapse to the
  row with the largest ``(knowledge_time, change_time)`` per
  ``valid_time``, which is the correction as soon as it lands.
* **Historical reports reproduce under the pre-revision basis.** A read
  bounded by ``end_known=<correction knowledge_time>`` filters the
  correction out and returns exactly what a report generated before it saw.

The version a correction replaces (its *superseded* version — the original
record identity) is captured from ClickHouse at correction time, returned on
the :class:`CorrectionResult`, and frozen (capped) into the run's
``run_params`` so the audit trail survives even if the caller discards the
return value.

Corrections are restricted to FLAT series: an OVERLAPPING series is
versioned by republication (a new ``knowledge_time`` forecast run through
``write()``), not by in-place correction of one reading.
"""

from __future__ import annotations

import asyncio
import math
from datetime import UTC, datetime
from typing import Any

import polars as pl
from timedb import profiling

from energydb import runs as runs_mod
from energydb._io import _ROUTING_AND_META_COLS, WriteResult, annotate_undefined_table
from energydb._persist import apply_manifest_unit_conversion
from energydb.errors import CorrectionTargetError, ManifestError, ValidationError
from energydb.paths import resolve_manifest

# Columns correct() stamps itself; a manifest carrying them would be
# ambiguous with the entry point's own audit semantics.
_CORRECTION_OWNED_COLS = ("knowledge_time", "change_time", "changed_by", "annotation", "run_id")

# Identity columns projected onto the result's superseded frame, in order:
# whichever routing columns the manifest carried, then the series key.
_AUDIT_IDENTITY_COLS = (
    "path",
    "node_uuid",
    "edge_uuid",
    "from_path",
    "to_path",
    "edge_type",
    "edge_name",
    "data_type",
    "name",
)

# The superseded detail frozen into run_params is capped; the full frame is
# on the CorrectionResult, and CH itself holds the complete version chain.
_SUPERSEDED_RUN_PARAMS_CAP = 500

_SUPERSEDED_SCHEMA = {
    "series_id": pl.Int64,
    "valid_time": pl.Datetime("us", "UTC"),
    "knowledge_time": pl.Datetime("us", "UTC"),
    "change_time": pl.Datetime("us", "UTC"),
    "value": pl.Float64,
    "changed_by": pl.Utf8,
    "annotation": pl.Utf8,
    "run_id": pl.UInt64,
}


class CorrectionResult(WriteResult):
    """The ``run_id`` of a correction write, carrying the superseded versions.

    Subclasses :class:`WriteResult` (itself an ``int`` run_id), so anything
    that works for a plain write works here. ``superseded`` holds one row
    per corrected reading: the manifest's routing columns plus
    ``data_type``, ``name``, ``valid_time`` and the replaced version's
    ``value``, ``knowledge_time``, ``change_time``, ``changed_by``,
    ``annotation`` and ``run_id`` — the original record's identity, captured
    from storage at correction time.
    """

    superseded: pl.DataFrame

    def __new__(cls, run_id: int, written: int, skipped: int, superseded: pl.DataFrame) -> CorrectionResult:
        self = super().__new__(cls, run_id, written, skipped)
        self.superseded = superseded
        return self

    def __repr__(self) -> str:
        return (
            f"CorrectionResult(run_id={int(self)}, written={self.written}, "
            f"skipped={self.skipped}, superseded={self.superseded.height})"
        )


def _validate_correction_inputs(df: pl.DataFrame, *, reason: str, changed_by: str, knowledge_time: datetime | None) -> None:
    """Reject a correction that is missing its audit payload or data columns.

    ``reason`` and ``changed_by`` are the point of the entry point — a
    correction without a recorded business justification and author is just
    a write, and belongs on ``write()``.
    """
    if not isinstance(reason, str) or not reason.strip():
        raise ValidationError(
            "correct() requires a non-empty 'reason': the business justification "
            "for the correction, stored as the annotation of every corrected row."
        )
    if not isinstance(changed_by, str) or not changed_by.strip():
        raise ValidationError(
            "correct() requires a non-empty 'changed_by': the author of the "
            "correction, stored on every corrected row."
        )
    if knowledge_time is not None and knowledge_time.tzinfo is None:
        raise ValidationError("knowledge_time must be timezone-aware.")
    missing_cols = {"valid_time", "value"} - set(df.columns)
    if missing_cols:
        raise ManifestError(f"Correction manifest is missing required columns: {sorted(missing_cols)}")
    owned = [c for c in _CORRECTION_OWNED_COLS if c in df.columns]
    if owned:
        raise ValidationError(
            f"Correction manifests must not carry {owned}: correct() stamps those "
            "columns itself. Pass knowledge_time= as a kwarg; reason= / changed_by= "
            "take the place of annotation / changed_by columns."
        )


def _fetch_latest_versions(ch, keys: pl.DataFrame) -> pl.DataFrame:
    """Winning stored version per ``(series_id, valid_time)`` for the exact key set.

    ``keys`` carries deduplicated ``(series_id, valid_time, retention)`` rows.
    The collapse mirrors timedb's latest read (largest ``(knowledge_time,
    change_time)`` wins), so the captured superseded version is exactly what
    a latest read returned just before the correction landed.
    """
    if keys.is_empty():
        return pl.DataFrame(schema=_SUPERSEDED_SCHEMA)
    params = {
        "sids": keys["series_id"].unique().to_list(),
        "rets": keys["retention"].unique().to_list(),
        "vts": keys["valid_time"].unique().to_list(),
    }
    sql = """
    SELECT series_id, valid_time, knowledge_time, change_time, value, changed_by, annotation, run_id
    FROM series_values
    WHERE series_id IN {sids:Array(UInt64)}
      AND retention IN {rets:Array(String)}
      AND valid_time IN {vts:Array(DateTime64(6, 'UTC'))}
    ORDER BY series_id, valid_time, knowledge_time DESC, change_time DESC
    LIMIT 1 BY series_id, valid_time
    """
    result = pl.from_arrow(ch.query_arrow(sql, parameters=params))
    assert isinstance(result, pl.DataFrame)
    if result.is_empty():
        return pl.DataFrame(schema=_SUPERSEDED_SCHEMA)
    # Normalize for the join back onto the resolved manifest (Int64 series_id)
    # and for the caller-facing frame (Utf8 over LowCardinality Categoricals).
    return result.with_columns(
        pl.col("series_id").cast(pl.Int64),
        pl.col("changed_by").cast(pl.Utf8),
        pl.col("annotation").cast(pl.Utf8),
    )


def _require_all_targets_published(resolved: pl.DataFrame, superseded: pl.DataFrame) -> None:
    """Raise :class:`CorrectionTargetError` for addressed readings with no stored value.

    Every resolved ``(series_id, valid_time)`` must have a winning stored
    version; a correction of a never-published reading is a caller bug
    (wrong series, wrong timestamp) that a silent first publish would mask.
    """
    if superseded.height == resolved.height:
        return
    missing_rows = resolved.join(
        superseded.select("series_id", "valid_time"), on=["series_id", "valid_time"], how="anti"
    )
    if missing_rows.is_empty():
        return
    route_cols = [c for c in _AUDIT_IDENTITY_COLS if c in missing_rows.columns]
    missing = [
        tuple(str(v) for v in row)
        for row in missing_rows.select([*route_cols, "valid_time"]).sort("valid_time").iter_rows()
    ]
    raise CorrectionTargetError(
        f"Correction target not found: {len(missing)} (series, valid_time) pair(s) have no "
        "published value to correct. Corrections revise published readings only; "
        "use write() for a first publish.",
        missing=missing,
    )


def _require_correction_wins(superseded: pl.DataFrame, *, kt: datetime) -> None:
    """Require the correction's ``knowledge_time`` to move past every version it replaces.

    A correction stamped at or before the superseded version's
    ``knowledge_time`` would lose the latest-read collapse to the very row
    it corrects — it would land in storage but never surface in current
    results, which defeats the entry point's contract.
    """
    if superseded.is_empty():
        return
    too_early = superseded.filter(pl.col("knowledge_time") >= kt)
    if too_early.is_empty():
        return
    worst = too_early["knowledge_time"].max()
    raise ValidationError(
        f"knowledge_time {kt.isoformat()} does not move past the knowledge_time of "
        f"{too_early.height} version(s) being corrected (latest: {worst.isoformat()}); "
        "the correction would not become the latest readable version. "
        "Pass a later knowledge_time, or omit it to use the current time."
    )


def _json_value(v: Any) -> Any:
    """Make a stored value JSONB-safe: NaN (the storage null sentinel) → None."""
    return None if (isinstance(v, float) and math.isnan(v)) else v


def _correction_run_params(audit: pl.DataFrame, *, reason: str, changed_by: str, kt: datetime) -> dict:
    """The ``correction`` block frozen into the run's ``run_params``.

    One entry per corrected reading (capped at
    ``_SUPERSEDED_RUN_PARAMS_CAP``): the series identity, the reading's
    ``valid_time``, and the superseded version's ``knowledge_time`` /
    ``value`` — enough to answer "what did this run replace, and why"
    without touching ClickHouse.
    """
    entries = [
        {
            "series_id": r["series_id"],
            "data_type": r["data_type"],
            "name": r["name"],
            "valid_time": r["valid_time"].isoformat(),
            "knowledge_time": r["knowledge_time"].isoformat(),
            "value": _json_value(r["value"]),
        }
        for r in audit.head(_SUPERSEDED_RUN_PARAMS_CAP).iter_rows(named=True)
    ]
    return {
        "reason": reason,
        "changed_by": changed_by,
        "knowledge_time": kt.isoformat(),
        "superseded": entries,
        "superseded_truncated": audit.height > _SUPERSEDED_RUN_PARAMS_CAP,
    }


async def correct_manifest(
    client,
    td,
    df: pl.DataFrame,
    *,
    reason: str,
    changed_by: str,
    knowledge_time: datetime | None = None,
    run_id: int | None = None,
    workflow_id: str | None = None,
    model_name: str | None = None,
    run_start_time: datetime | None = None,
    run_finish_time: datetime | None = None,
    run_params: dict | None = None,
) -> CorrectionResult:
    """Resolve a correction manifest, capture superseded versions, and write.

    The manifest is a regular write manifest (one routing column,
    ``data_type``, ``name``, ``valid_time``, ``value``, optional ``unit``)
    minus every audit column, which this entry point stamps itself. One
    ``knowledge_time`` governs the whole correction run: it is the moment
    the correction enters the knowledge window, shared by the ClickHouse
    rows and the ``runs`` record so the two stores tell one story.

    Pipeline: resolve (PG) → capture superseded versions (CH) → validate
    targets and the knowledge_time guard → upsert the run with the frozen
    audit block (PG) → insert the correction rows (CH). Every validation
    raise happens before the run upsert, so a rejected correction records
    no run; a CH failure after the upsert leaves an orphaned runs row
    findable by ``run_id``, the same contract as ``write()``.
    """
    _validate_correction_inputs(df, reason=reason, changed_by=changed_by, knowledge_time=knowledge_time)
    kt = knowledge_time or datetime.now(UTC)
    rid = run_id if run_id is not None else runs_mod.generate_run_id()

    # Normalize to the storage time unit so the superseded join and the CH
    # insert agree on dtype (a nanosecond manifest would otherwise anti-join
    # every row as "missing").
    if isinstance(df.schema["valid_time"], pl.Datetime):
        df = df.with_columns(pl.col("valid_time").cast(pl.Datetime("us", "UTC")))

    # Raw checkout mirroring write_manifest's autocommit fast path: the
    # namespace GUC binds at session level and is cleared before the
    # connection returns to the shared pool.
    async with annotate_undefined_table(), client._pool.connection() as conn:
        await conn.set_autocommit(True)
        if client._namespace is not None:
            await conn.execute(
                "SELECT set_config('energydb.namespace', %s, false)",
                (client._namespace,),
            )
        try:
            with profiling._phase(profiling.PHASE_EDB_RESOLVE):
                resolved, summary = await resolve_manifest(conn, df, attach_path=False)

            if summary.has_overlapping:
                raise ValidationError(
                    f"correct() is for FLAT series (actuals); {len(summary.overlapping_series_ids)} "
                    "resolved series are OVERLAPPING. Those are versioned by republication: "
                    "write a new knowledge_time version with write() instead."
                )
            if resolved.select("series_id", "valid_time").is_duplicated().any():
                raise ValidationError(
                    "Correction manifest contains duplicate (series, valid_time) rows; "
                    "one correction per reading per call."
                )

            if "unit" in resolved.columns:
                with profiling._phase(profiling.PHASE_EDB_UNIT_CONVERT):
                    resolved = apply_manifest_unit_conversion(resolved)

            keys = resolved.select("series_id", "valid_time", "retention").unique()
            superseded = await asyncio.to_thread(_fetch_latest_versions, td._ch, keys)
            _require_all_targets_published(resolved, superseded)
            _require_correction_wins(superseded, kt=kt)

            identity_cols = [c for c in _AUDIT_IDENTITY_COLS if c in resolved.columns]
            audit = superseded.join(
                resolved.select([*identity_cols, "series_id", "valid_time"]).unique(),
                on=["series_id", "valid_time"],
                how="left",
            ).sort([*identity_cols, "valid_time"])

            run = runs_mod.RunRow(
                run_id=rid,
                workflow_id=workflow_id,
                model_name=model_name,
                run_start_time=run_start_time or datetime.now(UTC),
                run_finish_time=run_finish_time,
                run_params={
                    **(run_params or {}),
                    "correction": _correction_run_params(audit, reason=reason, changed_by=changed_by, kt=kt),
                },
            )
            await runs_mod.upsert_run_row(conn, run)
        finally:
            if client._namespace is not None:
                await conn.execute("SELECT set_config('energydb.namespace', '', false)")
            await conn.set_autocommit(False)

    with profiling._phase(profiling.PHASE_EDB_MANIFEST_BUILD):
        keep = [c for c in resolved.columns if c not in _ROUTING_AND_META_COLS]
        write_df = resolved.select(keep).with_columns(
            pl.lit(rid, dtype=pl.Int64).alias("run_id"),
            pl.lit(changed_by, dtype=pl.Utf8).alias("changed_by"),
            pl.lit(reason, dtype=pl.Utf8).alias("annotation"),
        )

    # timedb is synchronous, hence the worker thread (same as write_manifest).
    counts = await asyncio.to_thread(td.write, write_df, knowledge_time=kt)

    superseded_out = audit.select(
        [*identity_cols, "valid_time", "value", "knowledge_time", "change_time", "changed_by", "annotation", "run_id"]
    )
    return CorrectionResult(rid, counts.written, counts.skipped, superseded_out)

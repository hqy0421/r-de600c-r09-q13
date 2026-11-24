"""Integration tests for corrections of published actuals.

``scope.correct`` / ``client.correct`` are the audited revision entry point:
the correction becomes the latest readable version, the superseded version
stays in storage, and reads bounded by ``end_known`` reproduce the
pre-revision report. Skipped if TIMEDB_PG_DSN or TIMEDB_CH_URL are not set.
"""

import os
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

import energydb as edb
from energydb import Client, CorrectionResult, CorrectionTargetError, ValidationError

if not (os.environ.get("TIMEDB_PG_DSN") and os.environ.get("TIMEDB_CH_URL")):
    pytest.skip(
        "TIMEDB_PG_DSN / TIMEDB_CH_URL not set: skipping integration tests",
        allow_module_level=True,
    )


BASE_VT = datetime(2026, 1, 1, tzinfo=UTC)
KT_1 = datetime(2026, 1, 2, 9, tzinfo=UTC)  # original publish
KT_2 = datetime(2026, 1, 3, 9, tzinfo=UTC)  # correction
REASON = "Meter recalibration after CT ratio audit (QE-114)"
AUTHOR = "qa:jane.doe"


def _ts_df(n: int = 4, *, base: float = 0.0) -> pl.DataFrame:
    times = pl.datetime_range(
        start=BASE_VT,
        end=BASE_VT + timedelta(hours=n - 1),
        interval="1h",
        time_unit="us",
        time_zone="UTC",
        eager=True,
    )
    return pl.DataFrame({"valid_time": times, "value": [base + float(i) for i in range(n)]})


@pytest.fixture
def edb_client():
    client = Client()
    client.delete()
    client.create()
    asset = edb.wind.WindTurbine(
        name="T01",
        capacity=1.0,
        timeseries=[
            edb.TimeSeries(name="power", unit="MW", data_type=edb.DataType.ACTUAL),
            edb.TimeSeries(
                name="power_fc",
                unit="MW",
                data_type=edb.DataType.FORECAST,
                timeseries_type=edb.TimeSeriesType.OVERLAPPING,
            ),
        ],
    )
    client.register_tree(edb.Portfolio(name="root", members=[asset]))
    # The published actuals every correction below starts from. Values are
    # non-zero: timedb 0.12.0's include_updates read drops a partition's first
    # row when its (value, annotation, changed_by) equals the type default
    # (0.0, '', ''), an upstream quirk these tests must not depend on.
    client.get_node("root", "T01").write(
        _ts_df(4, base=1.0), data_type="actual", name="power", knowledge_time=KT_1
    )
    yield client
    client.delete()
    client.close()


def _correct_one(client, **overrides):
    """Correct hour 1 (value 2.0 → 1.5) with the standard audit payload."""
    kwargs = {
        "data_type": "actual",
        "name": "power",
        "reason": REASON,
        "changed_by": AUTHOR,
        "knowledge_time": KT_2,
    }
    kwargs.update(overrides)
    return client.get_node("root", "T01").correct(
        pl.DataFrame({"valid_time": [BASE_VT + timedelta(hours=1)], "value": [1.5]}),
        **kwargs,
    )


def test_correction_lands_in_latest_and_history_reproduces(edb_client):
    _correct_one(edb_client)

    scope = edb_client.get_node("root", "T01")

    # Latest results include the correction.
    latest = scope.read(data_type="actual", name="power")
    assert latest["value"].to_list() == [1.0, 1.5, 3.0, 4.0]

    # A report generated before the correction reproduces under the
    # pre-revision basis: end_known excludes everything known at/after KT_2.
    before = scope.read(data_type="actual", name="power", end_known=KT_2)
    assert before["value"].to_list() == [1.0, 2.0, 3.0, 4.0]

    # The knowledge window selects versions on both sides of the correction.
    after = scope.read(data_type="actual", name="power", start_known=KT_2)
    assert after["valid_time"].to_list() == [BASE_VT + timedelta(hours=1)]
    assert after["value"].to_list() == [1.5]


def test_correction_result_carries_superseded_identity(edb_client):
    result = _correct_one(edb_client)

    assert isinstance(result, CorrectionResult)
    assert isinstance(int(result), int) and result.written == 1

    superseded = result.superseded
    assert superseded.height == 1
    row = superseded.row(0, named=True)
    # The original record's identity, captured from storage.
    assert row["path"] == "root/T01"
    assert row["data_type"] == "actual"
    assert row["name"] == "power"
    assert row["valid_time"] == BASE_VT + timedelta(hours=1)
    assert row["value"] == 2.0
    assert row["knowledge_time"] == KT_1


def test_correction_audit_columns_on_read(edb_client):
    _correct_one(edb_client)

    audit = edb_client.get_node("root", "T01").read(
        data_type="actual",
        name="power",
        include_updates=True,
        include_knowledge_time=True,
    )
    corrected = audit.filter(pl.col("knowledge_time") == KT_2)
    assert corrected.height == 1
    row = corrected.row(0, named=True)
    assert row["value"] == 1.5
    assert row["annotation"] == REASON
    assert row["changed_by"] == AUTHOR

    # The superseded version remains part of the chain.
    original = audit.filter(pl.col("knowledge_time") == KT_1)
    assert original.height == 4


def test_correction_run_params_freeze_reason_and_superseded(edb_client):
    result = _correct_one(edb_client, workflow_id="qe-114")

    node_uuid = edb_client.get_node("root", "T01").get_raw()["uuid"]
    series_id = next(
        s["series_id"] for s in edb_client.list_series(node_uuid) if s["name"] == "power"
    )
    runs = edb_client.read_runs_for_series(series_id=series_id)
    correction_run = next(r for r in runs if r["run_id"] == int(result))

    assert correction_run["workflow_id"] == "qe-114"
    block = correction_run["run_params"]["correction"]
    assert block["reason"] == REASON
    assert block["changed_by"] == AUTHOR
    assert block["knowledge_time"] == KT_2.isoformat()
    assert block["superseded_truncated"] is False
    assert len(block["superseded"]) == 1
    entry = block["superseded"][0]
    assert entry["series_id"] == series_id
    assert entry["data_type"] == "actual"
    assert entry["name"] == "power"
    assert entry["valid_time"] == (BASE_VT + timedelta(hours=1)).isoformat()
    assert entry["knowledge_time"] == KT_1.isoformat()
    assert entry["value"] == 2.0


def test_correct_unpublished_reading_raises(edb_client):
    with pytest.raises(CorrectionTargetError) as exc_info:
        edb_client.get_node("root", "T01").correct(
            pl.DataFrame(
                {
                    "valid_time": [BASE_VT + timedelta(days=10)],
                    "value": [9.9],
                }
            ),
            data_type="actual",
            name="power",
            reason=REASON,
            changed_by=AUTHOR,
        )
    missing = exc_info.value.missing
    assert missing is not None and len(missing) == 1
    # (path, data_type, name, valid_time) — routing identity plus the reading.
    assert missing[0][:3] == ("root/T01", "actual", "power")

    # A failed correction writes nothing and records no extra version.
    latest = edb_client.get_node("root", "T01").read(data_type="actual", name="power")
    assert latest["value"].to_list() == [1.0, 2.0, 3.0, 4.0]


def test_correct_overlapping_series_raises(edb_client):
    edb_client.get_node("root", "T01").write(
        _ts_df(2), data_type="forecast", name="power_fc", knowledge_time=KT_1
    )
    with pytest.raises(ValidationError, match="FLAT"):
        edb_client.get_node("root", "T01").correct(
            pl.DataFrame({"valid_time": [BASE_VT], "value": [5.0]}),
            data_type="forecast",
            name="power_fc",
            reason=REASON,
            changed_by=AUTHOR,
            knowledge_time=KT_2,
        )


def test_correct_requires_reason_and_author(edb_client):
    df = pl.DataFrame({"valid_time": [BASE_VT], "value": [5.0]})
    scope = edb_client.get_node("root", "T01")
    with pytest.raises(ValidationError, match="reason"):
        scope.correct(df, data_type="actual", name="power", reason="  ", changed_by=AUTHOR)
    with pytest.raises(ValidationError, match="changed_by"):
        scope.correct(df, data_type="actual", name="power", reason=REASON, changed_by="")


def test_correct_rejects_stale_knowledge_time(edb_client):
    # A correction stamped at/before the superseded version's knowledge_time
    # would never surface in latest reads.
    with pytest.raises(ValidationError, match="knowledge_time"):
        _correct_one(edb_client, knowledge_time=KT_1)
    with pytest.raises(ValidationError, match="timezone-aware"):
        _correct_one(edb_client, knowledge_time=datetime(2026, 1, 3, 9))


def test_correct_with_unit_conversion(edb_client):
    # Canonical unit is MW; the correction arrives in kW.
    _correct_one(edb_client, unit="kW")
    # 1.5 kW = 0.0015 MW
    latest = edb_client.get_node("root", "T01").read(data_type="actual", name="power")
    assert latest["value"].to_list()[1] == pytest.approx(0.0015)


def test_client_correct_bulk_manifest(edb_client):
    manifest = pl.DataFrame(
        {
            "path": ["root/T01", "root/T01"],
            "data_type": ["actual", "actual"],
            "name": ["power", "power"],
            "valid_time": [BASE_VT + timedelta(hours=2), BASE_VT + timedelta(hours=3)],
            "value": [20.5, 30.5],
        }
    )
    result = edb_client.correct(
        manifest,
        reason=REASON,
        changed_by=AUTHOR,
        knowledge_time=KT_2,
    )
    assert result.written == 2
    assert result.superseded.height == 2
    assert result.superseded["value"].to_list() == [3.0, 4.0]

    latest = edb_client.get_node("root", "T01").read(data_type="actual", name="power")
    assert latest["value"].to_list() == [1.0, 2.0, 20.5, 30.5]
    before = edb_client.get_node("root", "T01").read(
        data_type="actual", name="power", end_known=KT_2
    )
    assert before["value"].to_list() == [1.0, 2.0, 3.0, 4.0]


def test_correct_rejects_duplicate_rows_and_owned_columns(edb_client):
    vt = BASE_VT + timedelta(hours=1)
    scope = edb_client.get_node("root", "T01")

    dup = pl.DataFrame({"valid_time": [vt, vt], "value": [1.5, 1.6]})
    with pytest.raises(ValidationError, match="duplicate"):
        scope.correct(
            dup,
            data_type="actual",
            name="power",
            reason=REASON,
            changed_by=AUTHOR,
            knowledge_time=KT_2,
        )

    # Audit columns are stamped by correct() itself, never taken from the df.
    with pytest.raises(ValidationError, match="annotation"):
        scope.correct(
            pl.DataFrame({"valid_time": [vt], "value": [1.5], "annotation": ["sneaky"]}),
            data_type="actual",
            name="power",
            reason=REASON,
            changed_by=AUTHOR,
            knowledge_time=KT_2,
        )


def test_correct_edge_series(edb_client):
    """The same entry point serves edge-owned series (EdgeScope shares _BaseScope)."""
    from energydatamodel.reference import Reference

    bus_a = edb.grid.JunctionPoint(name="BusA")
    bus_b = edb.grid.JunctionPoint(name="BusB")
    edb_client.register_tree(edb.Portfolio(name="Grid", members=[bus_a, bus_b]))
    edb_client.create_edge(
        edb.grid.Line(name="L1", from_element=Reference(bus_a), to_element=Reference(bus_b))
    )
    scope = edb_client.get_edge("Grid/BusA", "Grid/BusB", type="Line", name="L1")
    scope.register_series(name="flow", canonical_unit="MW", data_type="actual", timeseries_type="FLAT")
    scope.write(_ts_df(2, base=1.0), data_type="actual", name="flow", knowledge_time=KT_1)

    result = scope.correct(
        pl.DataFrame({"valid_time": [BASE_VT], "value": [9.5]}),
        data_type="actual",
        name="flow",
        reason=REASON,
        changed_by=AUTHOR,
        knowledge_time=KT_2,
    )
    assert result.written == 1
    # EdgeScope routes by edge_uuid, so that is the identity the result carries.
    edge_uuid = scope.get_raw()["uuid"]
    assert result.superseded.row(0, named=True)["edge_uuid"] == str(edge_uuid)

    latest = scope.read(data_type="actual", name="flow")
    assert latest["value"].to_list() == [9.5, 2.0]
    before = scope.read(data_type="actual", name="flow", end_known=KT_2)
    assert before["value"].to_list() == [1.0, 2.0]

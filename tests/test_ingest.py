"""Tests for stage 1, ingest."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from fraud_pipeline import ingest
from fraud_pipeline.ingest import IngestError

from .synthetic import make_transactions, write_raw_csv


def test_missing_file_explains_how_to_get_it(sandbox_config) -> None:
    with pytest.raises(IngestError) as error:
        ingest.resolve_raw_file(sandbox_config)

    message = str(error.value)
    assert "creditcard.csv" in message
    assert "kaggle.com" in message
    assert "data/raw/README.md" in message


def test_empty_file_is_rejected(sandbox_config) -> None:
    from fraud_pipeline.paths import ensure_dir

    target = ensure_dir(sandbox_config.paths.raw()) / sandbox_config.dataset.spec().raw_file
    target.write_text("", encoding="utf-8")

    with pytest.raises(IngestError, match="empty"):
        ingest.resolve_raw_file(sandbox_config)


def test_header_only_file_is_rejected(sandbox_config) -> None:
    frame = make_transactions(sandbox_config, rows=10)
    write_raw_csv(sandbox_config, frame.iloc[0:0])

    with pytest.raises(IngestError, match="no rows"):
        ingest.run(sandbox_config)


def test_run_writes_parquet_and_manifest(sandbox_config) -> None:
    frame = make_transactions(sandbox_config, rows=500, fraud_rows=5)
    write_raw_csv(sandbox_config, frame)

    manifest = ingest.run(sandbox_config)

    interim = sandbox_config.paths.interim() / sandbox_config.ingest.interim_file
    manifest_file = sandbox_config.paths.interim() / sandbox_config.ingest.manifest_file

    assert interim.is_file()
    assert manifest_file.is_file()
    assert manifest.rows == 500
    assert manifest.fraud_rows == 5
    assert manifest.fraud_rate == pytest.approx(0.01)
    assert manifest.columns == 31


def test_manifest_records_a_checksum(sandbox_config) -> None:
    write_raw_csv(sandbox_config, make_transactions(sandbox_config, rows=300))
    manifest = ingest.run(sandbox_config)

    assert manifest.source_sha256 is not None
    assert len(manifest.source_sha256) == 64

    stored = json.loads(
        (sandbox_config.paths.interim() / sandbox_config.ingest.manifest_file).read_text(
            encoding="utf-8"
        )
    )
    assert stored["source_sha256"] == manifest.source_sha256


def test_checksum_can_be_switched_off(config_path, tmp_path) -> None:
    from fraud_pipeline.config import load_config

    cfg = load_config(
        config_path,
        overrides=[
            f"paths.data_raw={(tmp_path / 'raw').as_posix()}",
            f"paths.data_interim={(tmp_path / 'interim').as_posix()}",
            "ingest.checksum=false",
        ],
    )
    write_raw_csv(cfg, make_transactions(cfg, rows=200))

    assert ingest.run(cfg).source_sha256 is None


def test_the_same_file_always_hashes_the_same(sandbox_config) -> None:
    write_raw_csv(sandbox_config, make_transactions(sandbox_config, rows=200))
    path = sandbox_config.paths.raw() / sandbox_config.dataset.spec().raw_file

    assert ingest.file_sha256(path) == ingest.file_sha256(path)


def test_a_changed_file_hashes_differently(sandbox_config) -> None:
    path = sandbox_config.paths.raw() / sandbox_config.dataset.spec().raw_file

    write_raw_csv(sandbox_config, make_transactions(sandbox_config, rows=200, seed=1))
    first = ingest.file_sha256(path)

    write_raw_csv(sandbox_config, make_transactions(sandbox_config, rows=200, seed=2))
    assert ingest.file_sha256(path) != first


def test_chunked_and_single_read_agree(sandbox_config, config_path, tmp_path) -> None:
    """Reading in chunks must not change a single value."""
    from fraud_pipeline.config import load_config

    frame = make_transactions(sandbox_config, rows=1000, fraud_rows=10)
    write_raw_csv(sandbox_config, frame)
    path = sandbox_config.paths.raw() / sandbox_config.dataset.spec().raw_file

    chunked_config = load_config(
        config_path,
        overrides=[
            f"paths.data_raw={(tmp_path / 'raw').as_posix()}",
            "ingest.chunk_size=137",
        ],
    )

    single = ingest.read_raw(path, sandbox_config)
    chunked = ingest.read_raw(path, chunked_config)

    pd.testing.assert_frame_equal(single, chunked)


def test_missing_column_is_rejected(sandbox_config) -> None:
    frame = make_transactions(sandbox_config, rows=200).drop(columns=["V5"])
    write_raw_csv(sandbox_config, frame)

    with pytest.raises(IngestError, match="missing columns"):
        ingest.run(sandbox_config)


def test_unexpected_column_is_rejected(sandbox_config) -> None:
    frame = make_transactions(sandbox_config, rows=200)
    frame["Surprise"] = 1.0
    write_raw_csv(sandbox_config, frame)

    with pytest.raises(IngestError, match="unexpected columns"):
        ingest.run(sandbox_config)


def test_shuffled_column_order_is_repaired(sandbox_config) -> None:
    spec = sandbox_config.dataset.spec()
    frame = make_transactions(sandbox_config, rows=200)
    write_raw_csv(sandbox_config, frame[list(reversed(spec.expected_columns()))])

    ingest.run(sandbox_config)

    assert list(ingest.load_interim(sandbox_config).columns) == spec.expected_columns()


def test_non_numeric_values_are_rejected(sandbox_config) -> None:
    frame = make_transactions(sandbox_config, rows=200)
    frame["V3"] = frame["V3"].astype(str)
    frame.loc[5, "V3"] = "not a number"
    write_raw_csv(sandbox_config, frame)

    with pytest.raises(IngestError, match="expected column types"):
        ingest.run(sandbox_config)


def test_interim_round_trip_preserves_the_data(sandbox_config) -> None:
    frame = make_transactions(sandbox_config, rows=400, fraud_rows=8)
    write_raw_csv(sandbox_config, frame)
    ingest.run(sandbox_config)

    loaded = ingest.load_interim(sandbox_config)

    assert len(loaded) == len(frame)
    assert list(loaded.columns) == list(frame.columns)
    assert loaded["Class"].sum() == frame["Class"].sum()
    assert loaded["Class"].dtype == "int8"


def test_load_interim_before_ingest_explains_itself(sandbox_config) -> None:
    with pytest.raises(IngestError, match="fraud ingest"):
        ingest.load_interim(sandbox_config)


def test_load_manifest_before_ingest_explains_itself(sandbox_config) -> None:
    with pytest.raises(IngestError, match="fraud ingest"):
        ingest.load_manifest(sandbox_config)


def test_manifest_round_trips(sandbox_config) -> None:
    write_raw_csv(sandbox_config, make_transactions(sandbox_config, rows=300, fraud_rows=6))
    written = ingest.run(sandbox_config)

    assert ingest.load_manifest(sandbox_config) == written


def test_dtype_map_types_the_target_as_an_integer(config) -> None:
    spec = config.dataset.spec()
    dtypes = ingest.build_dtype_map(spec.expected_columns(), spec.target_column)

    assert dtypes["Class"] == "int8"
    assert dtypes["Amount"] == "float64"
    assert dtypes["V1"] == "float64"
    assert len(dtypes) == 31


@pytest.mark.needs_data
@pytest.mark.slow
def test_real_dataset_matches_the_published_shape(real_data_config) -> None:
    """The one test that reads the genuine file, skipped when it is not there."""
    spec = real_data_config.dataset.spec()
    path = ingest.resolve_raw_file(real_data_config)
    frame = ingest.read_raw(path, real_data_config)

    assert len(frame) == spec.expected_rows
    assert int(frame[spec.target_column].sum()) == spec.expected_fraud_rows
    assert list(frame.columns) == spec.expected_columns()

"""Stage 1. Read the raw transaction file and write a typed interim table.

This stage does as little thinking as possible. Its whole job is to turn a csv that
somebody downloaded by hand into a parquet file the rest of the pipeline can trust, and to
record exactly which file it came from.

Judgement calls about whether the data is any good belong in stage 2, not here. The only
failures raised here are the ones that make the file unreadable: it is missing, it is
empty, or its columns are not the columns we were promised.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from fraud_pipeline.config import Config
from fraud_pipeline.paths import ensure_dir

logger = logging.getLogger(__name__)

_HASH_CHUNK_BYTES = 1024 * 1024


class IngestError(RuntimeError):
    """The raw file cannot be read into the shape the pipeline expects."""


@dataclass(frozen=True)
class IngestManifest:
    """A record of what was read, so a result can be traced back to its input.

    This is written next to the interim table as json. If a metric ever looks surprising,
    the first question is which file produced it, and this answers that question.
    """

    dataset: str
    source_file: str
    source_bytes: int
    source_sha256: str | None
    rows: int
    columns: int
    fraud_rows: int
    fraud_rate: float
    interim_file: str
    ingested_at: str
    pipeline_version: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def file_sha256(path: Path, chunk_bytes: int = _HASH_CHUNK_BYTES) -> str:
    """Hash a file in chunks, so a large csv never has to be held in memory at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def build_dtype_map(expected_columns: list[str], target_column: str) -> dict[str, str]:
    """Type every column up front so pandas never has to infer.

    The target is the one column that is genuinely an integer. Everything else, including
    the seconds column, is a float. Inference would give the same answer here, but stating
    it means a malformed file fails loudly at read time instead of silently arriving as
    a column of strings.
    """
    dtypes = {name: "float64" for name in expected_columns if name != target_column}
    dtypes[target_column] = "int8"
    return dtypes


def resolve_raw_file(config: Config) -> Path:
    """Find the downloaded dataset, or explain clearly how to get it."""
    spec = config.dataset.spec()
    candidate = config.paths.raw() / spec.raw_file

    if not candidate.is_file():
        raise IngestError(
            f"raw data file not found: {candidate}\n"
            f"Download '{spec.name}' from {spec.source_url} and place "
            f"'{spec.raw_file}' in {config.paths.raw()}.\n"
            f"See data/raw/README.md for the full instructions."
        )
    if candidate.stat().st_size == 0:
        raise IngestError(f"raw data file is empty: {candidate}")

    return candidate


def read_raw(path: Path, config: Config) -> pd.DataFrame:
    """Read the csv with fixed dtypes, in chunks when the config asks for it."""
    spec = config.dataset.spec()
    dtypes = build_dtype_map(spec.expected_columns(), spec.target_column)
    chunk_size = config.ingest.chunk_size

    try:
        if chunk_size and chunk_size > 0:
            chunks = pd.read_csv(path, dtype=dtypes, chunksize=chunk_size)
            frame = pd.concat(chunks, ignore_index=True)
        else:
            frame = pd.read_csv(path, dtype=dtypes)
    except ValueError as error:
        # A dtype failure lands here, and it usually means the file is not the one we think.
        raise IngestError(
            f"could not read {path.name} with the expected column types: {error}"
        ) from error

    return frame


def check_columns(frame: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Fail if the file does not have exactly the columns the config describes.

    Returns the frame with its columns in the expected order, so no later stage has to
    care what order the csv happened to use.
    """
    spec = config.dataset.spec()
    expected = list(spec.expected_columns())
    actual = list(frame.columns)

    if actual == expected:
        return frame

    missing = [name for name in expected if name not in actual]
    unexpected = [name for name in actual if name not in expected]

    if missing or unexpected:
        parts = []
        if missing:
            parts.append(f"missing columns: {missing}")
        if unexpected:
            parts.append(f"unexpected columns: {unexpected}")
        raise IngestError("the raw file does not match the expected schema. " + "; ".join(parts))

    # Same names, different order. Harmless, so reorder and carry on.
    logger.warning("columns arrived in a different order than expected, reordering")
    return frame[expected]


def build_manifest(
    frame: pd.DataFrame,
    source: Path,
    interim: Path,
    config: Config,
    checksum: str | None,
) -> IngestManifest:
    from fraud_pipeline import __version__

    spec = config.dataset.spec()
    fraud_rows = int(frame[spec.target_column].sum())

    return IngestManifest(
        dataset=config.dataset.active,
        source_file=source.name,
        source_bytes=source.stat().st_size,
        source_sha256=checksum,
        rows=int(len(frame)),
        columns=int(frame.shape[1]),
        fraud_rows=fraud_rows,
        fraud_rate=round(fraud_rows / len(frame), 8) if len(frame) else 0.0,
        interim_file=interim.name,
        ingested_at=datetime.now(UTC).isoformat(timespec="seconds"),
        pipeline_version=__version__,
    )


def run(config: Config) -> IngestManifest:
    """Run stage 1 and return the manifest describing what was ingested."""
    source = resolve_raw_file(config)
    logger.info("reading %s (%.1f MB)", source.name, source.stat().st_size / 1024 / 1024)

    frame = read_raw(source, config)
    frame = check_columns(frame, config)

    if frame.empty:
        raise IngestError(f"{source.name} has a header but no rows")

    checksum = file_sha256(source) if config.ingest.checksum else None
    if checksum:
        logger.info("source sha256 %s", checksum[:16])

    interim_dir = ensure_dir(config.paths.interim())
    interim_file = interim_dir / config.ingest.interim_file
    frame.to_parquet(interim_file, index=False)

    manifest = build_manifest(frame, source, interim_file, config, checksum)
    manifest_file = interim_dir / config.ingest.manifest_file
    manifest_file.write_text(manifest.to_json(), encoding="utf-8")

    logger.info(
        "ingested %s rows and %s columns, %s of them fraud (%.3f percent)",
        f"{manifest.rows:,}",
        manifest.columns,
        manifest.fraud_rows,
        manifest.fraud_rate * 100,
    )
    logger.info("wrote %s", interim_file)
    logger.info("wrote %s", manifest_file)

    return manifest


def load_interim(config: Config) -> pd.DataFrame:
    """Read the table stage 1 produced. Used by every stage after this one."""
    interim_file = config.paths.interim() / config.ingest.interim_file
    if not interim_file.is_file():
        raise IngestError(
            f"interim table not found: {interim_file}\nRun the ingest stage first: fraud ingest"
        )
    return pd.read_parquet(interim_file)


def load_manifest(config: Config) -> IngestManifest:
    """Read the manifest stage 1 produced."""
    manifest_file = config.paths.interim() / config.ingest.manifest_file
    if not manifest_file.is_file():
        raise IngestError(
            f"ingest manifest not found: {manifest_file}\nRun the ingest stage first: fraud ingest"
        )
    return IngestManifest(**json.loads(manifest_file.read_text(encoding="utf-8")))

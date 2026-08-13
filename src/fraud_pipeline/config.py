"""Typed configuration loading.

``configs/config.yaml`` is the single source of truth for the pipeline. This module reads
it, validates it with pydantic, and hands every stage a typed object instead of a loose
dictionary. A typo in the config fails here with a clear message rather than three stages
later with a KeyError.

Values can be overridden without editing the file, which is what the ``--set`` flag on the
command line uses:

    from fraud_pipeline.config import load_config
    cfg = load_config(overrides=["project.seed=7", "models.xgboost.max_depth=8"])
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from fraud_pipeline.paths import default_config_path, resolve


class StrictModel(BaseModel):
    """Base class that rejects unknown keys, so a misspelled option is caught at load time."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ProjectConfig(StrictModel):
    name: str
    seed: int = 42


class PathsConfig(StrictModel):
    data_raw: str
    data_interim: str
    data_processed: str
    models: str
    reports: str
    figures: str
    tables: str

    def raw(self) -> Path:
        return resolve(self.data_raw)

    def interim(self) -> Path:
        return resolve(self.data_interim)

    def processed(self) -> Path:
        return resolve(self.data_processed)

    def model_dir(self) -> Path:
        return resolve(self.models)

    def figures_dir(self) -> Path:
        return resolve(self.figures)

    def tables_dir(self) -> Path:
        return resolve(self.tables)


class UlbDatasetSpec(StrictModel):
    name: str
    source_url: str
    raw_file: str
    target_column: str
    time_column: str
    amount_column: str
    pca_columns_prefix: str
    pca_columns_count: int
    expected_rows: int
    expected_fraud_rows: int

    def pca_columns(self) -> list[str]:
        """The anonymised component column names, V1 through V28."""
        return [f"{self.pca_columns_prefix}{i}" for i in range(1, self.pca_columns_count + 1)]

    def expected_columns(self) -> list[str]:
        return [self.time_column, *self.pca_columns(), self.amount_column, self.target_column]


class DatasetConfig(StrictModel):
    active: str
    ulb: UlbDatasetSpec

    @field_validator("active")
    @classmethod
    def _known_dataset(cls, value: str) -> str:
        allowed = {"ulb"}
        if value not in allowed:
            raise ValueError(f"dataset.active must be one of {sorted(allowed)}, got {value!r}")
        return value

    def spec(self) -> UlbDatasetSpec:
        """The spec for whichever dataset is currently active."""
        return getattr(self, self.active)


class IngestConfig(StrictModel):
    interim_file: str
    manifest_file: str
    checksum: bool = True
    chunk_size: int = 0


class SplitConfig(StrictModel):
    strategy: str
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    fallback_strategy: str = "stratified"
    output_dir: str = "splits"

    @field_validator("strategy", "fallback_strategy")
    @classmethod
    def _known_strategy(cls, value: str) -> str:
        allowed = {"time", "stratified"}
        if value not in allowed:
            raise ValueError(f"split strategy must be one of {sorted(allowed)}, got {value!r}")
        return value

    @model_validator(mode="after")
    def _fractions_sum_to_one(self) -> SplitConfig:
        total = self.train_fraction + self.validation_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split fractions must sum to 1.0, they sum to {total}")
        return self


class ValidationConfig(StrictModel):
    max_missing_fraction: float
    min_rows: int
    allowed_target_values: list[int]
    max_fraud_rate: float
    min_fraud_rate: float
    amount_min: float
    fail_on_duplicate_rows: bool = False
    drop_duplicates: bool = True
    min_fraud_rows_per_split: int = 30
    report_file: str = "validation_report.md"


class FeaturesConfig(StrictModel):
    amount_log: bool
    amount_zscore: bool
    hour_of_day: bool
    time_since_last: bool
    rolling_windows: list[int]
    scale_pca_columns: bool


class SmoteConfig(StrictModel):
    sampling_strategy: float
    k_neighbors: int


class ImbalanceConfig(StrictModel):
    strategies: list[str]
    smote: SmoteConfig

    @field_validator("strategies")
    @classmethod
    def _known_strategies(cls, value: list[str]) -> list[str]:
        allowed = {"none", "class_weight", "smote"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown imbalance strategies: {sorted(unknown)}")
        return value


class ModelSpec(BaseModel):
    """One model entry. Unknown keys are kept and passed to the estimator as hyperparameters."""

    model_config = ConfigDict(extra="allow", frozen=True)

    enabled: bool = True

    def params(self) -> dict[str, Any]:
        """Every key except ``enabled``, ready to hand to the estimator constructor."""
        return dict(self.model_extra or {})


class CostConfig(StrictModel):
    false_negative_cost: float
    false_positive_cost: float


class EvaluationConfig(StrictModel):
    primary_metric: str
    report_metrics: list[str]
    threshold_strategy: str
    fixed_recall_target: float
    cost: CostConfig

    @field_validator("threshold_strategy")
    @classmethod
    def _known_threshold_strategy(cls, value: str) -> str:
        allowed = {"cost", "max_f1", "fixed_recall"}
        if value not in allowed:
            raise ValueError(f"threshold_strategy must be one of {sorted(allowed)}, got {value!r}")
        return value


class ExplainabilityConfig(StrictModel):
    shap_enabled: bool
    shap_background_samples: int
    shap_top_features: int


class RegistryConfig(StrictModel):
    tracking_uri: str
    experiment_name: str
    registered_model_name: str
    promotion_metric: str
    promotion_min_improvement: float


class ServingConfig(StrictModel):
    api_host: str
    api_port: int
    streamlit_port: int
    api_base_url: str


class Config(StrictModel):
    """The whole configuration file, validated."""

    project: ProjectConfig
    paths: PathsConfig
    dataset: DatasetConfig
    ingest: IngestConfig
    split: SplitConfig
    validation: ValidationConfig
    features: FeaturesConfig
    imbalance: ImbalanceConfig
    models: dict[str, ModelSpec] = Field(default_factory=dict)
    evaluation: EvaluationConfig
    explainability: ExplainabilityConfig
    registry: RegistryConfig
    serving: ServingConfig

    def enabled_models(self) -> dict[str, ModelSpec]:
        """Only the models switched on in the config, in file order."""
        return {name: spec for name, spec in self.models.items() if spec.enabled}


def _set_by_dot_path(tree: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set ``tree["a"]["b"] = value`` given the string ``"a.b"``."""
    parts = dotted_key.split(".")
    node = tree
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            raise KeyError(f"cannot override {dotted_key!r}: {part!r} is not a section")
        node = child
    leaf = parts[-1]
    if leaf not in node:
        raise KeyError(f"cannot override {dotted_key!r}: {leaf!r} is not in the config")
    node[leaf] = value


def apply_overrides(tree: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply ``key.path=value`` strings to a raw config dictionary.

    The value is parsed as YAML, so ``0.8`` becomes a float, ``true`` becomes a bool and
    ``[1, 2]`` becomes a list. Anything unparseable stays a string.
    """
    if not overrides:
        return tree
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override {item!r} is not in key=value form")
        key, raw_value = item.split("=", 1)
        _set_by_dot_path(tree, key.strip(), yaml.safe_load(raw_value))
    return tree


def load_config(
    path: str | Path | None = None,
    overrides: list[str] | None = None,
) -> Config:
    """Read, override and validate the configuration file."""
    config_path = Path(path) if path is not None else default_config_path()
    if not config_path.is_file():
        raise FileNotFoundError(f"config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        tree = yaml.safe_load(handle) or {}

    if not isinstance(tree, dict):
        raise ValueError(f"config file must contain a mapping at the top level: {config_path}")

    return Config.model_validate(apply_overrides(tree, overrides))

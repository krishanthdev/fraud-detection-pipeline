"""Tests for stage 4, the training sweep.

Two guarantees are worth more than the rest.

The test split is never opened. Once a held out set has been used to choose between eighteen
candidates it is no longer held out, and a number from it would be the best of eighteen
guesses rather than an estimate of anything.

Nothing here picks a threshold. Every metric is threshold free and the probabilities are
written to disk, so stage 5 can tune against the cost model without refitting.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
from pydantic import ValidationError

from fraud_pipeline import train
from fraud_pipeline.config import Config, load_config
from fraud_pipeline.train import TrainingError, resolve_feature_sets

from .synthetic import make_transactions, write_raw_csv


@pytest.fixture(scope="module")
def trained_config(tmp_path_factory) -> Config:
    """Run the pipeline up to the point where training can start.

    Module scoped on purpose. Getting here means running ingest, validate, eda and features,
    and then the sweep itself fits twelve models. Rebuilding all of that for each of twenty
    tests took nearly five minutes, which is long enough that it would start getting skipped.
    Tests that need to mutate the config build their own from this one's data.
    """
    from fraud_pipeline import eda, features, ingest, validation
    from fraud_pipeline.paths import default_config_path

    tmp_path = tmp_path_factory.mktemp("training")
    config = load_config(
        default_config_path(),
        overrides=[
            f"paths.data_raw={(tmp_path / 'raw').as_posix()}",
            f"paths.data_interim={(tmp_path / 'interim').as_posix()}",
            f"paths.data_processed={(tmp_path / 'processed').as_posix()}",
            f"paths.models={(tmp_path / 'models').as_posix()}",
            f"paths.reports={(tmp_path / 'reports').as_posix()}",
            f"paths.tables={(tmp_path / 'reports' / 'tables').as_posix()}",
            f"paths.figures={(tmp_path / 'reports' / 'figures').as_posix()}",
            "validation.min_rows=100",
            "validation.min_fraud_rows_per_split=1",
            "eda.permutation_shuffles=5",
            "eda.mutual_info_sample=800",
            "eda.figures=false",
            "training.feature_sets=[all, selected]",
            "training.bootstrap_samples=40",
            "models.logistic_regression.max_iter=200",
            "models.random_forest.n_estimators=12",
            "models.random_forest.max_depth=5",
            "registry.enabled=false",
        ],
    )

    write_raw_csv(config, make_transactions(config, rows=6000, fraud_rows=45))
    ingest.run(config)
    validation.run(config)
    eda.run(config)
    features.run(config)
    return config


@pytest.fixture(scope="module")
def report(trained_config) -> train.TrainingReport:
    return train.run(trained_config)


def _config_sharing_data(base: Config, outputs, extra: list[str]) -> Config:
    """A fresh config reading the module fixture's data but writing somewhere of its own.

    The data is shared because building it is the slow part. The outputs are not, because a
    second `train.run` in another test would otherwise overwrite the shared results table and
    make an unrelated assertion fail. That happened, and it fails in the test that reads the
    table rather than the test that wrote it, which is the worst place for it to surface.
    """
    from fraud_pipeline.paths import default_config_path

    return load_config(
        default_config_path(),
        overrides=[
            f"paths.data_interim={base.paths.interim().as_posix()}",
            f"paths.data_processed={base.paths.processed().as_posix()}",
            f"paths.tables={(outputs / 'tables').as_posix()}",
            f"paths.models={(outputs / 'models').as_posix()}",
            *extra,
        ],
    )


# --------------------------------------------------------------------------------------
# The guarantees
# --------------------------------------------------------------------------------------


def test_the_config_refuses_to_score_on_test(config_tree) -> None:
    """The guard that keeps the held out set genuinely held out."""
    config_tree["training"]["eval_split"] = "test"
    with pytest.raises(ValidationError, match="must not score on the test split"):
        Config.model_validate(config_tree)


def test_scoring_happens_on_validation(report, trained_config) -> None:
    from fraud_pipeline.features import load_engineered

    validation_rows = len(load_engineered(trained_config, "validation"))
    for run in report.results:
        assert run.eval_rows == validation_rows


def test_the_test_split_is_never_read_during_training(trained_config, monkeypatch) -> None:
    """Make reading test raise, then confirm a full sweep still completes."""
    from fraud_pipeline import features as features_module

    real_loader = features_module.load_engineered

    def guarded(config, name):
        if name == "test":
            raise AssertionError("training read the test split")
        return real_loader(config, name)

    monkeypatch.setattr(train, "load_engineered", guarded)
    result = train.run(trained_config)

    assert result.results


def test_no_metric_depends_on_a_threshold(report) -> None:
    """Precision and recall at a chosen cut off belong to stage 5, not here."""
    for run in report.results:
        assert 0.0 <= run.average_precision <= 1.0
        assert 0.0 <= run.roc_auc <= 1.0
    frame = report.to_frame()
    assert "threshold" not in frame.columns
    assert "accuracy" not in frame.columns


# --------------------------------------------------------------------------------------
# The sweep
# --------------------------------------------------------------------------------------


def test_the_sweep_covers_every_combination(report, trained_config) -> None:
    from fraud_pipeline import models

    expected = (
        len(models.available_models(trained_config))
        * len(trained_config.imbalance.strategies)
        * len(trained_config.training.feature_sets)
    )
    assert len(report.results) == expected


def test_every_axis_of_the_sweep_appears(report, trained_config) -> None:
    """Derived from the registry rather than listed, so adding a model does not break it."""
    from fraud_pipeline import models

    frame = report.to_frame()

    assert set(frame["model"]) == set(models.available_models(trained_config))
    assert set(frame["imbalance"]) == set(trained_config.imbalance.strategies)
    assert set(frame["feature_set"]) == {"all", "selected"}


def test_the_sweep_now_covers_five_models(report) -> None:
    """The advanced models are in the comparison, not merely importable."""
    assert set(report.to_frame()["model"]) == {
        "logistic_regression",
        "random_forest",
        "xgboost",
        "lightgbm",
        "neural_net",
    }


def test_feature_sets_of_different_sizes_are_used(report) -> None:
    """If both sets had the same width, the comparison would be meaningless."""
    frame = report.to_frame()
    widths = frame.groupby("feature_set")["n_features"].first()

    assert widths["all"] > widths["selected"]


def _fake_run(
    name: str, ap: float, low: float, high: float, distinct: int = 900, warning: str = ""
):
    """A RunResult with only the fields the reporting logic reads."""
    return train.RunResult(
        model=name,
        label=name,
        imbalance="none",
        feature_set="all",
        feature_set_aliases="",
        n_features=10,
        average_precision=ap,
        ap_low=low,
        ap_high=high,
        roc_auc=0.9,
        brier_score=0.001,
        precision_at_50_recall=0.5,
        precision_at_80_recall=0.4,
        train_average_precision=ap,
        fit_seconds=1.0,
        eval_rows=1000,
        eval_positives=20,
        run_name=name,
        distinct_scores=distinct,
        warning=warning,
    )


def test_a_collapsed_run_cannot_win_the_comparison() -> None:
    """A model that stopped splitting must not top the table on an artifact.

    This is what happened: LightGBM with no imbalance handling emitted 110 distinct scores
    across 42,558 rows. Its metrics were not a measurement of anything.
    """
    report = train.TrainingReport(
        results=[
            _fake_run("collapsed", 0.99, 0.9, 1.0, distinct=3, warning="only 3 distinct scores"),
            _fake_run("healthy", 0.80, 0.7, 0.9),
        ]
    )

    assert report.best().model == "healthy"
    assert [r.model for r in report.degenerate()] == ["collapsed"]


def test_a_collapsed_run_is_left_out_of_the_tied_list() -> None:
    report = train.TrainingReport(
        results=[
            _fake_run("healthy", 0.80, 0.70, 0.90),
            _fake_run("also_fine", 0.78, 0.68, 0.88),
            _fake_run("collapsed", 0.79, 0.69, 0.89, distinct=2, warning="collapsed"),
        ]
    )

    tied = [r.model for r in report.indistinguishable_from_best()]

    assert "also_fine" in tied
    assert "collapsed" not in tied


def test_when_everything_collapsed_the_best_is_still_reported() -> None:
    """Returning nothing would hide the fact that the whole sweep failed."""
    report = train.TrainingReport(
        results=[
            _fake_run("a", 0.30, 0.2, 0.4, distinct=2, warning="collapsed"),
            _fake_run("b", 0.40, 0.3, 0.5, distinct=3, warning="collapsed"),
        ]
    )

    assert report.best().model == "b"


def test_a_healthy_run_carries_no_warning(report) -> None:
    for run in report.results:
        if not run.looks_degenerate:
            assert run.warning == ""
            assert run.distinct_scores > 0


def test_the_collapse_threshold_is_configurable(config_tree) -> None:
    config_tree["training"]["min_distinct_scores"] = 5
    assert Config.model_validate(config_tree).training.min_distinct_scores == 5


def _only_model(keep: str | None) -> list[str]:
    """Overrides that disable every model except one, or all of them when keep is None.

    Written against the registry rather than listed by hand. Naming two models here was what
    broke three tests when the sweep grew from two to five, and it would break them again on
    the next one.
    """
    from fraud_pipeline import models

    return [f"models.{name}.enabled=false" for name in models.REGISTRY if name != keep]


def _write_selection_payload(config: Config, sets: dict[str, list[str]]) -> None:
    """Write a selection file by hand, so a merging test does not depend on real data."""
    from fraud_pipeline.feature_selection import tagged_name
    from fraud_pipeline.features import ENGINEERED_TAG
    from fraud_pipeline.paths import ensure_dir

    path = ensure_dir(config.paths.interim()) / tagged_name(
        config.feature_selection.output_file, ENGINEERED_TAG
    )
    path.write_text(
        json.dumps({"active_set": "selected", "sets": sets}, indent=2), encoding="utf-8"
    )


def test_identical_feature_sets_are_trained_once(trained_config, tmp_path) -> None:
    """The waste this fixes.

    On the real engineered features `safe` and `all` are the same 44 columns, because the only
    tier 1 drop in the whole selection is `Time` and `Time` is not a model feature. Training
    both burned six fits out of eighteen, roughly ten minutes, and then reported identical
    numbers twice as if they were independent evidence.

    The selection file is written by hand here rather than reused from the fixture, because on
    small synthetic data the two sets do differ and the coincidence this test is about would
    not occur.
    """
    config = _config_sharing_data(
        trained_config,
        tmp_path,
        [
            f"paths.data_interim={(tmp_path / 'interim').as_posix()}",
            "training.feature_sets=[all, safe, selected]",
            "registry.enabled=false",
        ],
    )
    shared = ["V1", "V2", "V3", "amount_log"]
    _write_selection_payload(
        config, {"all": shared, "safe": list(shared), "selected": ["V1", "V2"]}
    )

    resolved = resolve_feature_sets(config)
    names = [name for name, _, _ in resolved]
    aliases = {name: alias for name, alias, _ in resolved}

    assert len(resolved) == 2, "three sets requested, two distinct"
    assert names == ["all", "selected"], "the first name requested wins"
    assert aliases["all"] == ["safe"]
    assert aliases["selected"] == []


def test_merged_sets_are_recorded_on_the_run(trained_config, tmp_path) -> None:
    """A merged run should say which other names it stands for."""
    config = _config_sharing_data(
        trained_config,
        tmp_path,
        [
            f"paths.data_interim={(tmp_path / 'interim').as_posix()}",
            "training.feature_sets=[all, safe]",
            "training.bootstrap_samples=20",
            "training.save_models=false",
            "models.logistic_regression.max_iter=150",
            *_only_model("logistic_regression"),
            "registry.enabled=false",
        ],
    )
    _write_selection_payload(
        config, {"all": ["V1", "V2", "V3"], "safe": ["V1", "V2", "V3"], "selected": ["V1"]}
    )
    result = train.run(config)

    # One model, three strategies, and one feature set after merging.
    assert len(result.results) == 3
    for run in result.results:
        assert run.feature_set == "all"
        assert run.feature_set_aliases == "safe"


def test_distinct_feature_sets_are_not_merged(trained_config, tmp_path) -> None:
    config = _config_sharing_data(
        trained_config,
        tmp_path,
        [
            f"paths.data_interim={(tmp_path / 'interim').as_posix()}",
            "training.feature_sets=[all, selected]",
            "registry.enabled=false",
        ],
    )
    _write_selection_payload(config, {"all": ["V1", "V2", "V3"], "selected": ["V1", "V2"]})

    resolved = resolve_feature_sets(config)

    assert len(resolved) == 2
    assert all(not aliases for _, aliases, _ in resolved)


def test_order_of_features_matters_for_merging(trained_config, tmp_path) -> None:
    """Two sets with the same features in a different order are not merged.

    Deliberate. Column order changes the matrix a model is handed, and for a random forest
    that changes which of two equally good splits gets picked. Treating them as identical
    would silently drop a run that produces different numbers.
    """
    config = _config_sharing_data(
        trained_config,
        tmp_path,
        [
            f"paths.data_interim={(tmp_path / 'interim').as_posix()}",
            "training.feature_sets=[all, safe]",
            "registry.enabled=false",
        ],
    )
    _write_selection_payload(config, {"all": ["V1", "V2"], "safe": ["V2", "V1"]})

    assert len(resolve_feature_sets(config)) == 2


def test_run_names_are_unique(report) -> None:
    names = [run.run_name for run in report.results]
    assert len(names) == len(set(names))


def test_the_best_run_is_the_highest_scoring(report) -> None:
    best = report.best()
    assert best.average_precision == max(r.average_precision for r in report.results)


def test_ranking_is_ordered(report) -> None:
    scores = [run.average_precision for run in report.ranked()]
    assert scores == sorted(scores, reverse=True)


def test_the_best_model_actually_learns(report) -> None:
    """A sanity floor on the winner, not on every run."""
    assert report.best().roc_auc > 0.75


def test_imbalance_handling_beats_chance(report) -> None:
    """The runs that are given a way to care about the rare class should find it.

    Deliberately not asserted for the `none` strategy. An unweighted model on a handful of
    positives is expected to do badly, which is the whole reason that row is in the
    comparison. An earlier version of this test required every run to clear 0.6 and failed on
    exactly that row, which was the test being wrong rather than the model.
    """
    handled = [r for r in report.results if r.imbalance != "none"]
    assert handled

    for run in handled:
        assert run.roc_auc > 0.6, f"{run.run_name} scored {run.roc_auc:.3f}"


def test_no_run_is_worse_than_guessing(report) -> None:
    """Even the unweighted baseline should not rank fraud below normal transactions."""
    for run in report.results:
        assert run.roc_auc > 0.5, f"{run.run_name} scored {run.roc_auc:.3f}"


# --------------------------------------------------------------------------------------
# Uncertainty
# --------------------------------------------------------------------------------------


def test_every_run_carries_an_interval(report) -> None:
    for run in report.results:
        assert run.ap_low <= run.average_precision <= run.ap_high


def test_overlapping_runs_are_reported_as_indistinguishable(report) -> None:
    tied = report.indistinguishable_from_best()
    best = report.best()

    assert best not in tied
    for run in tied:
        assert run.ap_low <= best.ap_high and best.ap_low <= run.ap_high


def test_the_summary_groups_by_each_axis(report) -> None:
    for key in ("model", "imbalance", "feature_set"):
        summary = train.summarise_by(report, key)
        assert not summary.empty
        assert "mean" in summary.columns


def test_summarising_by_an_unknown_axis_is_rejected(report) -> None:
    with pytest.raises(TrainingError, match="unknown grouping"):
        train.summarise_by(report, "nonsense")


# --------------------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------------------


def test_the_results_table_is_written(report, trained_config) -> None:
    tables = trained_config.paths.tables_dir()
    markdown = (tables / trained_config.training.results_file).read_text(encoding="utf-8")

    assert "# Baseline model results" in markdown
    assert "PR AUC" in markdown
    assert "Accuracy is not reported" in markdown
    assert (tables / trained_config.training.results_csv).is_file()


def test_the_results_csv_round_trips(report, trained_config) -> None:
    frame = train.load_results(trained_config)
    assert len(frame) == len(report.results)
    assert set(frame["run_name"]) == {r.run_name for r in report.results}


def test_predictions_are_saved_for_every_run(report, trained_config) -> None:
    """Stage 5 tunes a threshold from these rather than refitting eighteen models."""
    for run in report.results:
        predictions = train.load_predictions(trained_config, run.run_name)

        assert len(predictions) == run.eval_rows
        assert predictions["probability"].between(0, 1).all()
        assert int(predictions["Class"].sum()) == run.eval_positives


def test_saved_models_can_be_loaded_and_used(report, trained_config) -> None:
    from fraud_pipeline.feature_selection import load_selected_features
    from fraud_pipeline.features import ENGINEERED_TAG, load_engineered

    run = report.best()
    pipeline = train.load_model(trained_config, run.run_name)

    names = load_selected_features(trained_config, run.feature_set, tag=ENGINEERED_TAG)
    validation = load_engineered(trained_config, "validation")
    probabilities = pipeline.predict_proba(validation[names].to_numpy(dtype=float))[:, 1]

    assert len(probabilities) == run.eval_rows


def test_a_reloaded_model_reproduces_its_saved_predictions(report, trained_config) -> None:
    """The saved model and the saved predictions have to agree, or one of them is stale."""
    from fraud_pipeline.feature_selection import load_selected_features
    from fraud_pipeline.features import ENGINEERED_TAG, load_engineered

    run = report.best()
    pipeline = train.load_model(trained_config, run.run_name)
    names = load_selected_features(trained_config, run.feature_set, tag=ENGINEERED_TAG)
    validation = load_engineered(trained_config, "validation")

    fresh = pipeline.predict_proba(validation[names].to_numpy(dtype=float))[:, 1]
    saved = train.load_predictions(trained_config, run.run_name)["probability"].to_numpy()

    assert fresh == pytest.approx(saved, abs=1e-12)


def test_model_saving_can_be_switched_off(trained_config, tmp_path) -> None:
    # A fresh models directory, so a file left by the shared fixture cannot make this pass
    # for the wrong reason.
    config = _config_sharing_data(
        trained_config,
        tmp_path,
        [
            "training.feature_sets=[selected]",
            "training.save_models=false",
            "training.bootstrap_samples=30",
            "models.logistic_regression.max_iter=150",
            "models.random_forest.enabled=false",
            "registry.enabled=false",
        ],
    )
    result = train.run(config)

    with pytest.raises(TrainingError, match="saved model not found"):
        train.load_model(config, result.results[0].run_name)


def test_the_json_summary_answers_the_sweep_questions(report, trained_config) -> None:
    path = train.write_json_summary(report, trained_config)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["runs"] == len(report.results)
    assert payload["eval_split"] == "validation"
    assert set(payload["by_imbalance"]) == {"none", "class_weight", "smote"}
    assert set(payload["by_feature_set"]) == {"all", "selected"}


# --------------------------------------------------------------------------------------
# Reproducibility and failure modes
# --------------------------------------------------------------------------------------


def test_the_sweep_is_reproducible(trained_config) -> None:
    first = train.run(trained_config)
    second = train.run(trained_config)

    for a, b in zip(first.ranked(), second.ranked(), strict=True):
        assert a.run_name == b.run_name
        assert a.average_precision == pytest.approx(b.average_precision)
        assert a.ap_low == pytest.approx(b.ap_low)


def test_training_needs_the_engineered_tables(config_path, tmp_path) -> None:
    from fraud_pipeline.features import FeatureError

    config = load_config(
        config_path,
        overrides=[f"paths.data_processed={(tmp_path / 'empty').as_posix()}"],
    )
    with pytest.raises(FeatureError, match="fraud features"):
        train.run(config)


def test_a_missing_feature_column_is_reported_clearly(trained_config) -> None:
    from fraud_pipeline import models
    from fraud_pipeline.features import load_engineered

    train_frame = load_engineered(trained_config, "train")
    evaluation = load_engineered(trained_config, "validation")

    with pytest.raises(TrainingError, match="features missing"):
        train.fit_one(
            models.REGISTRY["random_forest"],
            "none",
            "selected",
            ["not_a_real_column"],
            train_frame,
            evaluation,
            trained_config,
        )


def test_training_fails_clearly_when_no_model_is_available(trained_config, tmp_path) -> None:
    config = _config_sharing_data(
        trained_config,
        tmp_path,
        [*_only_model(None), "registry.enabled=false"],
    )
    with pytest.raises(TrainingError, match="no models"):
        train.run(config)


def test_unknown_feature_sets_are_rejected_by_the_config(config_tree) -> None:
    config_tree["training"]["feature_sets"] = ["all", "imaginary"]
    with pytest.raises(ValidationError, match="unknown feature sets"):
        Config.model_validate(config_tree)


def test_an_empty_feature_set_list_is_rejected(config_tree) -> None:
    config_tree["training"]["feature_sets"] = []
    with pytest.raises(ValidationError, match="at least one feature set"):
        Config.model_validate(config_tree)


# --------------------------------------------------------------------------------------
# MLflow
# --------------------------------------------------------------------------------------


def test_a_relative_sqlite_uri_is_anchored_at_the_repo_root(config_path) -> None:
    """Otherwise running from a different folder starts a second, empty experiment store."""
    from fraud_pipeline.paths import repo_root

    config = load_config(config_path, overrides=["registry.tracking_uri=sqlite:///mlflow.db"])
    resolved = train.resolve_tracking_uri(config)

    assert resolved.startswith("sqlite:///")
    assert repo_root().as_posix() in resolved


def test_a_non_sqlite_uri_is_left_alone(config_path) -> None:
    config = load_config(config_path, overrides=["registry.tracking_uri=http://localhost:5000"])
    assert train.resolve_tracking_uri(config) == "http://localhost:5000"


def test_tracking_can_be_switched_off(config_path) -> None:
    config = load_config(config_path, overrides=["registry.enabled=false"])
    tracker = train.build_tracker(config)

    # The no op tracker has the same shape, so the loop never branches on it.
    tracker.start("a-run")
    tracker.log({"a": 1}, {"b": 2.0})
    assert tracker.finish() is None


def test_a_broken_tracker_does_not_lose_the_sweep(config_path, monkeypatch) -> None:
    """Tracking is a convenience. The results table on disk is the deliverable."""
    config = load_config(config_path, overrides=["registry.tracking_uri=sqlite:///mlflow.db"])

    def explode(_config):
        raise RuntimeError("no database today")

    monkeypatch.setattr(train, "_MlflowTracker", explode)
    tracker = train.build_tracker(config)

    assert tracker.finish() is None


def test_predictions_missing_from_disk_is_reported_clearly(trained_config) -> None:
    with pytest.raises(TrainingError, match="predictions not found"):
        train.load_predictions(trained_config, "not__a__run")


def test_results_missing_from_disk_is_reported_clearly(config_path, tmp_path) -> None:
    config = load_config(
        config_path, overrides=[f"paths.tables={(tmp_path / 'nothing').as_posix()}"]
    )
    with pytest.raises(TrainingError, match="fraud train"):
        train.load_results(config)


def test_the_report_frame_has_one_row_per_run(report) -> None:
    frame = report.to_frame()
    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == len(report.results)
    assert "overfit_gap" not in frame.columns  # a property, not a stored field

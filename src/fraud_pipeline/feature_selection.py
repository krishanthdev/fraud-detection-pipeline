"""Automatic feature selection driven by the exploratory analysis.

Feature selection usually happens as a paragraph in a notebook saying which columns somebody
decided to drop. This does it as rules with measured thresholds, so the same input always
produces the same feature set and every drop comes with the evidence attached.

Rules come in two tiers, and the split between them is the important part.

**Tier 1 is correctness.** A leaking feature, a duplicate of another feature, a column that
is the same value all the way down, or a feature whose values in the next time period fall
outside anything seen in training. Keeping any of these is a mistake rather than a
preference, so tier 1 always applies.

**Tier 2 is judgement.** Dropping features that carry no measurable signal on their own.
This one is arguable, because univariate screening cannot see interactions: a feature that
is useless alone can matter in combination with another, and tree models tolerate useless
features fairly well. So it is separated out, it can be switched off, and the pipeline can
produce all three feature sets so the argument gets settled with numbers instead of
opinions.

The three sets:

``all``
    Every feature. No selection at all, the honest baseline.
``safe``
    Tier 1 applied. The set you would defend in a code review.
``selected``
    Tier 1 and tier 2. The smallest defensible set.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from fraud_pipeline.config import Config
from fraud_pipeline.eda import NoiseCeiling

logger = logging.getLogger(__name__)

SET_NAMES = ("all", "safe", "selected")

TIER_ONE = 1
TIER_TWO = 2


class SelectionError(RuntimeError):
    """Feature selection cannot produce a usable set."""


@dataclass
class FeatureDecision:
    """What was decided about one feature, and why.

    ``reasons`` is a list because a feature can fail more than one rule, and knowing it
    failed three is more informative than knowing it failed the first one checked.
    """

    feature: str
    dropped_by_tier: int | None = None
    reasons: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    kept_as_input: bool = False

    def drop(self, tier: int, reason: str) -> None:
        self.reasons.append(reason)
        # A tier 1 drop outranks a tier 2 drop. A feature that leaks is not merely weak.
        if self.dropped_by_tier is None or tier < self.dropped_by_tier:
            self.dropped_by_tier = tier

    def in_set(self, set_name: str) -> bool:
        """Whether this feature survives into the named set."""
        if set_name == "all":
            return True
        if self.dropped_by_tier is None:
            return True
        if set_name == "safe":
            return self.dropped_by_tier > TIER_ONE
        return False  # "selected" keeps nothing that was dropped

    @property
    def status(self) -> str:
        if self.dropped_by_tier is None:
            return "keep"
        return f"drop (tier {self.dropped_by_tier})"


@dataclass
class SelectionResult:
    """Every decision from one selection run, plus the sets they produce."""

    decisions: list[FeatureDecision]
    noise_ceiling: float
    ks_noise_ceiling: float
    noise_ceiling_detail: str
    active_set: str
    warnings: list[str] = field(default_factory=list)

    def by_name(self) -> dict[str, FeatureDecision]:
        return {decision.feature: decision for decision in self.decisions}

    def features(self, set_name: str | None = None) -> list[str]:
        """The feature names in a named set, in the order they were ranked."""
        chosen = set_name or self.active_set
        if chosen not in SET_NAMES:
            raise SelectionError(f"unknown feature set {chosen!r}, expected one of {SET_NAMES}")
        return [d.feature for d in self.decisions if d.in_set(chosen)]

    def dropped(self, set_name: str | None = None) -> list[FeatureDecision]:
        chosen = set_name or self.active_set
        return [d for d in self.decisions if not d.in_set(chosen)]

    def pipeline_inputs(self) -> list[str]:
        """Columns kept in the data files whatever the rules said.

        Being a pipeline input is not the same as being a model feature. Time is dropped as
        a feature because it cannot generalise, and kept as an input because hour of day is
        derived from it.
        """
        return [d.feature for d in self.decisions if d.kept_as_input]

    def counts(self) -> dict[str, int]:
        return {name: len(self.features(name)) for name in SET_NAMES}

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_set": self.active_set,
            "noise_ceiling": self.noise_ceiling,
            "ks_noise_ceiling": self.ks_noise_ceiling,
            "noise_ceiling_detail": self.noise_ceiling_detail,
            "counts": self.counts(),
            "sets": {name: self.features(name) for name in SET_NAMES},
            "pipeline_inputs": self.pipeline_inputs(),
            "warnings": self.warnings,
            "decisions": [asdict(d) for d in self.decisions],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_markdown(self, stats: pd.DataFrame) -> str:
        counts = self.counts()
        lines = [
            "# Feature selection",
            "",
            f"Active set: **{self.active_set}**, "
            f"{counts[self.active_set]} of {counts['all']} features kept.",
            "",
            "| Set | Features | What it is |",
            "| --- | --- | --- |",
            f"| all | {counts['all']} | no selection, the honest baseline |",
            f"| safe | {counts['safe']} | tier 1 only, the drops that are correctness problems |",
            f"| selected | {counts['selected']} | tier 1 and tier 2 |",
            "",
            "## Noise ceiling",
            "",
            self.noise_ceiling_detail,
            "",
            "A feature has to fall below **both** ceilings before it counts as noise. AUC",
            "only sees whether fraud sits consistently high or low, so it is blind to a",
            "feature where fraud clusters in the middle of the range. KS catches that, and",
            "requiring both is what stops the rule throwing away real signal.",
            "",
        ]

        if self.warnings:
            lines += ["## Warnings", ""]
            lines += [f"- {w}" for w in self.warnings]
            lines += [""]

        dropped = [d for d in self.decisions if d.dropped_by_tier is not None]
        lines += ["## Dropped", ""]
        if dropped:
            lines += ["| Feature | Tier | Why |", "| --- | --- | --- |"]
            lines += [
                f"| {d.feature} | {d.dropped_by_tier} | {'; '.join(d.reasons)} |" for d in dropped
            ]
        else:
            lines.append("Nothing was dropped. Every rule passed on every feature.")
        lines.append("")

        lookup = stats.set_index("feature")
        lines += [
            "## Kept",
            "",
            "| Feature | AUC | KS | Mutual info | PSI | Range coverage |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for name in self.features():
            row = lookup.loc[name]
            lines.append(
                f"| {name} | {row['auc']:.4f} | {row['ks']:.4f} | {row['mutual_info']:.4f} | "
                f"{row['psi']:.3f} | {row['range_coverage']:.4f} |"
            )

        inputs = self.pipeline_inputs()
        if inputs:
            lines += [
                "",
                "## Kept as pipeline inputs",
                "",
                "Columns kept in the data files whatever the rules decided, because later "
                "stages derive features from them. Being an input is not the same as being "
                "a model feature.",
                "",
                f"{', '.join(inputs)}",
            ]

        return "\n".join([*lines, ""])


# --------------------------------------------------------------------------------------
# The rules
# --------------------------------------------------------------------------------------


def _apply_leakage_rule(
    decisions: dict[str, FeatureDecision], stats: pd.DataFrame, config: Config
) -> list[str]:
    """A feature that separates the classes almost perfectly on its own is a leak.

    Real predictive features on an imbalanced fraud problem do not reach 0.99 AUC alone. A
    feature that does is usually something derived from the answer: a flag set after the
    fraud was confirmed, an ID assigned during the investigation, a refund amount.

    This one gets a loud warning rather than a quiet drop, because if it fires it means
    something is wrong with the data rather than with the feature.
    """
    limit = config.feature_selection.leakage_auc
    warnings = []

    for row in stats.itertuples():
        if row.auc >= limit:
            decisions[row.feature].drop(
                TIER_ONE,
                f"suspected target leakage, univariate AUC {row.auc:.4f} at or above {limit}",
            )
            message = (
                f"{row.feature} reaches {row.auc:.4f} AUC on its own. That is not a good "
                f"feature, it is almost certainly derived from the target. Check where it "
                f"comes from before trusting any result that includes it."
            )
            warnings.append(message)
            logger.error(message)

    return warnings


def _apply_constant_rule(
    decisions: dict[str, FeatureDecision], stats: pd.DataFrame, config: Config
) -> None:
    """A column that is nearly always the same value cannot separate anything."""
    limit = config.feature_selection.dominant_value_share

    for row in stats.itertuples():
        if row.n_unique <= 1:
            decisions[row.feature].drop(TIER_ONE, "constant, one value in every row")
        elif row.dominant_share >= limit:
            decisions[row.feature].drop(
                TIER_ONE,
                f"near constant, one value covers {row.dominant_share:.2%} of rows (limit {limit:.2%})",
            )


def _apply_duplicate_rule(
    decisions: dict[str, FeatureDecision],
    stats: pd.DataFrame,
    pairs: pd.DataFrame,
    config: Config,
) -> None:
    """Of two features carrying the same information, keep the one with more signal.

    Pairs are handled strongest correlation first, and a feature already dropped is skipped,
    so a chain of three mutually correlated features collapses to one rather than to none.
    """
    if pairs.empty:
        return

    auc = stats.set_index("feature")["auc"].to_dict()
    limit = config.feature_selection.duplicate_correlation

    for pair in pairs.sort_values("abs_corr", ascending=False).itertuples():
        if pair.abs_corr < limit:
            continue
        a, b = pair.feature_a, pair.feature_b
        if decisions[a].dropped_by_tier is not None or decisions[b].dropped_by_tier is not None:
            continue

        weaker, stronger = (a, b) if auc.get(a, 0.5) < auc.get(b, 0.5) else (b, a)
        decisions[weaker].drop(
            TIER_ONE,
            f"duplicate of {stronger}, correlation {pair.abs_corr:.4f}, and the weaker of the two",
        )


def _apply_range_rule(
    decisions: dict[str, FeatureDecision], stats: pd.DataFrame, config: Config
) -> None:
    """Drop a feature whose next period values fall outside the range it was fitted on.

    This is the rule that catches a raw timestamp. Time only ever increases, so every
    validation value is larger than every training value and the model is being asked to
    extrapolate beyond anything it has seen. A shifted distribution is survivable. No
    overlap at all is not.
    """
    limit = config.feature_selection.min_range_coverage

    for row in stats.itertuples():
        if row.range_coverage < limit:
            decisions[row.feature].drop(
                TIER_ONE,
                f"only {row.range_coverage:.2%} of validation values fall inside the "
                f"training range (limit {limit:.0%}), so it cannot generalise",
            )


def _apply_low_signal_rule(
    decisions: dict[str, FeatureDecision],
    stats: pd.DataFrame,
    noise: NoiseCeiling,
    config: Config,
) -> None:
    """Drop features that cannot be told apart from random numbers.

    Both thresholds are permutation noise ceilings rather than round numbers. That matters
    here: with only a few hundred fraud rows the AUC noise floor sits near 0.55, well above
    the 0.5 that intuition suggests, so a hand picked threshold would keep pure noise.

    A feature has to fail **both** measures before it is dropped, and that is the whole
    point of the rule.

    AUC only sees a monotonic relationship, so it asks whether fraud sits consistently high
    or low on this feature. KS asks whether the two distributions differ in any way at all.
    A feature where fraud clusters in the middle of the range scores near 0.5 on AUC while
    scoring high on KS, and an AUC only rule would throw it away.

    This is not hypothetical. On the ULB data, Amount scores 0.548 AUC, below the ceiling,
    while its KS distance is 0.26, far above it. Fraudulent amounts really are distributed
    differently, they are just not consistently larger or smaller. An AUC only version of
    this rule dropped Amount, which was wrong.

    The weakness that remains is interactions. A feature that matters only in combination
    with another still gets dropped, because nothing univariate can see that. That is why
    this is tier 2, why it can be switched off, and why the `safe` and `all` sets exist to
    measure against.
    """
    if not config.feature_selection.drop_low_signal:
        return

    for row in stats.itertuples():
        weak_ranking = row.auc < noise.ceiling
        weak_distribution = row.ks < noise.ks_ceiling

        if weak_ranking and weak_distribution:
            decisions[row.feature].drop(
                TIER_TWO,
                f"no measurable signal: AUC {row.auc:.4f} below the {noise.ceiling:.4f} "
                f"ceiling and KS {row.ks:.4f} below the {noise.ks_ceiling:.4f} ceiling",
            )
        elif weak_ranking:
            # Worth saying out loud. This feature would have been dropped by the naive
            # version of this rule, and keeping it is a deliberate decision.
            logger.info(
                "%s kept: AUC %.4f is below the ceiling, but KS %.4f is above it, so the "
                "distributions differ without fraud being consistently higher or lower",
                row.feature,
                row.auc,
                row.ks,
            )


def _collect_psi_warnings(stats: pd.DataFrame, config: Config) -> list[str]:
    """Report features that moved between train and validation, without acting on them.

    A shifted feature can still be a useful one, and on this dataset half the features moved
    to some degree, so dropping them all would leave almost nothing. It belongs in the
    report so it can inform monitoring later.
    """
    limit = config.feature_selection.psi_warn
    shifted = stats[stats["psi"] >= limit].sort_values("psi", ascending=False)

    if shifted.empty:
        return []

    listed = ", ".join(f"{r.feature} ({r.psi:.2f})" for r in shifted.itertuples())
    return [
        f"{len(shifted)} feature(s) shifted between train and validation with PSI at or "
        f"above {limit}: {listed}. These are kept, but they are the first place to look if "
        f"the model does worse on test than on validation."
    ]


# --------------------------------------------------------------------------------------
# Putting it together
# --------------------------------------------------------------------------------------


def select_features(
    stats: pd.DataFrame,
    pairs: pd.DataFrame,
    noise: NoiseCeiling,
    config: Config,
    duplicate_columns: list[tuple[str, str]] | None = None,
) -> SelectionResult:
    """Run every rule and return the decisions.

    Rules run in tier order so the recorded reason for a drop is the most serious one. A
    leaking feature that is also weak should read as a leak.
    """
    decisions = {row.feature: FeatureDecision(feature=row.feature) for row in stats.itertuples()}
    if not decisions:
        raise SelectionError("no features to select from")

    warnings = _apply_leakage_rule(decisions, stats, config)

    for first, second in duplicate_columns or []:
        if second in decisions:
            decisions[second].drop(TIER_ONE, f"identical to {first}, byte for byte")

    _apply_constant_rule(decisions, stats, config)
    _apply_duplicate_rule(decisions, stats, pairs, config)
    _apply_range_rule(decisions, stats, config)
    _apply_low_signal_rule(decisions, stats, noise, config)

    warnings += _collect_psi_warnings(stats, config)

    for row in stats.itertuples():
        decision = decisions[row.feature]
        decision.kept_as_input = row.feature in config.feature_selection.keep_as_input
        decision.evidence = {
            "auc": round(float(row.auc), 4),
            "ks": round(float(row.ks), 4),
            "abs_corr": round(float(row.abs_corr), 4),
            "mutual_info": round(float(row.mutual_info), 4),
            "psi": round(float(row.psi), 4),
            "range_coverage": round(float(row.range_coverage), 6),
        }

    result = SelectionResult(
        decisions=[decisions[row.feature] for row in stats.itertuples()],
        noise_ceiling=noise.ceiling,
        ks_noise_ceiling=noise.ks_ceiling,
        noise_ceiling_detail=noise.describe(),
        active_set=config.feature_selection.active_set,
        warnings=warnings,
    )

    if not result.features():
        raise SelectionError(
            f"every feature was dropped from the {result.active_set!r} set. "
            f"Loosen the thresholds in feature_selection, or set active_set to 'safe'."
        )

    return result


def tagged_name(filename: str, tag: str | None) -> str:
    """Insert a tag before the extension, so two passes do not overwrite each other.

    Selection runs twice: once in the eda stage over the columns the file arrived with, and
    again in the feature stage over the engineered columns the model will actually see. Both
    are worth keeping. Sharing one filename meant the second pass silently replaced the
    first, so the committed report no longer matched the run it described.
    """
    if not tag:
        return filename
    stem, _, extension = filename.rpartition(".")
    return f"{stem}_{tag}.{extension}" if stem else f"{filename}_{tag}"


def write_selection(
    result: SelectionResult, stats: pd.DataFrame, config: Config, tag: str | None = None
) -> tuple[str, str]:
    """Write the machine readable json and the human readable markdown."""
    from fraud_pipeline.paths import ensure_dir

    json_path = ensure_dir(config.paths.interim()) / tagged_name(
        config.feature_selection.output_file, tag
    )
    json_path.write_text(result.to_json(), encoding="utf-8")

    report_path = ensure_dir(config.paths.tables_dir()) / tagged_name(
        config.feature_selection.report_file, tag
    )
    report_path.write_text(result.to_markdown(stats), encoding="utf-8")

    return str(json_path), str(report_path)


def load_selection(config: Config, tag: str | None = None) -> dict[str, Any]:
    """Read a selection back. Stage 3 and everything after it uses this."""
    path = config.paths.interim() / tagged_name(config.feature_selection.output_file, tag)

    if not path.is_file():
        stage = "features" if tag else "eda"
        raise SelectionError(
            f"feature selection not found: {path}\nRun the {stage} stage first: fraud {stage}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_selected_features(
    config: Config, set_name: str | None = None, tag: str | None = None
) -> list[str]:
    """The feature names a later stage should train on."""
    payload = load_selection(config, tag)
    chosen = set_name or payload.get("active_set", "selected")
    if chosen not in payload["sets"]:
        raise SelectionError(f"unknown feature set {chosen!r}, expected one of {SET_NAMES}")
    return list(payload["sets"][chosen])

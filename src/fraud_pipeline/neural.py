"""A small neural network, wrapped so it behaves like any other scikit learn classifier.

The wrapper exists so the training sweep does not need to know that one of its five models is
a different kind of object. It gets a scaler in front of it, a SMOTE step before it, and a
``predict_proba`` call after it, exactly like logistic regression does. Everything torch
specific is behind ``fit``.

Three decisions inside are worth reading before trusting a number that comes out of it.

**Early stopping needs data the network is not fitted on**, so a slice is carved out of the
training rows. That slice is *inside* the training window, never the project's validation
split, which stays clean for comparing models against each other.

**That inner slice is a stratified random sample, not the last N rows in time.**
A chronological inner split would sit better with the rest of the project, and it cannot be
used here: under SMOTE the rows arriving at the network are synthetic and shuffled, so time
order no longer exists. Using one scheme for two strategies and a different one for the third
would make the comparison between them meaningless, which is a worse problem than the one it
would solve.

**Early stopping watches average precision, not loss.** Loss under a weighted objective moves
for reasons that have nothing to do with whether the ranking improved, and the ranking is what
the project reports.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

#: Below this many positives in the inner slice, early stopping is measuring noise.
_MIN_EARLY_STOPPING_POSITIVES = 5


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    """A dense network for tabular binary classification.

    Args:
        hidden_sizes: width of each hidden layer.
        dropout: dropout probability applied after every hidden layer.
        learning_rate: Adam learning rate.
        batch_size: rows per gradient step.
        epochs: maximum passes over the training data.
        early_stopping_patience: epochs without improvement before stopping. 0 disables it.
        validation_fraction: share of the training rows held back for early stopping.
        class_weight: ``"balanced"`` weights the positive class up by the class ratio, which
            is the same idea scikit learn's keyword expresses, applied through the loss.
        random_state: seed for the weights, the shuffling and the inner split.
        verbose: log a line per epoch.
    """

    def __init__(
        self,
        hidden_sizes: tuple[int, ...] | list[int] = (64, 32),
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        batch_size: int = 512,
        epochs: int = 30,
        early_stopping_patience: int = 5,
        validation_fraction: float = 0.1,
        class_weight: str | None = None,
        random_state: int = 42,
        verbose: bool = False,
    ) -> None:
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.validation_fraction = validation_fraction
        self.class_weight = class_weight
        self.random_state = random_state
        self.verbose = verbose

    # ----------------------------------------------------------------------------------
    # Construction
    # ----------------------------------------------------------------------------------

    def _build_network(self, n_features: int):
        import torch.nn as nn

        layers: list[Any] = []
        width = n_features
        for size in self.hidden_sizes:
            layers += [nn.Linear(width, size), nn.ReLU(), nn.Dropout(self.dropout)]
            width = size
        # One logit. BCEWithLogitsLoss applies the sigmoid itself, in a form that does not
        # overflow the way a separate sigmoid followed by a log would.
        layers.append(nn.Linear(width, 1))
        return nn.Sequential(*layers)

    def _positive_weight(self, y: np.ndarray) -> float | None:
        """How much more a positive counts than a negative, or None for no weighting."""
        if self.class_weight != "balanced":
            return None

        positives = float(y.sum())
        negatives = float(len(y) - positives)
        if positives == 0:
            return None
        return negatives / positives

    # ----------------------------------------------------------------------------------
    # Fitting
    # ----------------------------------------------------------------------------------

    def fit(self, X, y):
        import torch
        from torch import nn

        features = np.asarray(X, dtype=np.float32)
        target = np.asarray(y).astype(np.float32).ravel()

        if features.ndim != 2:
            raise ValueError(f"expected a 2d feature matrix, got shape {features.shape}")
        if len(features) != len(target):
            raise ValueError(f"X has {len(features)} rows and y has {len(target)}")

        self.classes_ = np.unique(target).astype(int)
        self.n_features_in_ = features.shape[1]

        torch.manual_seed(self.random_state)
        # Single threaded so that two runs on the same machine agree. Float addition is not
        # associative, so thread count changes the last digits, and this project asserts that
        # a rerun reproduces its numbers.
        torch.set_num_threads(1)

        train_x, train_y, holdout = self._split_for_early_stopping(features, target)

        network = self._build_network(self.n_features_in_)
        optimiser = torch.optim.Adam(network.parameters(), lr=self.learning_rate)

        pos_weight = self._positive_weight(train_y)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], dtype=torch.float32)
            if pos_weight is not None
            else None
        )

        x_tensor = torch.from_numpy(train_x)
        y_tensor = torch.from_numpy(train_y).unsqueeze(1)
        generator = torch.Generator().manual_seed(self.random_state)

        best_score, best_state, waited = -np.inf, None, 0
        self.n_epochs_run_ = 0

        for epoch in range(self.epochs):
            network.train()
            order = torch.randperm(len(x_tensor), generator=generator)

            for start in range(0, len(order), self.batch_size):
                batch = order[start : start + self.batch_size]
                optimiser.zero_grad()
                loss = criterion(network(x_tensor[batch]), y_tensor[batch])
                loss.backward()
                optimiser.step()

            self.n_epochs_run_ = epoch + 1

            if holdout is None:
                continue

            score = self._score_holdout(network, holdout)
            if score > best_score:
                best_score, waited = score, 0
                best_state = {k: v.detach().clone() for k, v in network.state_dict().items()}
            else:
                waited += 1

            if self.verbose:
                logger.info("epoch %s: holdout average precision %.4f", epoch + 1, score)

            if self.early_stopping_patience and waited >= self.early_stopping_patience:
                logger.debug("early stopping at epoch %s", epoch + 1)
                break

        if best_state is not None:
            network.load_state_dict(best_state)

        self.best_score_ = float(best_score) if holdout is not None else float("nan")
        self.network_ = network
        return self

    def _split_for_early_stopping(self, features: np.ndarray, target: np.ndarray):
        """Carve an inner slice out of the training rows, or decline to.

        Returns ``(train_x, train_y, holdout)`` where holdout is ``None`` when early stopping
        cannot be done honestly. With a few positives in the slice, the score it produces is
        noise, and stopping on noise is worse than not stopping at all, so the network trains
        for the full number of epochs instead.
        """
        if self.early_stopping_patience <= 0 or self.validation_fraction <= 0:
            return features, target, None

        positives = int(target.sum())
        expected = positives * self.validation_fraction
        if positives < 2 or expected < _MIN_EARLY_STOPPING_POSITIVES:
            logger.debug(
                "not enough positives (%s) for an early stopping slice, training all epochs",
                positives,
            )
            return features, target, None

        train_x, holdout_x, train_y, holdout_y = train_test_split(
            features,
            target,
            test_size=self.validation_fraction,
            stratify=target,
            random_state=self.random_state,
        )
        return train_x, train_y, (holdout_x, holdout_y)

    def _score_holdout(self, network, holdout) -> float:
        import torch

        holdout_x, holdout_y = holdout
        network.eval()
        with torch.no_grad():
            logits = network(torch.from_numpy(holdout_x)).squeeze(1)
            probabilities = torch.sigmoid(logits).numpy()

        if holdout_y.sum() == 0:
            return 0.0
        return float(average_precision_score(holdout_y, probabilities))

    # ----------------------------------------------------------------------------------
    # Predicting
    # ----------------------------------------------------------------------------------

    def predict_proba(self, X) -> np.ndarray:
        import torch

        if not hasattr(self, "network_"):
            raise RuntimeError("this classifier has not been fitted yet")

        features = np.asarray(X, dtype=np.float32)
        if features.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} features, got {features.shape[1]}")

        self.network_.eval()
        with torch.no_grad():
            logits = self.network_(torch.from_numpy(features)).squeeze(1)
            positive = torch.sigmoid(logits).numpy().astype(np.float64)

        # The two column form scikit learn uses, so nothing downstream needs a special case.
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

"""Feature engineering and payload adaptation.

Two responsibilities live here, deliberately separated:

``PairwiseFeatureBuilder``
    A scikit-learn transformer. It is *inside* the fitted pipeline, so the exact same
    transformation is applied at training time and at serving time. This removes the
    single most common source of training/serving skew.

``FeatureTransformer``
    An **Adapter**: it converts the API's nested JSON payload into the flat, column-ordered
    ``DataFrame`` that the pipeline expects. The model layer therefore never has to know
    anything about HTTP or Pydantic.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.constants import (
    BASE_FEATURES,
    COMPARISON_FEATURES,
    GROUP_1_FEATURES,
    GROUP_2_FEATURES,
    LEAKAGE_FEATURES,
    N_GROUP_FEATURES,
)
from src.exceptions import InvalidFeaturePayloadError


def drop_leakage_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove post-campaign columns (``g1_21``, ``g2_21``, ``c_28``).

    These variables are recorded *after* the campaign has run. They correlate almost
    perfectly with the target but are unavailable at prediction time, so training on
    them produces a model that looks excellent offline and is useless in production.
    """
    present = [column for column in LEAKAGE_FEATURES if column in frame.columns]
    return frame.drop(columns=present)


class PairwiseFeatureBuilder(BaseEstimator, TransformerMixin):
    """Derive explicit group-1 vs group-2 comparison features.

    The task is inherently *relative*: we are not scoring one group, we are choosing
    between two. Tree models can approximate ``g1_i - g2_i`` only through many splits,
    so providing the differences (and optionally ratios) directly makes the signal far
    easier to learn.

    Parameters
    ----------
    add_differences:
        Add ``diff_i = g1_i - g2_i`` for every paired variable.
    add_ratios:
        Add ``ratio_i = g1_i / (g2_i + epsilon)`` for every paired variable.
    epsilon:
        Numerical guard used in the ratio denominator.
    """

    def __init__(
        self,
        add_differences: bool = True,
        add_ratios: bool = True,
        epsilon: float = 1e-6,
    ) -> None:
        self.add_differences = add_differences
        self.add_ratios = add_ratios
        self.epsilon = epsilon

    def fit(self, X: pd.DataFrame, y: Any = None) -> PairwiseFeatureBuilder:  # noqa: N803
        """Record the incoming column order so ``transform`` is deterministic."""
        frame = self._as_frame(X)
        self.feature_names_in_ = list(frame.columns)
        self.n_features_in_ = len(self.feature_names_in_)
        self.feature_names_out_ = list(self._build(frame).columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        """Return the input frame enriched with the derived comparison features."""
        frame = self._as_frame(X)
        out = self._build(frame)
        if hasattr(self, "feature_names_out_"):
            # Guarantee identical column order between fit and transform.
            out = out.reindex(columns=self.feature_names_out_)
        return out

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Expose output names so the pipeline stays introspectable."""
        return np.asarray(getattr(self, "feature_names_out_", []), dtype=object)

    # ------------------------------------------------------------------ internals

    def _build(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        derived: dict[str, pd.Series] = {}

        for index in range(1, N_GROUP_FEATURES + 1):
            left, right = f"g1_{index}", f"g2_{index}"
            if left not in frame.columns or right not in frame.columns:
                continue

            g1 = pd.to_numeric(frame[left], errors="coerce")
            g2 = pd.to_numeric(frame[right], errors="coerce")

            if self.add_differences:
                derived[f"diff_{index}"] = g1 - g2
            if self.add_ratios:
                derived[f"ratio_{index}"] = g1 / (g2.abs() + self.epsilon)

        if derived:
            out = pd.concat([out, pd.DataFrame(derived, index=frame.index)], axis=1)

        # Replace infinities produced by ratios with NaN so the imputer handles them.
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _as_frame(X: Any) -> pd.DataFrame:  # noqa: N803
        if isinstance(X, pd.DataFrame):
            return X
        raise InvalidFeaturePayloadError(
            "PairwiseFeatureBuilder expects a pandas DataFrame with named columns; "
            f"received {type(X).__name__}."
        )


class FeatureTransformer:
    """Adapter converting API payloads into a model-ready ``DataFrame``.

    Responsibilities:

    * validate that the required feature keys are present,
    * reject post-campaign (leakage) keys if a caller tries to send them,
    * coerce values to floats, mapping ``None`` to ``NaN`` for the pipeline's imputer,
    * emit columns in the canonical order the pipeline was fitted on.
    """

    def __init__(self, expected_columns: list[str] | None = None) -> None:
        self.expected_columns = list(expected_columns or BASE_FEATURES)

    def from_payload(
        self,
        group_1: Mapping[str, float | None],
        group_2: Mapping[str, float | None],
        comparison: Mapping[str, float | None],
    ) -> pd.DataFrame:
        """Build a single-row frame from the three feature blocks of one comparison."""
        return self.from_payloads([(group_1, group_2, comparison)])

    def from_payloads(
        self,
        payloads: Iterable[
            tuple[
                Mapping[str, float | None],
                Mapping[str, float | None],
                Mapping[str, float | None],
            ]
        ],
    ) -> pd.DataFrame:
        """Build an N-row frame from N comparisons (batch endpoint)."""
        rows: list[dict[str, float]] = []

        for group_1, group_2, comparison in payloads:
            self._validate_block(group_1, GROUP_1_FEATURES, "group_1")
            self._validate_block(group_2, GROUP_2_FEATURES, "group_2")
            self._validate_block(comparison, COMPARISON_FEATURES, "comparison")

            merged: dict[str, Any] = {**dict(group_1), **dict(group_2), **dict(comparison)}
            rows.append(
                {column: self._to_float(merged.get(column)) for column in self.expected_columns}
            )

        if not rows:
            raise InvalidFeaturePayloadError("At least one comparison must be provided.")

        return pd.DataFrame(rows, columns=self.expected_columns)

    def from_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Adapt an arbitrary DataFrame (e.g. a CSV upload) to the canonical contract."""
        missing = [column for column in self.expected_columns if column not in frame.columns]
        if missing:
            raise InvalidFeaturePayloadError(
                f"Uploaded data is missing {len(missing)} required column(s): "
                f"{', '.join(missing[:10])}{'...' if len(missing) > 10 else ''}"
            )
        return frame.reindex(columns=self.expected_columns).apply(pd.to_numeric, errors="coerce")

    # ------------------------------------------------------------------ internals

    @staticmethod
    def _validate_block(
        block: Mapping[str, float | None],
        required: list[str],
        block_name: str,
    ) -> None:
        keys = set(block)

        leaked = keys.intersection(LEAKAGE_FEATURES)
        if leaked:
            raise InvalidFeaturePayloadError(
                f"'{block_name}' contains post-campaign variable(s) {sorted(leaked)}. "
                "These are recorded after the campaign and must never be used as inputs."
            )

        missing = [key for key in required if key not in keys]
        if missing:
            raise InvalidFeaturePayloadError(
                f"'{block_name}' is missing required key(s): {', '.join(missing)}"
            )

        unexpected = sorted(keys.difference(required))
        if unexpected:
            raise InvalidFeaturePayloadError(
                f"'{block_name}' contains unexpected key(s): {', '.join(unexpected)}"
            )

    @staticmethod
    def _to_float(value: Any) -> float:
        if value is None:
            return float("nan")
        try:
            return float(value)
        except (TypeError, ValueError) as error:
            raise InvalidFeaturePayloadError(f"Value {value!r} is not numeric.") from error

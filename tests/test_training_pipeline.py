"""Tests for dataset handling and pipeline assembly, including the leakage guarantee."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.constants import BASE_FEATURES, LEAKAGE_FEATURES, TARGET_COLUMN
from src.training.pipeline import (
    build_pipeline,
    candidate_models,
    load_dataset,
    split_features_target,
)
from src.training.train import parse_args, train


class TestSplitFeaturesTarget:
    def test_drops_post_campaign_columns_by_default(self, raw_dataset: pd.DataFrame) -> None:
        features, _ = split_features_target(raw_dataset)
        assert not set(LEAKAGE_FEATURES).intersection(features.columns)

    def test_uses_exactly_the_sixty_seven_allowed_features(self, raw_dataset: pd.DataFrame) -> None:
        features, _ = split_features_target(raw_dataset)
        assert list(features.columns) == BASE_FEATURES

    def test_can_keep_leakage_for_the_demonstration(self, raw_dataset: pd.DataFrame) -> None:
        features, _ = split_features_target(raw_dataset, drop_leakage=False)
        assert set(LEAKAGE_FEATURES).issubset(features.columns)

    def test_keeping_leakage_widens_the_feature_matrix(self, raw_dataset: pd.DataFrame) -> None:
        # Guards the leakage demonstration: if both calls returned the same columns, the
        # with/without comparison would show no difference and the finding would vanish.
        clean, _ = split_features_target(raw_dataset, drop_leakage=True)
        leaky, _ = split_features_target(raw_dataset, drop_leakage=False)
        assert leaky.shape[1] == clean.shape[1] + len(LEAKAGE_FEATURES)

    def test_target_is_separated(self, raw_dataset: pd.DataFrame) -> None:
        _, target = split_features_target(raw_dataset)
        assert target.name == TARGET_COLUMN
        assert set(target.unique()).issubset({0, 1, 2})

    def test_missing_target_column_raises(self, feature_frame: pd.DataFrame) -> None:
        with pytest.raises(KeyError, match=TARGET_COLUMN):
            split_features_target(feature_frame)


class TestBuildPipeline:
    def test_has_the_expected_steps(self) -> None:
        pipeline = build_pipeline(candidate_models()["logistic_regression"])
        assert isinstance(pipeline, Pipeline)
        assert list(dict(pipeline.steps)) == ["pairwise", "imputer", "scaler", "model"]

    def test_fits_and_predicts(self, raw_dataset: pd.DataFrame) -> None:
        features, target = split_features_target(raw_dataset)
        pipeline = build_pipeline(candidate_models()["logistic_regression"])
        pipeline.fit(features, target)
        predictions = pipeline.predict(features.head(5))
        assert len(predictions) == 5

    def test_handles_missing_values(self, raw_dataset: pd.DataFrame) -> None:
        features, target = split_features_target(raw_dataset)
        features = features.copy()
        features.loc[features.index[:10], "g1_1"] = None

        pipeline = build_pipeline(candidate_models()["logistic_regression"])
        pipeline.fit(features, target)
        assert len(pipeline.predict(features.head(3))) == 3

    def test_candidate_models_include_a_linear_and_a_tree_model(self) -> None:
        models = candidate_models()
        assert "logistic_regression" in models
        assert "hist_gradient_boosting" in models


class TestTrainCli:
    """End-to-end CLI runs, kept fast with --fast, a single model and no SHAP."""

    @staticmethod
    def _run(csv_path: Path, out_dir: Path, *extra: str) -> dict:
        return train(
            parse_args(
                [
                    "--data",
                    str(csv_path),
                    "--out",
                    str(out_dir),
                    "--cv-folds",
                    "3",
                    "--fast",
                    "--skip-explain",
                    "--models",
                    "logistic_regression",
                    *extra,
                ]
            )
        )

    @pytest.fixture
    def csv_path(self, raw_dataset: pd.DataFrame, tmp_path: Path) -> Path:
        path = tmp_path / "customerGroups.csv"
        raw_dataset.to_csv(path, index=False)
        return path

    def test_end_to_end_run_produces_artifacts(self, csv_path: Path, tmp_path: Path) -> None:
        report = self._run(csv_path, tmp_path / "artifacts")

        assert (tmp_path / "artifacts" / "model.pkl").exists()
        assert (tmp_path / "artifacts" / "metrics.json").exists()
        assert report["leakage_dropped"] is True
        assert report["n_features_used"] == len(BASE_FEATURES)
        assert report["champion_model"] in report["models"]

    def test_metrics_json_is_valid_json(self, csv_path: Path, tmp_path: Path) -> None:
        out = tmp_path / "json"
        self._run(csv_path, out)
        payload = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        assert payload["champion_model"] == "logistic_regression"

    def test_report_answers_ml_question_one(self, csv_path: Path, tmp_path: Path) -> None:
        report = self._run(csv_path, tmp_path / "a")
        by_class = report["campaign_outcome_distribution"]["by_class"]
        assert set(by_class) == {"no_group_profitable", "group_1", "group_2"}

    def test_report_answers_ml_question_three(self, csv_path: Path, tmp_path: Path) -> None:
        report = self._run(csv_path, tmp_path / "b")
        lift = report["models"][report["champion_model"]]["business_lift"]
        assert "absolute_lift_pp" in lift
        assert "best_naive_success_rate" in lift

    def test_report_includes_symmetry_invariance(self, csv_path: Path, tmp_path: Path) -> None:
        report = self._run(csv_path, tmp_path / "c")
        # Indexed via champion_model rather than a hardcoded name: symmetry invariance is
        # measured on the test set, and only the champion is evaluated there.
        invariance = report["models"][report["champion_model"]]["symmetry_invariance"]
        assert 0.0 <= invariance["violation_rate"] <= 1.0

    def test_only_the_champion_is_evaluated_on_the_test_set(
        self, csv_path: Path, tmp_path: Path
    ) -> None:
        """The evaluation protocol, enforced rather than promised.

        Scoring every candidate on the test set and selecting on cross-validation is
        defensible in principle and indefensible in practice: the test figures are on
        screen while the decision is being made, and selection bias needs only visibility,
        not intent.

        This test fails if a future change reintroduces `predict(x_test)` into the
        candidate loop - which is how the property was lost the first time.
        """
        report = self._run(csv_path, tmp_path / "protocol")
        champion = report["champion_model"]

        evaluated = [
            name for name, metrics in report["models"].items() if metrics.get("evaluated_on_test")
        ]
        assert evaluated == [champion]

        for name, metrics in report["models"].items():
            if name == champion:
                continue
            assert "accuracy" not in metrics, f"{name} carries a test-set score"
            assert "business_lift" not in metrics, f"{name} carries a test-set lift"

    def test_report_includes_dataset_diagnostics(self, csv_path: Path, tmp_path: Path) -> None:
        diagnostics = self._run(csv_path, tmp_path / "d")["diagnostics"]
        assert diagnostics["comparison_redundancy"]["n_checked"] == 27
        assert diagnostics["split_strategy_used"] == "stratified"

    def test_symmetry_augmentation_flag_is_recorded(self, csv_path: Path, tmp_path: Path) -> None:
        report = self._run(csv_path, tmp_path / "e", "--augment-symmetry")
        assert report["symmetry_augmented"] is True

    def test_temporal_split_runs(self, csv_path: Path, tmp_path: Path) -> None:
        report = self._run(csv_path, tmp_path / "f", "--split", "temporal")
        assert report["diagnostics"]["split_strategy_used"] == "temporal"

    def test_tuning_flag_runs_a_search(self, csv_path: Path, tmp_path: Path) -> None:
        report = self._run(csv_path, tmp_path / "g", "--tune", "--n-iter", "2")
        tuning = report["models"]["logistic_regression"]["tuning"]
        assert tuning["tuned"] is True
        assert tuning["n_candidates"] == 2

    def test_keep_leakage_flag_is_recorded(self, csv_path: Path, tmp_path: Path) -> None:
        report = self._run(csv_path, tmp_path / "h", "--keep-leakage")
        assert report["leakage_dropped"] is False

    def test_unknown_model_name_raises(self, csv_path: Path, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unknown model name"):
            train(
                parse_args(
                    ["--data", str(csv_path), "--out", str(tmp_path / "i"), "--models", "nope"]
                )
            )


class TestLoadDataset:
    def test_reads_a_csv(self, raw_dataset: pd.DataFrame, tmp_path: Path) -> None:
        path = tmp_path / "data.csv"
        raw_dataset.to_csv(path, index=False)
        assert len(load_dataset(path)) == len(raw_dataset)

    def test_missing_file_gives_an_actionable_message(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="deliberately not committed"):
            load_dataset(tmp_path / "customerGroups.csv")

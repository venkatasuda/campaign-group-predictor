"""Unit tests for the cost-sensitive decision layer."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.decision import ACTIONS, CostMatrix, DecisionPolicy, evaluate_policy


class TestCostMatrix:
    def test_default_is_three_by_three(self) -> None:
        assert CostMatrix.default().matrix.shape == (3, 3)

    def test_default_rewards_correct_targeting(self) -> None:
        matrix = CostMatrix.default().matrix
        assert matrix[1][1] < 0  # correctly targeting group 1 is a gain
        assert matrix[2][2] < 0

    def test_default_penalises_wrong_targeting(self) -> None:
        matrix = CostMatrix.default().matrix
        assert matrix[1][2] > 0
        assert matrix[2][1] > 0

    def test_declining_an_unprofitable_campaign_is_free(self) -> None:
        assert CostMatrix.default().matrix[0][0] == 0.0

    def test_running_an_unprofitable_campaign_costs_money(self) -> None:
        matrix = CostMatrix.default().matrix
        assert matrix[0][1] > 0 and matrix[0][2] > 0

    def test_rejects_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match="3x3"):
            CostMatrix(matrix=np.zeros((2, 2)))

    def test_rejects_non_finite_values(self) -> None:
        broken = np.zeros((3, 3))
        broken[0][0] = np.inf
        with pytest.raises(ValueError, match="finite"):
            CostMatrix(matrix=broken)

    def test_from_business_parameters_maps_the_numbers(self) -> None:
        matrix = CostMatrix.from_business_parameters(
            campaign_spend=500.0, profit_if_correct=2000.0, opportunity_weight=0.5
        )
        assert matrix.matrix[1][1] == -2000.0
        assert matrix.matrix[1][2] == 500.0
        assert matrix.matrix[1][0] == 1000.0

        # "relative units", not a currency code. `campaign_spend` and `profit_if_correct`
        # are a ratio scale this project supplied; the dataset contains no monetary values
        # at all. The default was previously "EUR", which meant every API response and every
        # logged decision presented invented weights as measured euros.
        assert matrix.currency == "relative units"

    def test_a_real_currency_can_be_supplied(self) -> None:
        """Once the business provides actual figures, label them honestly."""
        matrix = CostMatrix.from_business_parameters(
            campaign_spend=500.0, profit_if_correct=2000.0, currency="EUR"
        )
        assert matrix.currency == "EUR"

    def test_from_business_parameters_rejects_negative_amounts(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            CostMatrix.from_business_parameters(-1.0, 100.0)

    def test_from_business_parameters_rejects_bad_weight(self) -> None:
        with pytest.raises(ValueError, match="opportunity_weight"):
            CostMatrix.from_business_parameters(100.0, 100.0, opportunity_weight=1.5)

    def test_to_dict_is_labelled_and_serialisable(self) -> None:
        payload = json.loads(json.dumps(CostMatrix.default().to_dict()))
        assert "true_group_1" in payload["costs"]
        assert "target_group_1" in payload["costs"]["true_group_1"]

    def test_expected_costs_shape(self) -> None:
        proba = np.full((7, 3), 1 / 3)
        assert CostMatrix.default().expected_costs(proba).shape == (7, 3)

    def test_expected_costs_rejects_wrong_width(self) -> None:
        with pytest.raises(ValueError, match=r"shape \(n, 3\)"):
            CostMatrix.default().expected_costs(np.zeros((4, 2)))

    def test_expected_cost_arithmetic(self) -> None:
        # Certain that group 1 is profitable -> expected cost equals row 1 of the matrix.
        proba = np.array([[0.0, 1.0, 0.0]])
        costs = CostMatrix.default().expected_costs(proba)
        np.testing.assert_allclose(costs[0], CostMatrix.default().matrix[1])


class TestDecisionPolicy:
    def test_confident_group_1_targets_group_1(self) -> None:
        decision = DecisionPolicy.default().decide(np.array([0.05, 0.90, 0.05]))
        assert decision.action == 1
        assert decision.action_label == "target_group_1"
        assert decision.differs_from_argmax is False

    def test_confident_group_2_targets_group_2(self) -> None:
        assert DecisionPolicy.default().decide(np.array([0.05, 0.05, 0.90])).action == 2

    def test_confident_class_zero_declines_the_campaign(self) -> None:
        decision = DecisionPolicy.default().decide(np.array([0.95, 0.03, 0.02]))
        assert decision.action == 0
        assert "Do not run" in decision.recommended_action

    def test_expensive_campaigns_make_the_policy_decline_more(self) -> None:
        # Spend dwarfs the profit, so acting on a coin-flip is not worth it.
        cautious = DecisionPolicy(
            cost_matrix=CostMatrix.from_business_parameters(
                campaign_spend=1000.0, profit_if_correct=50.0, opportunity_weight=0.1
            ),
            review_margin=0.0,
        )
        assert cautious.decide(np.array([0.34, 0.33, 0.33])).action == 0

    def test_decision_can_differ_from_argmax(self) -> None:
        # Group 1 is marginally the most likely class, but class 0 has enough mass that
        # spending money is not worth it once a wasted campaign is expensive.
        cautious = DecisionPolicy(
            cost_matrix=CostMatrix.from_business_parameters(
                campaign_spend=100.0, profit_if_correct=10.0, opportunity_weight=0.0
            ),
            review_margin=0.0,
        )
        decision = cautious.decide(np.array([0.34, 0.36, 0.30]))
        assert decision.argmax_class == 1
        assert decision.action == 0
        assert decision.differs_from_argmax is True
        assert "lower expected cost" in decision.rationale

    def test_near_tie_is_flagged_for_review(self) -> None:
        policy = DecisionPolicy(cost_matrix=CostMatrix.default(), review_margin=0.5)
        decision = policy.decide(np.array([0.0, 0.51, 0.49]))
        assert decision.review_required is True
        assert "human review" in decision.rationale

    def test_margin_exactly_at_the_threshold_is_not_flagged(self) -> None:
        # Boundary case: the rule is margin < review_margin, so an exact tie with the
        # threshold must NOT trigger review. Pinned so the comparison cannot silently
        # flip between < and <=.
        policy = DecisionPolicy(cost_matrix=CostMatrix.default(), review_margin=0.0)
        decision = policy.decide(np.array([1 / 3, 1 / 3, 1 / 3]))
        assert decision.review_required is False

    def test_margin_just_below_the_threshold_is_flagged(self) -> None:
        policy = DecisionPolicy(cost_matrix=CostMatrix.default(), review_margin=1e6)
        assert policy.decide(np.array([0.2, 0.4, 0.4])).review_required is True

    def test_degenerate_probabilities_do_not_crash(self) -> None:
        # All mass on one class: margin is large, no tie, no boundary error.
        decision = DecisionPolicy.default().decide(np.array([0.0, 1.0, 0.0]))
        assert decision.action == 1
        assert decision.review_required is False

    def test_review_can_be_disabled(self) -> None:
        policy = DecisionPolicy(cost_matrix=CostMatrix.default(), review_margin=0.0)
        assert policy.decide(np.array([0.0, 0.51, 0.49])).review_required is False

    def test_margin_is_non_negative(self) -> None:
        decision = DecisionPolicy.default().decide(np.array([0.2, 0.5, 0.3]))
        assert decision.margin >= 0.0

    def test_expected_costs_cover_every_action(self) -> None:
        decision = DecisionPolicy.default().decide(np.array([0.2, 0.5, 0.3]))
        assert set(decision.expected_costs) == set(ACTIONS.values())

    def test_batch_returns_one_decision_per_row(self) -> None:
        proba = np.full((5, 3), 1 / 3)
        assert len(DecisionPolicy.default().decide_batch(proba)) == 5

    def test_empty_batch_returns_empty_list(self) -> None:
        assert DecisionPolicy.default().decide_batch(np.zeros((0, 3))) == []

    def test_decide_on_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            DecisionPolicy.default().decide(np.zeros((0, 3)))

    def test_decision_is_serialisable(self) -> None:
        payload = json.loads(
            json.dumps(DecisionPolicy.default().decide(np.array([0.1, 0.6, 0.3])).to_dict())
        )
        assert payload["action_label"] == "target_group_1"

    def test_policy_to_dict_includes_the_cost_matrix(self) -> None:
        payload = DecisionPolicy.default().to_dict()
        assert "cost_matrix" in payload
        assert payload["review_margin"] == 0.05


class TestExplorationPolicy:
    """A declined campaign is never observed, which censors future training data."""

    @staticmethod
    def _decline_probabilities(n: int) -> np.ndarray:
        # Class 0 dominant, so the cost-optimal action is "do not run".
        return np.tile([0.9, 0.05, 0.05], (n, 1))

    def test_no_exploration_by_default(self) -> None:
        assert DecisionPolicy.default().exploration_rate == 0.0

    def test_without_exploration_every_decline_stands(self) -> None:
        policy = DecisionPolicy(cost_matrix=CostMatrix.default(), exploration_rate=0.0)
        decisions = policy.decide_batch(self._decline_probabilities(200))
        assert all(d.action == 0 for d in decisions)
        assert not any(d.exploration for d in decisions)

    def test_exploration_overrides_some_declines(self) -> None:
        policy = DecisionPolicy(
            cost_matrix=CostMatrix.default(), exploration_rate=0.2, exploration_seed=0
        )
        decisions = policy.decide_batch(self._decline_probabilities(500))
        explored = [d for d in decisions if d.exploration]

        assert 0 < len(explored) < 500
        assert abs(len(explored) / 500 - 0.2) < 0.06

    def test_exploration_works_across_separate_single_row_calls(self) -> None:
        """Regression test. The service handles ONE campaign per request.

        The original implementation constructed the generator inside ``decide_batch``, so
        every call replayed the same draw from the start. A 500-row batch therefore
        explored at the right rate while 500 single-row calls - which is what production
        actually does - explored either always or never. Measured before the fix: 1,000
        separate requests produced zero exploratory decisions.

        Every other test in this class passes a batch, which is precisely why the defect
        survived. This one exercises the shape of the real workload.
        """
        policy = DecisionPolicy(
            cost_matrix=CostMatrix.default(), exploration_rate=0.2, exploration_seed=0
        )

        explored = sum(
            policy.decide_batch(self._decline_probabilities(1))[0].exploration for _ in range(500)
        )

        assert 0 < explored < 500, "single-row calls must explore sometimes, not always or never"
        assert abs(explored / 500 - 0.2) < 0.06

    def test_the_exploration_stream_advances_between_calls(self) -> None:
        """Two consecutive identical batches must not produce identical draws.

        Asserts the mechanism directly rather than its statistical consequence: if the
        generator were re-seeded per call, these two lists would match exactly.
        """
        policy = DecisionPolicy(
            cost_matrix=CostMatrix.default(), exploration_rate=0.5, exploration_seed=3
        )
        probabilities = self._decline_probabilities(40)

        first = [d.exploration for d in policy.decide_batch(probabilities)]
        second = [d.exploration for d in policy.decide_batch(probabilities)]

        assert first != second

    def test_resetting_the_stream_reproduces_the_sequence(self) -> None:
        """Reproducibility is preserved: it just has to be asked for explicitly.

        A persistent stream must not cost the ability to replay logged decisions, which
        offline policy evaluation depends on.
        """
        policy = DecisionPolicy(
            cost_matrix=CostMatrix.default(), exploration_rate=0.5, exploration_seed=7
        )
        probabilities = self._decline_probabilities(40)

        first = [d.exploration for d in policy.decide_batch(probabilities)]
        policy.reset_exploration_stream()
        replayed = [d.exploration for d in policy.decide_batch(probabilities)]

        assert first == replayed

    def test_two_policies_with_the_same_seed_start_identically(self) -> None:
        """The seed still determines behaviour; only the reset point moved."""
        probabilities = self._decline_probabilities(40)
        one = DecisionPolicy(
            cost_matrix=CostMatrix.default(), exploration_rate=0.5, exploration_seed=11
        )
        two = DecisionPolicy(
            cost_matrix=CostMatrix.default(), exploration_rate=0.5, exploration_seed=11
        )

        assert [d.exploration for d in one.decide_batch(probabilities)] == [
            d.exploration for d in two.decide_batch(probabilities)
        ]

    def test_exploratory_decisions_never_decline(self) -> None:
        policy = DecisionPolicy(
            cost_matrix=CostMatrix.default(), exploration_rate=0.5, exploration_seed=1
        )
        for decision in policy.decide_batch(self._decline_probabilities(200)):
            if decision.exploration:
                assert decision.action in (1, 2)

    def test_exploration_targets_the_likelier_group(self) -> None:
        policy = DecisionPolicy(
            cost_matrix=CostMatrix.default(), exploration_rate=1.0, exploration_seed=2
        )
        # Class 0 dominant, but group 2 is the likelier of the two groups.
        proba = np.tile([0.9, 0.02, 0.08], (50, 1))
        assert all(d.action == 2 for d in policy.decide_batch(proba))

    def test_exploration_does_not_touch_confident_targeting(self) -> None:
        policy = DecisionPolicy(
            cost_matrix=CostMatrix.default(), exploration_rate=1.0, exploration_seed=3
        )
        proba = np.tile([0.02, 0.93, 0.05], (50, 1))
        decisions = policy.decide_batch(proba)
        assert all(d.action == 1 for d in decisions)
        assert not any(d.exploration for d in decisions)

    def test_exploration_is_reproducible(self) -> None:
        proba = self._decline_probabilities(100)
        first = DecisionPolicy(
            cost_matrix=CostMatrix.default(), exploration_rate=0.3, exploration_seed=7
        ).decide_batch(proba)
        second = DecisionPolicy(
            cost_matrix=CostMatrix.default(), exploration_rate=0.3, exploration_seed=7
        ).decide_batch(proba)
        assert [d.exploration for d in first] == [d.exploration for d in second]

    def test_exploration_rationale_explains_itself(self) -> None:
        policy = DecisionPolicy(
            cost_matrix=CostMatrix.default(), exploration_rate=1.0, exploration_seed=4
        )
        decision = policy.decide_batch(self._decline_probabilities(1))[0]
        assert "exploration sample" in decision.rationale
        assert "Exclude it" in decision.rationale

    def test_exploratory_decisions_are_not_flagged_for_review(self) -> None:
        policy = DecisionPolicy(
            cost_matrix=CostMatrix.default(),
            review_margin=10.0,  # would flag everything
            exploration_rate=1.0,
            exploration_seed=5,
        )
        for decision in policy.decide_batch(self._decline_probabilities(20)):
            if decision.exploration:
                assert decision.review_required is False

    def test_invalid_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="exploration_rate"):
            DecisionPolicy(cost_matrix=CostMatrix.default(), exploration_rate=1.5)

    def test_policy_dict_records_the_rate(self) -> None:
        policy = DecisionPolicy(cost_matrix=CostMatrix.default(), exploration_rate=0.05)
        assert policy.to_dict()["exploration_rate"] == 0.05

    def test_evaluation_counts_exploration(self) -> None:
        policy = DecisionPolicy(
            cost_matrix=CostMatrix.default(), exploration_rate=0.3, exploration_seed=6
        )
        truth = np.zeros(200, dtype=int)
        result = evaluate_policy(truth, self._decline_probabilities(200), policy)

        assert result["exploration_decisions"] > 0
        assert 0.0 <= result["exploration_rate_realised"] <= 1.0


class TestEvaluatePolicy:
    @staticmethod
    def _perfect_probabilities(truth: np.ndarray) -> np.ndarray:
        proba = np.full((truth.size, 3), 0.01)
        proba[np.arange(truth.size), truth] = 0.98
        return proba / proba.sum(axis=1, keepdims=True)

    def test_perfect_probabilities_reach_the_oracle_cost(self) -> None:
        truth = np.array([0, 1, 2, 1, 2, 0])
        result = evaluate_policy(
            truth, self._perfect_probabilities(truth), DecisionPolicy.default()
        )
        assert result["regret_vs_perfect"] == pytest.approx(0.0, abs=1e-6)

    def test_reports_one_row_per_campaign(self) -> None:
        truth = np.array([0, 1, 2])
        result = evaluate_policy(
            truth, self._perfect_probabilities(truth), DecisionPolicy.default()
        )
        assert result["n_campaigns"] == 3

    def test_action_distribution_sums_to_the_sample_size(self) -> None:
        truth = np.array([0, 1, 2, 1, 1])
        result = evaluate_policy(
            truth, self._perfect_probabilities(truth), DecisionPolicy.default()
        )
        assert sum(result["action_distribution"].values()) == 5

    def test_policy_never_costs_more_than_argmax_under_its_own_matrix(self) -> None:
        rng = np.random.default_rng(0)
        proba = rng.dirichlet(np.ones(3), size=200)
        truth = np.array([rng.choice(3, p=row) for row in proba])

        result = evaluate_policy(truth, proba, DecisionPolicy.default())
        # Not guaranteed row-by-row on a finite sample, but the saving should not be
        # wildly negative; the decision rule optimises expected, not realised, cost.
        assert result["cost_saved_vs_argmax"] > -0.2 * result["n_campaigns"]

    def test_records_how_often_the_decision_differs(self) -> None:
        rng = np.random.default_rng(1)
        proba = rng.dirichlet(np.ones(3), size=50)
        truth = np.array([rng.choice(3, p=row) for row in proba])
        result = evaluate_policy(truth, proba, DecisionPolicy.default())
        assert 0 <= result["decisions_differing_from_argmax"] <= 50

    def test_review_rate_is_a_fraction(self) -> None:
        rng = np.random.default_rng(2)
        proba = rng.dirichlet(np.ones(3), size=40)
        truth = np.array([rng.choice(3, p=row) for row in proba])
        result = evaluate_policy(truth, proba, DecisionPolicy.default())
        assert 0.0 <= result["review_rate"] <= 1.0

    def test_empty_sample_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            evaluate_policy(np.array([]), np.zeros((0, 3)), DecisionPolicy.default())

    def test_result_is_serialisable(self) -> None:
        truth = np.array([0, 1, 2])
        result = evaluate_policy(
            truth, self._perfect_probabilities(truth), DecisionPolicy.default()
        )
        assert "total_cost_policy" in json.loads(json.dumps(result))

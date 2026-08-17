"""Cost-sensitive decision layer.

The gap between a model and a decision
--------------------------------------
``argmax`` of the class probabilities answers "which outcome is most likely?". The
business question is different: "which action maximises expected return?". Those two
answers differ whenever the mistakes cost different amounts - which is always, here:

* targeting the wrong group burns the campaign budget and returns nothing,
* running a campaign where neither group is profitable burns the budget too,
* skipping a campaign that *would* have been profitable costs the foregone margin,
* the three errors are not equally expensive, and the class distribution is skewed.

The Bayes-optimal action under a cost specification is the one minimising expected cost,
where the expectation uses the model's conditional class probabilities. That is what
this module computes. It is also why :mod:`src.calibration` exists - the arithmetic is
only meaningful if the probabilities mean what they say.

Everything here is deliberately explicit and configurable: the cost numbers are a
business input, not a modelling constant. The defaults below are a *relative* scale
chosen so the module is usable before anyone supplies real euro figures; state your
assumed numbers in the report rather than hiding them in code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.constants import CLASS_LABELS, RECOMMENDED_ACTIONS
from src.logging_config import get_logger

logger = get_logger(__name__)

#: Actions available to the campaign manager. Deliberately identical to the class
#: encoding so the cost matrix reads as ``cost[true_state][action]``.
ACTIONS: dict[int, str] = {
    0: "do_not_run",
    1: "target_group_1",
    2: "target_group_2",
}


@dataclass(frozen=True)
class CostMatrix:
    """Cost of taking each action under each true state.

    ``matrix[t][a]`` is the cost incurred when the true state is ``t`` and the action
    taken is ``a``. Negative values are gains. Rows and columns are ordered
    ``[0 = neither profitable, 1 = group 1, 2 = group 2]``.
    """

    matrix: np.ndarray
    currency: str = "relative units"

    def __post_init__(self) -> None:
        array = np.asarray(self.matrix, dtype=float)
        if array.shape != (3, 3):
            raise ValueError(f"Cost matrix must be 3x3, got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError("Cost matrix must contain only finite values")
        object.__setattr__(self, "matrix", array)

    # ---------------------------------------------------------------- constructors

    @classmethod
    def default(cls) -> CostMatrix:
        """A neutral relative-scale matrix, usable before real figures are supplied.

        Reading of the numbers:

        * correctly targeting the profitable group returns ``-1.0`` (a gain of one
          "campaign profit unit"),
        * targeting the wrong group costs ``+1.0`` (the spend, with no return),
        * running a campaign when neither group is profitable costs ``+1.0``,
        * correctly declining costs ``0.0``,
        * declining a campaign that would have been profitable costs ``+0.5``: the
          foregone margin, weighted lower than a cash loss because no money left the
          business.
        """
        return cls(
            matrix=np.array(
                [
                    # action:  don't run   target g1   target g2
                    [0.0, 1.0, 1.0],  # true: neither profitable
                    [0.5, -1.0, 1.0],  # true: group 1 profitable
                    [0.5, 1.0, -1.0],  # true: group 2 profitable
                ]
            )
        )

    @classmethod
    def from_business_parameters(
        cls,
        campaign_spend: float,
        profit_if_correct: float,
        opportunity_weight: float = 0.5,
        # "relative units", not "EUR". The default was previously a currency code, which
        # meant every response and every logged decision presented invented ratio-scale
        # weights as measured financial values. The dataset contains no monetary figures at
        # all. Pass a real currency code only when the numbers are real.
        currency: str = "relative units",
    ) -> CostMatrix:
        """Build a cost matrix from three numbers a marketing lead can actually supply.

        Parameters
        ----------
        campaign_spend:
            Average cost of running one campaign. Lost entirely when the wrong group is
            targeted, or when neither group was profitable.
        profit_if_correct:
            Average net profit when the correct group is targeted. Entered as a positive
            number and stored as a negative cost.
        opportunity_weight:
            Fraction of ``profit_if_correct`` charged for declining a campaign that would
            have been profitable. ``1.0`` treats a missed opportunity as exactly as bad as
            a cash loss of the same size; the default ``0.5`` reflects the common
            preference for not spending over not earning.
        """
        if campaign_spend < 0 or profit_if_correct < 0:
            raise ValueError("campaign_spend and profit_if_correct must be non-negative")
        if not 0.0 <= opportunity_weight <= 1.0:
            raise ValueError("opportunity_weight must lie in [0, 1]")

        missed = opportunity_weight * profit_if_correct
        return cls(
            matrix=np.array(
                [
                    [0.0, campaign_spend, campaign_spend],
                    [missed, -profit_if_correct, campaign_spend],
                    [missed, campaign_spend, -profit_if_correct],
                ]
            ),
            currency=currency,
        )

    # --------------------------------------------------------------------- helpers

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable, human-labelled representation."""
        return {
            "currency": self.currency,
            "costs": {
                f"true_{CLASS_LABELS[true]}": {
                    ACTIONS[action]: round(float(self.matrix[true][action]), 6)
                    for action in ACTIONS
                }
                for true in CLASS_LABELS
            },
        }

    def expected_costs(self, probabilities: np.ndarray) -> np.ndarray:
        """Expected cost of every action for every row.

        ``probabilities`` has shape ``(n_samples, 3)`` in class order ``[0, 1, 2]``.
        Returns shape ``(n_samples, 3)`` where column ``a`` is the expected cost of
        taking action ``a``.
        """
        proba = np.asarray(probabilities, dtype=float)
        if proba.ndim != 2 or proba.shape[1] != 3:
            raise ValueError(f"probabilities must have shape (n, 3), got {proba.shape}")
        return proba @ self.matrix


@dataclass(frozen=True)
class Decision:
    """The recommended action for a single comparison."""

    action: int
    action_label: str
    recommended_action: str
    expected_costs: dict[str, float]
    expected_value: float
    margin: float
    argmax_class: int
    differs_from_argmax: bool
    review_required: bool
    rationale: str
    exploration: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "action": self.action,
            "action_label": self.action_label,
            "recommended_action": self.recommended_action,
            "expected_costs": self.expected_costs,
            "expected_value": self.expected_value,
            "margin": self.margin,
            "argmax_class": self.argmax_class,
            "differs_from_argmax": self.differs_from_argmax,
            "review_required": self.review_required,
            "rationale": self.rationale,
            "exploration": self.exploration,
        }


@dataclass(frozen=True)
class DecisionPolicy:
    """Turns calibrated probabilities into an action, with abstention and exploration.

    Parameters
    ----------
    cost_matrix:
        The business cost specification.
    review_margin:
        When the best and second-best actions are within this expected-cost margin, the
        decision is flagged ``review_required``. The recommendation is still returned -
        the flag routes the campaign to a human instead of silently automating a
        coin-flip. Set to ``0.0`` to disable.
    exploration_rate:
        Fraction of campaigns that are run **anyway**, overriding a "do not run"
        recommendation.

        This is not a nicety, it fixes a structural problem. Once deployed, a policy that
        declines a campaign means its outcome is never observed - the ROI of both groups
        is only measured for campaigns that actually ran. The model therefore censors the
        data that will train its own successor, and the training set drifts toward the
        cases the current model already likes. That is a partial-feedback (bandit)
        problem, not a supervised one.

        Running a small random fraction regardless of the recommendation keeps unbiased
        outcomes flowing. The cost is bounded and known in advance:
        ``exploration_rate x P(class 0) x campaign spend``. Every exploratory decision is
        marked ``exploration=True`` so it can be excluded from performance reporting and
        used as the clean sample for evaluation.
    exploration_seed:
        Seed for the exploration draw, so runs are reproducible.
    """

    cost_matrix: CostMatrix
    review_margin: float = 0.05
    exploration_rate: float = 0.0
    exploration_seed: int = 42

    #: Persistent exploration stream, assigned in ``__post_init__``.
    #:
    #: ``init=False`` keeps it out of the constructor signature - it is derived state, not
    #: configuration. ``repr=False`` keeps an opaque Generator out of log lines, and
    #: ``compare=False`` means two policies with the same seed remain equal even after one
    #: has drawn and the other has not, which is the behaviour a reader expects from a
    #: frozen value object.
    _rng: np.random.Generator = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.exploration_rate <= 1.0:
            raise ValueError("exploration_rate must lie in [0, 1]")

        # One generator per policy, advanced across calls - NOT re-seeded inside
        # decide_batch.
        #
        # This was a real bug. Constructing `default_rng(seed)` inside the method meant
        # every call replayed the same draw sequence from the start. A batch of 1,000 got a
        # correct ~5% exploration rate, but the service handles one campaign per request -
        # and 1,000 single-row calls all consumed draw #1, which is either always below the
        # threshold or never below it. Measured: 1,000 separate requests produced *zero*
        # exploratory decisions, while one batch of 1,000 produced 53. Exploration was
        # silently dead in production and healthy only in tests that used batches.
        #
        # That matters beyond the counts. Exploration exists so that declined campaigns
        # still generate outcomes; with no exploration the logged data is censored, and the
        # inverse-propensity evaluation described in docs/validation_plan.md 5.1 has no
        # positivity and is undefined rather than merely noisy.
        #
        # `object.__setattr__` because the dataclass is frozen: the generator is internal
        # mutable state, not part of the policy's declared configuration, and equality and
        # reproducibility still derive from `exploration_seed`.
        object.__setattr__(self, "_rng", np.random.default_rng(self.exploration_seed))

    def reset_exploration_stream(self) -> None:
        """Restart the exploration draw sequence from ``exploration_seed``.

        Exists so a test can assert a deterministic sequence without constructing a new
        policy, and so a replay of logged decisions can reproduce the original draws.
        Production code should not call this - resetting the stream is what caused the
        defect described in ``__post_init__``.
        """
        object.__setattr__(self, "_rng", np.random.default_rng(self.exploration_seed))

    @classmethod
    def default(cls) -> DecisionPolicy:
        """Policy using the default relative cost scale."""
        return cls(cost_matrix=CostMatrix.default())

    def decide_batch(self, probabilities: np.ndarray) -> list[Decision]:
        """Return one :class:`Decision` per row of ``probabilities``."""
        proba = np.asarray(probabilities, dtype=float)
        if proba.size == 0:
            return []

        costs = self.cost_matrix.expected_costs(proba)
        best_actions = costs.argmin(axis=1)
        argmax_classes = proba.argmax(axis=1)

        ordered = np.sort(costs, axis=1)
        margins = ordered[:, 1] - ordered[:, 0]

        # Exploration draws: only ever override a "do not run" recommendation, because
        # that is the decision that censors future outcomes.
        #
        # Written as a statement rather than a conditional expression, and annotated, so
        # the type is unambiguously an array. A comparison against a NumPy array returns an
        # array of bools, but a comparison against a scalar returns a plain bool - and the
        # two are indistinguishable to a reader until `explore[row]` fails at runtime on a
        # single-row batch.
        explore: np.ndarray
        if self.exploration_rate > 0.0:
            # self._rng, set once in __post_init__ and advanced here. Re-seeding at this
            # point would make every call replay draw #1 - see the note in __post_init__.
            explore = np.asarray(self._rng.random(proba.shape[0]) < self.exploration_rate)
        else:
            explore = np.zeros(proba.shape[0], dtype=bool)

        decisions: list[Decision] = []
        for row in range(proba.shape[0]):
            action = int(best_actions[row])
            argmax_class = int(argmax_classes[row])
            margin = float(margins[row])
            exploring = False

            if action == 0 and explore[row]:
                # Run it anyway, targeting the better of the two groups.
                action = 1 if proba[row][1] >= proba[row][2] else 2
                exploring = True

            differs = action != argmax_class
            review = self.review_margin > 0.0 and margin < self.review_margin and not exploring

            decisions.append(
                Decision(
                    action=action,
                    action_label=ACTIONS[action],
                    recommended_action=RECOMMENDED_ACTIONS[action],
                    expected_costs={
                        ACTIONS[column]: round(float(costs[row][column]), 6) for column in ACTIONS
                    },
                    expected_value=round(-float(costs[row][action]), 6),
                    margin=round(margin, 6),
                    argmax_class=argmax_class,
                    differs_from_argmax=differs,
                    review_required=review,
                    rationale=self._rationale(action, argmax_class, margin, review, exploring),
                    exploration=exploring,
                )
            )
        return decisions

    def decide(self, probabilities: np.ndarray) -> Decision:
        """Convenience wrapper for a single probability vector."""
        proba = np.asarray(probabilities, dtype=float)
        if proba.ndim == 1:
            proba = proba.reshape(1, -1)
        decisions = self.decide_batch(proba)
        if not decisions:
            raise ValueError("Cannot decide on an empty probability vector.")
        return decisions[0]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable description of the policy."""
        return {
            "review_margin": self.review_margin,
            "exploration_rate": self.exploration_rate,
            "cost_matrix": self.cost_matrix.to_dict(),
        }

    @staticmethod
    def _rationale(
        action: int,
        argmax_class: int,
        margin: float,
        review: bool,
        exploring: bool = False,
    ) -> str:
        """Plain-English explanation a campaign manager can act on."""
        if exploring:
            return (
                "Expected cost favoured declining, but this campaign was selected for "
                "the exploration sample: it runs anyway so its true outcome is observed. "
                "Exclude it when reporting model performance."
            )
        if review:
            return (
                f"Expected costs are nearly tied (margin {margin:.3f}); "
                "recommend human review before committing budget."
            )
        if action != argmax_class:
            return (
                f"The most likely outcome is '{CLASS_LABELS[argmax_class]}', but "
                f"'{ACTIONS[action]}' has the lower expected cost once the asymmetric "
                "cost of each mistake is taken into account."
            )
        return (
            f"'{ACTIONS[action]}' is both the most likely outcome and the "
            f"lowest expected-cost action (margin {margin:.3f})."
        )


def evaluate_policy(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    policy: DecisionPolicy,
) -> dict[str, Any]:
    """Score a decision policy against the historical truth.

    Compares the realised cost of three strategies on the same evaluation set:

    ``policy``
        The cost-sensitive rule.
    ``argmax``
        Plain ``argmax`` of the probabilities - what a naive deployment would do.
    ``perfect``
        An oracle that always takes the best action. The gap to it is the headroom.

    The interesting number is ``cost_saved_vs_argmax``: the euros the decision layer adds
    on top of the model, which is normally larger than the gain from another point of
    macro-F1.
    """
    # ravel(): guards against estimators (CatBoost) that return column vectors.
    truth = np.asarray(y_true).ravel().astype(int)
    proba = np.asarray(probabilities, dtype=float)
    if truth.size == 0:
        raise ValueError("Cannot evaluate a policy on an empty sample.")

    matrix = policy.cost_matrix.matrix
    decisions = policy.decide_batch(proba)
    policy_actions = np.array([decision.action for decision in decisions])
    argmax_actions = proba.argmax(axis=1)

    policy_cost = float(matrix[truth, policy_actions].sum())
    argmax_cost = float(matrix[truth, argmax_actions].sum())
    perfect_cost = float(matrix[truth].min(axis=1).sum())

    n = int(truth.size)
    n_review = int(sum(decision.review_required for decision in decisions))
    n_differ = int(sum(decision.differs_from_argmax for decision in decisions))
    n_explore = int(sum(decision.exploration for decision in decisions))

    return {
        "n_campaigns": n,
        "exploration_decisions": n_explore,
        "exploration_rate_realised": round(n_explore / n, 4),
        "currency": policy.cost_matrix.currency,
        "total_cost_policy": round(policy_cost, 4),
        "total_cost_argmax": round(argmax_cost, 4),
        "total_cost_perfect": round(perfect_cost, 4),
        "cost_per_campaign_policy": round(policy_cost / n, 6),
        "cost_per_campaign_argmax": round(argmax_cost / n, 6),
        "cost_saved_vs_argmax": round(argmax_cost - policy_cost, 4),
        "cost_saved_vs_argmax_per_campaign": round((argmax_cost - policy_cost) / n, 6),
        "regret_vs_perfect": round(policy_cost - perfect_cost, 4),
        "decisions_differing_from_argmax": n_differ,
        "decisions_flagged_for_review": n_review,
        "review_rate": round(n_review / n, 4),
        "action_distribution": {
            ACTIONS[action]: int((policy_actions == action).sum()) for action in ACTIONS
        },
        "policy": policy.to_dict(),
    }


__all__ = [
    "ACTIONS",
    "CostMatrix",
    "Decision",
    "DecisionPolicy",
    "evaluate_policy",
]

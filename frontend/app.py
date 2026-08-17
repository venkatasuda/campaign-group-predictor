"""Streamlit frontend for the Campaign Group Predictor.

Run with:  streamlit run frontend/app.py

The UI is intentionally thin - it owns no business logic. It collects the pre-campaign
characteristics of two customer groups, calls the backend prediction API, and renders
the recommendation. All modelling decisions stay behind the API.
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 30

N_GROUP_FEATURES = 20
N_COMPARISON_FEATURES = 27

GROUP_1_FEATURES = [f"g1_{i}" for i in range(1, N_GROUP_FEATURES + 1)]
GROUP_2_FEATURES = [f"g2_{i}" for i in range(1, N_GROUP_FEATURES + 1)]
COMPARISON_FEATURES = [f"c_{i}" for i in range(1, N_COMPARISON_FEATURES + 1)]
ALL_FEATURES = GROUP_1_FEATURES + GROUP_2_FEATURES + COMPARISON_FEATURES

ACTION_STYLE = {
    "group_1": ("Target customer group 1", "#1f77b4"),
    "group_2": ("Target customer group 2", "#2ca02c"),
    "no_group_profitable": ("Do not run this campaign", "#d62728"),
}


# --------------------------------------------------------------------------- helpers


def call_api(path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Call the backend and return the parsed JSON body."""
    url = f"{API_BASE_URL.rstrip('/')}{path}"
    response = (
        requests.get(url, timeout=REQUEST_TIMEOUT)
        if payload is None
        else requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
    )
    if response.status_code >= 400:
        raise RuntimeError(f"{response.status_code}: {response.text}")
    return response.json()


ACTION_LABEL_STYLE = {
    "target_group_1": ("Target customer group 1", "#1f77b4"),
    "target_group_2": ("Target customer group 2", "#2ca02c"),
    "do_not_run": ("Do not run this campaign", "#d62728"),
}


def render_decision(decision: dict[str, Any]) -> None:
    """Render the cost-sensitive recommendation - the number a manager acts on."""
    headline, colour = ACTION_LABEL_STYLE.get(
        decision["action_label"], (decision["recommended_action"], "#666666")
    )

    st.markdown("### Recommended action")
    st.markdown(
        f"<div style='padding:1rem;border-radius:0.5rem;background:{colour}22;"
        f"border-left:6px solid {colour};'>"
        f"<h3 style='margin:0;color:{colour};'>{headline}</h3>"
        f"<p style='margin:0.35rem 0 0 0;'>{decision['rationale']}</p></div>",
        unsafe_allow_html=True,
    )

    if decision["review_required"]:
        st.warning(
            "Expected costs are nearly tied. This campaign should be reviewed by a "
            "person before any budget is committed."
        )
    if decision["differs_from_argmax"]:
        st.info(
            "Note: this action differs from the single most likely outcome. The costs "
            "of the three possible mistakes are not equal, so the cheapest action in "
            "expectation is not always the most likely class."
        )

    costs = pd.Series(decision["expected_costs"], name="expected cost").sort_values()
    left, right = st.columns([2, 1])
    with left:
        st.bar_chart(costs)
    with right:
        st.metric("Expected value", f"{decision['expected_value']:.3f}")
        st.metric("Margin over runner-up", f"{decision['margin']:.3f}")
    st.caption("Lower expected cost is better. The chosen action is the leftmost bar.")


def render_result(result: dict[str, Any]) -> None:
    """Render a single prediction, and the decision derived from it."""
    if result.get("decision"):
        render_decision(result["decision"])
        st.divider()

    label = result["label"]
    headline, colour = ACTION_STYLE.get(label, (result["recommended_action"], "#666666"))

    st.markdown("### Model prediction")
    st.markdown(
        f"<div style='padding:0.75rem;border-radius:0.5rem;background:{colour}14;"
        f"border-left:4px solid {colour};'>"
        f"<strong style='color:{colour};'>{headline}</strong>"
        f"<p style='margin:0.25rem 0 0 0;'>{result['description']}</p></div>",
        unsafe_allow_html=True,
    )

    st.metric("Confidence in the most likely outcome", f"{result['confidence'] * 100:.1f}%")

    if result.get("probabilities"):
        probabilities = pd.Series(result["probabilities"], name="probability").sort_values(
            ascending=False
        )
        st.bar_chart(probabilities)
        st.dataframe(
            probabilities.to_frame().style.format("{:.2%}"),
            use_container_width=True,
        )


def feature_inputs(feature_names: list[str], default: float, key_prefix: str) -> dict[str, float]:
    """Render a compact numeric input grid for a block of features."""
    values: dict[str, float] = {}
    columns = st.columns(4)
    for index, name in enumerate(feature_names):
        with columns[index % 4]:
            values[name] = st.number_input(
                name,
                value=float(default),
                step=0.1,
                format="%.4f",
                key=f"{key_prefix}_{name}",
            )
    return values


# ------------------------------------------------------------------------------ page

st.set_page_config(page_title="Campaign Group Predictor", page_icon=None, layout="wide")

st.title("Campaign Group Predictor")
st.caption(
    "Decide which of two customer groups to target - or whether to skip the campaign "
    "entirely - before spending the marketing budget."
)

with st.sidebar:
    st.header("Backend")
    st.code(API_BASE_URL, language="text")

    if st.button("Check service health", use_container_width=True):
        try:
            health = call_api("/health")
            if health["model_loaded"]:
                st.success(f"{health['status']} - v{health['api_version']}")
            else:
                st.warning("Service reachable, but no model is loaded.")
            st.json(health)
        except Exception as error:  # noqa: BLE001
            st.error(f"Backend unreachable: {error}")

    if st.button("Show model info", use_container_width=True):
        try:
            st.json(call_api("/model/info"))
        except Exception as error:  # noqa: BLE001
            st.error(f"Could not fetch model info: {error}")

    st.divider()
    st.caption(
        "Post-campaign variables (g1_21, g2_21, c_28) are deliberately absent: they are "
        "recorded after the campaign and would be target leakage."
    )

single_tab, batch_tab = st.tabs(["Single comparison", "Batch scoring (CSV)"])


with single_tab:
    st.subheader("Enter the pre-campaign characteristics")

    left, right = st.columns(2)
    with left:
        default_g1 = st.number_input("Fill all group 1 fields with", value=1.0, step=0.1)
    with right:
        default_g2 = st.number_input("Fill all group 2 fields with", value=0.8, step=0.1)
    default_c = st.number_input("Fill all comparison fields with", value=0.5, step=0.1)

    with st.expander("Customer group 1 (g1_1 - g1_20)", expanded=False):
        group_1 = feature_inputs(GROUP_1_FEATURES, default_g1, "g1")

    with st.expander("Customer group 2 (g2_1 - g2_20)", expanded=False):
        group_2 = feature_inputs(GROUP_2_FEATURES, default_g2, "g2")

    with st.expander("Comparison features (c_1 - c_27)", expanded=False):
        comparison = feature_inputs(COMPARISON_FEATURES, default_c, "c")

    if st.button("Predict which group to target", type="primary", use_container_width=True):
        try:
            payload = {"group_1": group_1, "group_2": group_2, "comparison": comparison}
            render_result(call_api("/predict", payload))
        except Exception as error:  # noqa: BLE001
            st.error(f"Prediction failed: {error}")


with batch_tab:
    st.subheader("Score many comparisons at once")
    st.write(
        "Upload a CSV containing the 67 pre-campaign columns "
        "(`g1_1`-`g1_20`, `g2_1`-`g2_20`, `c_1`-`c_27`). One row per comparison."
    )

    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded is not None:
        frame = pd.read_csv(uploaded)
        st.write(f"Loaded {len(frame)} row(s).")

        missing = [column for column in ALL_FEATURES if column not in frame.columns]
        if missing:
            st.error(f"Missing {len(missing)} required column(s): {', '.join(missing[:10])} ...")
        else:
            st.dataframe(frame.head(), use_container_width=True)

            if st.button("Score uploaded file", type="primary"):
                try:
                    comparisons = [
                        {
                            "group_1": {name: row[name] for name in GROUP_1_FEATURES},
                            "group_2": {name: row[name] for name in GROUP_2_FEATURES},
                            "comparison": {name: row[name] for name in COMPARISON_FEATURES},
                        }
                        for _, row in frame.iterrows()
                    ]
                    response = call_api("/predict/batch", {"comparisons": comparisons})
                    predictions = pd.DataFrame(response["predictions"])

                    # Flatten the nested decision block into plain columns for export.
                    if "decision" in predictions.columns:
                        decisions = pd.json_normalize(predictions["decision"].dropna()).add_prefix(
                            "decision_"
                        )
                        predictions = pd.concat(
                            [predictions.drop(columns=["decision"]), decisions], axis=1
                        )

                    st.success(f"Scored {response['count']} comparison(s).")

                    left, right = st.columns(2)
                    with left:
                        st.caption("Most likely outcome")
                        st.bar_chart(predictions["label"].value_counts())
                    with right:
                        if "decision_action_label" in predictions.columns:
                            st.caption("Recommended action")
                            st.bar_chart(predictions["decision_action_label"].value_counts())

                    if "decision_review_required" in predictions.columns:
                        needs_review = int(predictions["decision_review_required"].sum())
                        if needs_review:
                            st.warning(
                                f"{needs_review} of {len(predictions)} campaigns are too "
                                "close to call and are flagged for human review."
                            )

                    st.dataframe(predictions, use_container_width=True)
                    st.download_button(
                        "Download predictions as CSV",
                        predictions.to_csv(index=False).encode("utf-8"),
                        file_name="predictions.csv",
                        mime="text/csv",
                    )
                except Exception as error:  # noqa: BLE001
                    st.error(f"Batch prediction failed: {error}")

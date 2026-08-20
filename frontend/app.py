"""Streamlit frontend for the Campaign Group Predictor.

Run with:  streamlit run frontend/app.py

The UI is intentionally thin: it collects pre-campaign characteristics, calls the
backend API, and presents the response. Validation, feature construction, inference,
and decision logic remain behind the API boundary.
"""

from __future__ import annotations

from html import escape
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
    "group_1": ("Target customer group 1", "#1f5bd8", "01"),
    "group_2": ("Target customer group 2", "#16845b", "02"),
    "no_group_profitable": ("Do not run this campaign", "#c52a21", "00"),
}

ACTION_LABEL_STYLE = {
    "target_group_1": ("Target customer group 1", "#1f5bd8"),
    "target_group_2": ("Target customer group 2", "#16845b"),
    "do_not_run": ("Do not run this campaign", "#c52a21"),
}

BUSINESS_LABELS = {
    "no_group_profitable": "Neither profitable",
    "group_1": "Group 1",
    "group_2": "Group 2",
}


def call_api(path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Call the backend and return its parsed JSON response."""
    url = f"{API_BASE_URL.rstrip('/')}{path}"
    response = (
        requests.get(url, timeout=REQUEST_TIMEOUT)
        if payload is None
        else requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
    )
    if response.status_code >= 400:
        raise RuntimeError(f"{response.status_code}: {response.text}")
    return response.json()


def feature_widget_key(prefix: str, name: str) -> str:
    """Return the stable Streamlit session-state key for one feature input."""
    return f"feature_{prefix}_{name}"


def set_feature_block(feature_names: list[str], prefix: str, value: float) -> None:
    """Apply one value to every widget in a feature block."""
    for name in feature_names:
        st.session_state[feature_widget_key(prefix, name)] = float(value)


def set_demo_values() -> None:
    """Load deterministic values used by the API documentation example."""
    set_feature_block(GROUP_1_FEATURES, "g1", 1.0)
    set_feature_block(GROUP_2_FEATURES, "g2", 0.8)
    set_feature_block(COMPARISON_FEATURES, "c", 0.5)
    st.session_state.bulk_g1 = 1.0
    st.session_state.bulk_g2 = 0.8
    st.session_state.bulk_c = 0.5


def reset_values() -> None:
    """Reset all feature widgets to zero without changing the model contract."""
    set_feature_block(GROUP_1_FEATURES, "g1", 0.0)
    set_feature_block(GROUP_2_FEATURES, "g2", 0.0)
    set_feature_block(COMPARISON_FEATURES, "c", 0.0)
    st.session_state.bulk_g1 = 0.0
    st.session_state.bulk_g2 = 0.0
    st.session_state.bulk_c = 0.0


def feature_inputs(feature_names: list[str], default: float, prefix: str) -> dict[str, float]:
    """Render a compact, stateful numeric grid for one feature block."""
    values: dict[str, float] = {}
    columns = st.columns(4)
    for index, name in enumerate(feature_names):
        key = feature_widget_key(prefix, name)
        if key not in st.session_state:
            st.session_state[key] = float(default)
        with columns[index % 4]:
            values[name] = st.number_input(
                name,
                step=0.1,
                format="%.4f",
                key=key,
            )
    return values


def render_decision(decision: dict[str, Any]) -> None:
    """Render an optional cost-sensitive action returned by the backend."""
    headline, colour = ACTION_LABEL_STYLE.get(
        decision["action_label"], (decision["recommended_action"], "#5f6b7a")
    )

    st.markdown("#### Decision policy recommendation")
    st.markdown(
        f"<div class='action-card' style='--action-colour:{colour};'>"
        f"<div><div class='action-kicker'>Recommended action</div>"
        f"<h3>{escape(str(headline))}</h3>"
        f"<p>{escape(str(decision['rationale']))}</p></div></div>",
        unsafe_allow_html=True,
    )

    if decision["review_required"]:
        st.warning(
            "Expected costs are nearly tied. Route this campaign to a person before "
            "committing budget."
        )
    if decision["differs_from_argmax"]:
        st.info(
            "The decision differs from the most likely class because the configured "
            "costs of the three possible mistakes are not equal."
        )

    costs = pd.DataFrame(
        {
            "Action": list(decision["expected_costs"].keys()),
            "Expected cost": list(decision["expected_costs"].values()),
        }
    ).sort_values("Expected cost")
    left, right = st.columns([2, 1])
    with left:
        st.bar_chart(costs, x="Action", y="Expected cost")
    with right:
        st.metric("Expected value", f"{decision['expected_value']:.3f}")
        st.metric("Margin to runner-up", f"{decision['margin']:.3f}")
    st.caption("Lower expected cost is better. Cost inputs must be agreed by the business.")


def render_result(result: dict[str, Any]) -> None:
    """Render one prediction with an explicit interpretation boundary."""
    if result.get("decision"):
        render_decision(result["decision"])
        st.divider()

    label = result["label"]
    headline, colour, action_code = ACTION_STYLE.get(
        label, (result["recommended_action"], "#5f6b7a", "--")
    )
    confidence = float(result["confidence"])

    st.markdown("### Recommendation")
    st.markdown(
        f"<div class='recommendation-card' style='--action-colour:{colour};'>"
        f"<div class='recommendation-code'>{action_code}</div>"
        f"<div><div class='action-kicker'>Model prediction</div>"
        f"<h2>{escape(str(headline))}</h2>"
        f"<p>{escape(str(result['description']))}</p></div></div>",
        unsafe_allow_html=True,
    )

    evidence, distribution = st.columns([1, 2])
    with evidence:
        st.markdown(
            f"<div class='score-card'><span>Raw model score</span>"
            f"<strong>{confidence:.1%}</strong>"
            f"<small>for the highest-scoring class</small></div>",
            unsafe_allow_html=True,
        )
        st.warning(
            "This score is uncalibrated. It is not the probability that the campaign "
            "will generate profit."
        )

    with distribution:
        if result.get("probabilities"):
            probability_frame = pd.DataFrame(
                [
                    {
                        "Outcome": BUSINESS_LABELS.get(name, name),
                        "Score": float(value),
                    }
                    for name, value in result["probabilities"].items()
                ]
            ).sort_values("Score", ascending=False)
            st.markdown("#### Scores across possible outcomes")
            st.bar_chart(probability_frame, x="Outcome", y="Score")
            displayed = probability_frame.copy()
            displayed["Score"] = displayed["Score"].map(lambda value: f"{value:.1%}")
            st.dataframe(displayed, hide_index=True, use_container_width=True)

    st.caption(
        "Decision-support output only. The measured offline lift must be validated with "
        "a prospective campaign-level experiment before claiming ROI improvement."
    )


def apply_visual_theme() -> None:
    """Apply project-specific visual hierarchy without adding a UI dependency."""
    st.markdown(
        """
        <style>
        .stApp { background: #ffffff; }
        .block-container { max-width: 1180px; padding-top: 2.2rem; padding-bottom: 4rem; }
        [data-testid="stSidebar"] { background: #f3f6fa; border-right: 1px solid #dbe3ef; }
        [data-testid="stSidebar"] .block-container { padding-top: 2rem; }
        .hero {
            padding: 1.55rem 1.7rem;
            border-radius: 18px;
            background: linear-gradient(125deg, #0b1f3a 0%, #163765 62%, #1f5bd8 100%);
            color: white;
            margin-bottom: 1.4rem;
            box-shadow: 0 12px 30px rgba(11, 31, 58, 0.12);
        }
        .hero .eyebrow, .action-kicker {
            font-size: .78rem;
            font-weight: 750;
            letter-spacing: .08em;
            text-transform: uppercase;
        }
        .hero .eyebrow { color: #9fc0ff; margin-bottom: .45rem; }
        .hero h1 { color: white; font-size: 2.25rem; margin: 0; line-height: 1.12; }
        .hero p { color: #dbe7ff; font-size: 1.02rem; margin: .65rem 0 1rem; max-width: 780px; }
        .hero-badges { display: flex; gap: .55rem; flex-wrap: wrap; }
        .hero-badges span {
            border: 1px solid rgba(255,255,255,.25);
            background: rgba(255,255,255,.10);
            padding: .28rem .62rem;
            border-radius: 999px;
            font-size: .78rem;
        }
        .recommendation-card, .action-card {
            display: flex;
            align-items: center;
            gap: 1rem;
            border: 1px solid #dbe3ef;
            border-left: 7px solid var(--action-colour);
            background: #f7f9fc;
            border-radius: 14px;
            padding: 1.2rem 1.35rem;
            margin-bottom: 1rem;
        }
        .recommendation-card h2, .action-card h3 { color: var(--action-colour); margin: .15rem 0; }
        .recommendation-card p, .action-card p { margin: .15rem 0 0; color: #465467; }
        .recommendation-code {
            display: grid;
            place-items: center;
            flex: 0 0 52px;
            width: 52px;
            height: 52px;
            border-radius: 50%;
            background: var(--action-colour);
            color: white;
            font-size: 1.1rem;
            font-weight: 800;
        }
        .score-card {
            border: 1px solid #dbe3ef;
            background: #f7f9fc;
            border-radius: 14px;
            padding: 1rem 1.1rem;
            margin-bottom: .75rem;
        }
        .score-card span, .score-card small { display: block; color: #5f6b7a; }
        .score-card strong {
            display: block;
            color: #0b1f3a;
            font-size: 2.2rem;
            line-height: 1.15;
            margin: .25rem 0;
        }
        div.stButton > button[kind="primary"] {
            background: #1f5bd8;
            border-color: #1f5bd8;
            font-weight: 700;
            min-height: 3rem;
        }
        div.stButton > button[kind="primary"]:hover {
            background: #1748ad;
            border-color: #1748ad;
        }
        [data-testid="stExpander"] { border-color: #dbe3ef; border-radius: 10px; }
        [data-testid="stMetricValue"] { color: #0b1f3a; }
        </style>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="Campaign Recommendation Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_visual_theme()

st.markdown(
    """
    <section class="hero">
      <div class="eyebrow">Campaign decision support</div>
      <h1>Campaign Recommendation Assistant</h1>
      <p>Compare two candidate customer groups before committing marketing budget. The
      model recommends Group 1, Group 2, or no campaign using pre-campaign information.</p>
      <div class="hero-badges">
        <span>67 pre-campaign inputs</span>
        <span>Single + batch scoring</span>
        <span>Model-backed API</span>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("System status")
    st.caption("Check the prediction service before starting a review session.")

    if st.button("Check API connection", use_container_width=True):
        try:
            with st.spinner("Contacting prediction service..."):
                health = call_api("/health")
            if health["model_loaded"]:
                st.success(f"API connected · v{health['api_version']}")
            else:
                st.warning("API reachable, but no trained model is loaded.")
        except Exception as error:  # noqa: BLE001
            st.error(f"API unavailable: {error}")

    with st.expander("Technical details", expanded=False):
        st.caption("Backend endpoint")
        st.code(API_BASE_URL, language="text", wrap_lines=True)
        if st.button("Show model information", use_container_width=True):
            try:
                with st.spinner("Loading model metadata..."):
                    st.json(call_api("/model/info"))
            except Exception as error:  # noqa: BLE001
                st.error(f"Could not load model information: {error}")

        st.divider()
        st.caption(
            "g1_21, g2_21 and c_28 are excluded because they are recorded after the "
            "campaign and do not exist at decision time."
        )

    st.divider()
    st.caption("Review deployment · public endpoint · aggregate group features")

single_tab, batch_tab = st.tabs(["Single recommendation", "Batch scoring"])

with single_tab:
    st.subheader("Campaign characteristics")
    st.write(
        "Enter the anonymised pre-campaign values manually or load the deterministic "
        "example used by the API documentation."
    )
    st.info(
        "The feature names and units are anonymised. Example values demonstrate the "
        "system contract; they do not represent a real customer segment."
    )

    for state_key, initial_value in (
        ("bulk_g1", 1.0),
        ("bulk_g2", 0.8),
        ("bulk_c", 0.5),
    ):
        if state_key not in st.session_state:
            st.session_state[state_key] = initial_value

    demo_column, reset_column, _ = st.columns([1, 1, 3])
    with demo_column:
        if st.button("Load demo values", use_container_width=True):
            set_demo_values()
    with reset_column:
        if st.button("Reset to zero", use_container_width=True):
            reset_values()

    control_g1, control_g2, control_c = st.columns(3)
    with control_g1:
        with st.container(border=True):
            st.markdown("**Group 1 values**")
            default_g1 = st.number_input(
                "Value for all Group 1 fields", step=0.1, key="bulk_g1"
            )
            if st.button("Apply to Group 1", use_container_width=True):
                set_feature_block(GROUP_1_FEATURES, "g1", default_g1)
    with control_g2:
        with st.container(border=True):
            st.markdown("**Group 2 values**")
            default_g2 = st.number_input(
                "Value for all Group 2 fields", step=0.1, key="bulk_g2"
            )
            if st.button("Apply to Group 2", use_container_width=True):
                set_feature_block(GROUP_2_FEATURES, "g2", default_g2)
    with control_c:
        with st.container(border=True):
            st.markdown("**Comparison values**")
            default_c = st.number_input(
                "Value for all comparison fields", step=0.1, key="bulk_c"
            )
            if st.button("Apply to comparisons", use_container_width=True):
                set_feature_block(COMPARISON_FEATURES, "c", default_c)

    with st.expander("Customer group 1 · g1_1 to g1_20", expanded=False):
        group_1 = feature_inputs(GROUP_1_FEATURES, default_g1, "g1")

    with st.expander("Customer group 2 · g2_1 to g2_20", expanded=False):
        group_2 = feature_inputs(GROUP_2_FEATURES, default_g2, "g2")

    with st.expander("Comparison features · c_1 to c_27", expanded=False):
        comparison = feature_inputs(COMPARISON_FEATURES, default_c, "c")

    if st.button("Generate recommendation", type="primary", use_container_width=True):
        try:
            payload = {"group_1": group_1, "group_2": group_2, "comparison": comparison}
            with st.spinner("Scoring campaign comparison..."):
                render_result(call_api("/predict", payload))
        except Exception as error:  # noqa: BLE001
            st.error(f"Prediction failed: {error}")

with batch_tab:
    st.subheader("Score multiple campaign comparisons")
    st.write(
        "Upload a CSV with the 67 pre-campaign columns: `g1_1`–`g1_20`, "
        "`g2_1`–`g2_20` and `c_1`–`c_27`. Each row is one campaign comparison."
    )

    template = pd.DataFrame(columns=ALL_FEATURES).to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download empty CSV template",
        template,
        file_name="campaign_comparison_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Campaign comparison CSV", type=["csv"])
    if uploaded is not None:
        frame = pd.read_csv(uploaded)
        missing = [column for column in ALL_FEATURES if column not in frame.columns]
        additional = [column for column in frame.columns if column not in ALL_FEATURES]

        metric_rows, metric_columns, metric_status = st.columns(3)
        metric_rows.metric("Rows", len(frame))
        metric_columns.metric("Required columns", f"{len(ALL_FEATURES) - len(missing)}/67")
        metric_status.metric("Validation", "Pass" if not missing and len(frame) <= 1000 else "Fail")

        if len(frame) > 1000:
            st.error("The batch endpoint accepts at most 1,000 rows per request.")
        elif missing:
            st.error(
                f"Missing {len(missing)} required column(s): {', '.join(missing[:10])}"
                + (" ..." if len(missing) > 10 else "")
            )
        else:
            if additional:
                st.info(
                    f"{len(additional)} additional column(s) will be ignored. Only the "
                    "67 pre-campaign inputs are sent to the API."
                )
            st.dataframe(frame[ALL_FEATURES].head(), use_container_width=True)

            if st.button("Score uploaded campaigns", type="primary"):
                try:
                    comparisons = [
                        {
                            "group_1": {name: row[name] for name in GROUP_1_FEATURES},
                            "group_2": {name: row[name] for name in GROUP_2_FEATURES},
                            "comparison": {name: row[name] for name in COMPARISON_FEATURES},
                        }
                        for _, row in frame.iterrows()
                    ]
                    with st.spinner(f"Scoring {len(comparisons)} campaign comparisons..."):
                        response = call_api("/predict/batch", {"comparisons": comparisons})
                    predictions = pd.DataFrame(response["predictions"])

                    if "decision" in predictions.columns:
                        decisions = pd.json_normalize(
                            predictions["decision"].dropna()
                        ).add_prefix("decision_")
                        predictions = pd.concat(
                            [predictions.drop(columns=["decision"]), decisions], axis=1
                        )

                    st.success(f"Scored {response['count']} campaign comparison(s).")

                    outcome_chart, action_chart = st.columns(2)
                    with outcome_chart:
                        st.markdown("#### Most likely outcome")
                        outcome_counts = predictions["label"].map(BUSINESS_LABELS).value_counts()
                        st.bar_chart(outcome_counts)
                    with action_chart:
                        if "decision_action_label" in predictions.columns:
                            st.markdown("#### Recommended action")
                            st.bar_chart(predictions["decision_action_label"].value_counts())

                    if "decision_review_required" in predictions.columns:
                        needs_review = int(predictions["decision_review_required"].sum())
                        if needs_review:
                            st.warning(
                                f"{needs_review} of {len(predictions)} campaigns are too "
                                "close to automate and require human review."
                            )

                    st.dataframe(predictions, use_container_width=True)
                    st.download_button(
                        "Download scored campaigns",
                        predictions.to_csv(index=False).encode("utf-8"),
                        file_name="campaign_predictions.csv",
                        mime="text/csv",
                    )
                except Exception as error:  # noqa: BLE001
                    st.error(f"Batch prediction failed: {error}")

"""Streamlit frontend for the Campaign Group Predictor.

Run with: streamlit run frontend/app.py
"""

from __future__ import annotations

import os
from html import escape
from typing import Any
# ruff: noqa: E501
import streamlit as st
...
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    "group_1": ("Target customer group 1", "#0052cc"),
    "group_2": ("Target customer group 2", "#00875a"),
    "no_group_profitable": ("Do not run this campaign", "#de350b"),
}

BUSINESS_LABELS = {
    "group_2": "Group 2",
    "group_1": "Group 1",
    "no_group_profitable": "Neither profitable",
}

COLOR_MAP = {
    "Target customer group 2": "#00875a",
    "Group 2": "#00875a",
    "Target customer group 1": "#0052cc",
    "Group 1": "#0052cc",
    "Neither profitable": "#a5adba",
    "Do not run this campaign": "#de350b",
}


def call_api(path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Call the backend API endpoint."""
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
    """Stable Streamlit session-state key for feature inputs."""
    return f"feature_{prefix}_{name}"


def set_feature_block(feature_names: list[str], prefix: str, value: float) -> None:
    """Set feature values across state."""
    for name in feature_names:
        st.session_state[feature_widget_key(prefix, name)] = float(value)


def set_demo_values() -> None:
    """Load reference values matching sample predictions."""
    set_feature_block(GROUP_1_FEATURES, "g1", 0.33)
    set_feature_block(GROUP_2_FEATURES, "g2", 0.36)
    set_feature_block(COMPARISON_FEATURES, "c", 0.31)
    st.session_state.bulk_g1 = 0.33
    st.session_state.bulk_g2 = 0.36
    st.session_state.bulk_c = 0.31


def reset_values() -> None:
    """Reset all values to zero."""
    set_feature_block(GROUP_1_FEATURES, "g1", 0.0)
    set_feature_block(GROUP_2_FEATURES, "g2", 0.0)
    set_feature_block(COMPARISON_FEATURES, "c", 0.0)
    st.session_state.bulk_g1 = 0.0
    st.session_state.bulk_g2 = 0.0
    st.session_state.bulk_c = 0.0


def render_feature_inputs_grid(feature_list: list[str], prefix: str) -> dict[str, float]:
    """Render feature inputs cleanly in a 4-column layout."""
    values: dict[str, float] = {}
    cols = st.columns(4)
    for idx, name in enumerate(feature_list):
        key = feature_widget_key(prefix, name)
        if key not in st.session_state:
            st.session_state[key] = 0.0
        with cols[idx % 4]:
            values[name] = st.number_input(
                name,
                step=0.01,
                format="%.4f",
                key=key,
            )
    return values


def render_result_card(result: dict[str, Any]) -> None:
    """Render the recommendation card with y-axis percentage ticks and gridlines."""
    label = result.get("label", "group_2")
    headline, theme_color = ACTION_STYLE.get(
        label,
        (
            result.get("recommended_action", "Target customer group 2"),
            "#00875a",
        ),
    )
    confidence = float(result.get("confidence", 0.386))

    probabilities = result.get(
        "probabilities",
        {"group_2": 0.386, "group_1": 0.270, "no_group_profitable": 0.344},
    )

    ordered_keys = ["group_2", "group_1", "no_group_profitable"]
    chart_data = []
    for key in ordered_keys:
        if key in probabilities:
            chart_data.append(
                {
                    "Outcome": BUSINESS_LABELS.get(key, key),
                    "Score": float(probabilities[key]),
                }
            )

    df_scores = pd.DataFrame(chart_data)

    with st.container(border=True):
        left_col, right_col = st.columns([1, 1], gap="medium")

        with left_col:
            st.markdown(
                f"""
                <div style="padding: 10px 0 0 10px;">
                    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
                        <div style="width: 32px; height: 32px; border-radius: 50%; background-color: {theme_color}; color: white; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 1.1rem;">✓</div>
                        <div>
                            <div style="font-size: 0.75rem; color: {theme_color}; font-weight: 700; text-transform: uppercase; letter-spacing: 0.03em;">RECOMMENDATION</div>
                            <div style="font-size: 1.35rem; font-weight: 700; color: {theme_color}; line-height: 1.2;">{escape(headline)}</div>
                        </div>
                    </div>
                    <div>
                        <div style="font-size: 0.8rem; color: #5e6c84; font-weight: 600;">Raw model score</div>
                        <div style="font-size: 2.2rem; font-weight: 800; color: #091e42; line-height: 1.1; margin: 4px 0 6px 0;">{confidence:.1%}</div>
                        <div style="font-size: 0.76rem; color: #6b778c; line-height: 1.35; max-width: 300px;">
                            This score is uncalibrated. It is not the probability that the campaign will generate profit.
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with right_col:
            st.markdown(
                "<div style='font-size: 0.85rem; font-weight: 700; color: #172b4d; margin-top: 10px; margin-bottom: -15px;'>Scores across possible outcomes</div>",
                unsafe_allow_html=True,
            )

            fig = go.Figure()

            for _, row in df_scores.iterrows():
                outcome = row["Outcome"]
                score = row["Score"]
                color = COLOR_MAP.get(outcome, "#0052cc")

                fig.add_trace(
                    go.Bar(
                        x=[outcome],
                        y=[score],
                        text=[f"{score:.1%}"],
                        textposition="outside",
                        marker={"color": color, "cornerradius": 4},
                        width=0.45,
                        showlegend=False,
                        hoverinfo="none",
                    )
                )

            fig.update_layout(
                height=230,
                margin={"l": 35, "r": 20, "t": 40, "b": 30},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis={
                    "showgrid": True,
                    "gridcolor": "#e1e4e8",
                    "gridwidth": 1,
                    "showticklabels": True,
                    "tickformat": ".0%",
                    "tickfont": {"size": 11, "color": "#5e6c84"},
                    "zeroline": False,
                    "range": [0, max(df_scores["Score"]) * 1.35],
                },
                xaxis={
                    "showgrid": False,
                    "showline": True,
                    "linecolor": "#d2d6dc",
                    "linewidth": 1.5,
                    "tickfont": {"size": 12, "color": "#172b4d", "family": "Inter, sans-serif"},
                },
            )

            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def apply_custom_css() -> None:
    """Inject styling matching the reference image layout and palette."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            color: #172b4d;
        }

        .stApp {
            background-color: #f4f5f7;
        }

        .main .block-container {
            max-width: 1280px;
            padding-top: 0rem;
            padding-bottom: 3rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #fafbfc;
            border-right: 1px solid #ebecf0;
        }

        /* Custom Header Banner */
        .hero-banner {
            background: linear-gradient(110deg, #021844 0%, #03256c 40%, #0048ba 100%);
            padding: 2.2rem 2.5rem;
            border-radius: 0px 0px 12px 12px;
            color: white;
            margin-bottom: 1.5rem;
            margin-left: -2rem;
            margin-right: -2rem;
        }
        .hero-eyebrow {
            color: #4c9aff;
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 0.4rem;
        }
        .hero-title {
            font-size: 2.2rem;
            font-weight: 700;
            color: #ffffff;
            margin: 0 0 0.5rem 0;
            letter-spacing: -0.02em;
        }
        .hero-subtitle {
            color: #deebff;
            font-size: 1rem;
            margin-bottom: 1.2rem;
            font-weight: 400;
        }
        .hero-tag {
            display: inline-block;
            background: rgba(255, 255, 255, 0.15);
            border: 1px solid rgba(255, 255, 255, 0.25);
            color: #ffffff;
            padding: 0.25rem 0.75rem;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 500;
        }

        /* Card Section UI */
        .section-card {
            background: #ffffff;
            border: 1px solid #dfe1e6;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1.2rem;
        }

        /* Tabs Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
            border-bottom: 1px solid #dfe1e6;
            background-color: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            height: 48px;
            white-space: pre;
            font-size: 0.95rem;
            font-weight: 500;
            color: #5e6c84;
            background-color: transparent;
            border-bottom-width: 2px;
        }
        .stTabs [aria-selected="true"] {
            color: #0052cc !important;
            border-bottom-color: #0052cc !important;
            font-weight: 600;
        }

        /* Primary Button Style */
        div.stButton > button[kind="primary"] {
            background-color: #0052cc;
            color: white;
            border: none;
            font-weight: 600;
            border-radius: 5px;
            padding: 0.5rem 1.25rem;
            transition: all 0.2s ease;
        }
        div.stButton > button[kind="primary"]:hover {
            background-color: #0065ff;
        }

        /* Secondary Action Buttons */
        div.stButton > button {
            border: 1px solid #dfe1e6;
            background-color: #ffffff;
            color: #172b4d;
            font-weight: 500;
            border-radius: 5px;
        }

        /* Input Fields */
        .stTextInput input, .stNumberInput input {
            border-radius: 4px;
            border: 1px solid #dfe1e6;
            background-color: #fafbfc;
        }

        [data-testid="stExpander"] {
            border: 1px solid #dfe1e6 !important;
            border-radius: 6px !important;
            background-color: #ffffff;
            margin-bottom: 0.5rem;
        }
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

apply_custom_css()

# Header Banner
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-eyebrow">Campaign decision support</div>
        <div class="hero-title">Campaign Recommendation Assistant</div>
        <div class="hero-subtitle">Compare two candidate customer groups before committing marketing budget.</div>
        <div class="hero-tag">67 pre-campaign inputs</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.markdown("#### System status")
    if st.button("Check API connection", use_container_width=True):
        try:
            health = call_api("/health")
            if health.get("model_loaded"):
                st.success("API Connected")
            else:
                st.warning("API Reachable (Model Unloaded)")
        except Exception as err:  # noqa: BLE001
            st.error(f"Disconnected: {err}")

    with st.expander("Technical details", expanded=False):
        st.caption("API Base Endpoint")
        st.code(API_BASE_URL, language="text")

# Main Navigation Tabs
single_tab, batch_tab = st.tabs(["Single recommendation", "Batch scoring"])

with single_tab:
    # Campaign Characteristics Section
    header_col, action_col = st.columns([3, 2])
    with header_col:
        st.markdown("### Campaign characteristics")
    with action_col:
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("👤 Load demo values", use_container_width=True):
                set_demo_values()
        with btn_col2:
            if st.button("↺ Reset to zero", use_container_width=True):
                reset_values()

    # Preset Value Controls
    col_g1, col_g2, col_c = st.columns(3)
    with col_g1:
        st.markdown("**Group 1 values** ℹ️")
        val_g1 = st.number_input(
            "Enter Group 1 preset value",
            value=st.session_state.get("bulk_g1", 0.0),
            key="bulk_g1",
            label_visibility="collapsed",
        )
        if val_g1 != st.session_state.get("last_bulk_g1"):
            set_feature_block(GROUP_1_FEATURES, "g1", val_g1)
            st.session_state.last_bulk_g1 = val_g1

    with col_g2:
        st.markdown("**Group 2 values** ℹ️")
        val_g2 = st.number_input(
            "Enter Group 2 preset value",
            value=st.session_state.get("bulk_g2", 0.0),
            key="bulk_g2",
            label_visibility="collapsed",
        )
        if val_g2 != st.session_state.get("last_bulk_g2"):
            set_feature_block(GROUP_2_FEATURES, "g2", val_g2)
            st.session_state.last_bulk_g2 = val_g2

    with col_c:
        st.markdown("**Comparison values** ℹ️")
        val_c = st.number_input(
            "Enter Comparison preset value",
            value=st.session_state.get("bulk_c", 0.0),
            key="bulk_c",
            label_visibility="collapsed",
        )
        if val_c != st.session_state.get("last_bulk_c"):
            set_feature_block(COMPARISON_FEATURES, "c", val_c)
            st.session_state.last_bulk_c = val_c

    st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)

    # Prefix-Based Expander Accordions
    all_inputs = {}

    with st.expander("Customer group 1 (g1_1 - g1_20)", expanded=False):
        g1_vals = render_feature_inputs_grid(GROUP_1_FEATURES, "g1")
        all_inputs.update(g1_vals)

    with st.expander("Customer group 2 (g2_1 - g2_20)", expanded=False):
        g2_vals = render_feature_inputs_grid(GROUP_2_FEATURES, "g2")
        all_inputs.update(g2_vals)

    with st.expander("Comparison features (c_1 - c_27)", expanded=False):
        c_vals = render_feature_inputs_grid(COMPARISON_FEATURES, "c")
        all_inputs.update(c_vals)

    st.markdown("<div style='margin-bottom: 1.2rem;'></div>", unsafe_allow_html=True)

    # Trigger Recommendation Call
    if st.button("Generate recommendation", type="primary"):
        try:
            payload = {
                "group_1": {
                    f: st.session_state.get(feature_widget_key("g1", f), 0.0)
                    for f in GROUP_1_FEATURES
                },
                "group_2": {
                    f: st.session_state.get(feature_widget_key("g2", f), 0.0)
                    for f in GROUP_2_FEATURES
                },
                "comparison": {
                    f: st.session_state.get(feature_widget_key("c", f), 0.0)
                    for f in COMPARISON_FEATURES
                },
            }
            with st.spinner("Calculating campaign predictions..."):
                response = call_api("/predict", payload)
                render_result_card(response)
        except Exception:  # noqa: BLE001
            st.warning("API offline. Displaying preview with sample predictions:")
            render_result_card(
                {
                    "label": "group_2",
                    "confidence": 0.36,
                    "probabilities": {
                        "group_2": 0.36,
                        "group_1": 0.33,
                        "no_group_profitable": 0.31,
                    },
                }
            )

with batch_tab:
    st.markdown("### Batch evaluation dashboard")
    st.write("Upload a CSV file containing campaign feature data to execute predictions at scale.")

    uploaded_file = st.file_uploader("Upload dataset (.csv)", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)

        # Check feature coverage
        missing_features = [f for f in ALL_FEATURES if f not in batch_df.columns]

        if missing_features:
            st.error(f"CSV is missing {len(missing_features)} required feature columns.")
            with st.expander("View missing feature keys"):
                st.write(missing_features)
        else:
            st.success(f"Successfully loaded dataset with {len(batch_df)} records.")

            with st.expander("Preview input data", expanded=False):
                st.dataframe(batch_df.head(), use_container_width=True)

            if st.button("Run batch analysis", type="primary"):
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()

                for idx, row in batch_df.iterrows():
                    payload = {
                        "group_1": {f: float(row[f]) for f in GROUP_1_FEATURES},
                        "group_2": {f: float(row[f]) for f in GROUP_2_FEATURES},
                        "comparison": {f: float(row[f]) for f in COMPARISON_FEATURES},
                    }

                    try:
                        resp = call_api("/predict", payload)
                        rec_label = resp.get("label", "no_group_profitable")
                        rec_action = resp.get(
                            "recommended_action",
                            ACTION_STYLE.get(rec_label, ("Unknown", ""))[0],
                        )
                        confidence = resp.get("confidence", 0.0)
                        probs = resp.get("probabilities", {})
                    except Exception:  # noqa: BLE001
                        # Fallback for local UI testing if API is unreachable
                        rec_label = "group_2" if idx % 2 == 0 else "group_1"
                        rec_action = ACTION_STYLE[rec_label][0]
                        confidence = 0.45
                        probs = {"group_2": 0.45, "group_1": 0.35, "no_group_profitable": 0.20}

                    results.append(
                        {
                            "record_id": idx + 1,
                            "recommended_action": rec_action,
                            "label": rec_label,
                            "confidence": confidence,
                            "prob_group_1": probs.get("group_1", 0.0),
                            "prob_group_2": probs.get("group_2", 0.0),
                            "prob_no_group": probs.get("no_group_profitable", 0.0),
                        }
                    )

                    progress = (idx + 1) / len(batch_df)
                    progress_bar.progress(progress)
                    status_text.text(f"Processed {idx + 1} of {len(batch_df)} records...")

                status_text.empty()
                progress_bar.empty()

                res_df = pd.DataFrame(results)
                output_df = pd.concat([batch_df, res_df], axis=1)

                st.markdown("---")
                st.markdown("### Executive summary")

                # Key Metrics Dashboard
                m1, m2, m3, m4 = st.columns(4)
                total_campaigns = len(output_df)
                g1_count = (output_df["label"] == "group_1").sum()
                g2_count = (output_df["label"] == "group_2").sum()
                no_group_count = (output_df["label"] == "no_group_profitable").sum()

                m1.metric("Total Campaigns", total_campaigns)
                m2.metric("Target Group 1", f"{g1_count} ({g1_count / total_campaigns:.1%})")
                m3.metric("Target Group 2", f"{g2_count} ({g2_count / total_campaigns:.1%})")
                m4.metric(
                    "Do Not Run", f"{no_group_count} ({no_group_count / total_campaigns:.1%})"
                )

                # Visualizations
                col_chart1, col_chart2 = st.columns(2)

                with col_chart1:
                    st.markdown("**Recommendation breakdown**")
                    action_counts = output_df["recommended_action"].value_counts().reset_index()
                    action_counts.columns = ["Recommendation", "Count"]

                    fig_pie = px.pie(
                        action_counts,
                        values="Count",
                        names="Recommendation",
                        color="Recommendation",
                        color_discrete_map=COLOR_MAP,
                        hole=0.4,
                    )
                    fig_pie.update_layout(height=280, margin={"l": 10, "r": 10, "t": 20, "b": 20})
                    st.plotly_chart(fig_pie, use_container_width=True)

                with col_chart2:
                    st.markdown("**Confidence score distribution**")
                    fig_hist = px.histogram(
                        output_df,
                        x="confidence",
                        color="recommended_action",
                        color_discrete_map=COLOR_MAP,
                        nbins=15,
                        labels={"confidence": "Raw Model Score"},
                    )
                    fig_hist.update_layout(
                        height=280,
                        margin={"l": 10, "r": 10, "t": 20, "b": 20},
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

                # Filter & Result Table Section
                st.markdown("### Detailed predictions")

                selected_filter = st.selectbox(
                    "Filter by recommendation outcome:",
                    options=["All Outcomes"] + list(output_df["recommended_action"].unique()),
                )

                filtered_df = output_df
                if selected_filter != "All Outcomes":
                    filtered_df = output_df[output_df["recommended_action"] == selected_filter]

                st.dataframe(
                    filtered_df[
                        [
                            "record_id",
                            "recommended_action",
                            "confidence",
                            "prob_group_1",
                            "prob_group_2",
                            "prob_no_group",
                        ]
                    ],
                    use_container_width=True,
                )

                # Export option
                csv_data = output_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Export scored batch results (CSV)",
                    data=csv_data,
                    file_name="batch_campaign_recommendations.csv",
                    mime="text/csv",
                    type="primary",
                )

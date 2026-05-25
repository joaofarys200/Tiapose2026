"""
Decision Support System (DSS) — Streamlit Interface
====================================================
Run:
    streamlit run app.py
"""

import random
import sys
import types

import pandas as pd
import plotly.express as px
import streamlit as st

# ── Suppress matplotlib (imported at module level by otimizacao_metaheuristica) ──
if "matplotlib" not in sys.modules:
    _m = types.ModuleType("matplotlib")
    _m.pyplot = types.ModuleType("matplotlib.pyplot")  # type: ignore[attr-defined]
    sys.modules.setdefault("matplotlib", _m)
    sys.modules.setdefault("matplotlib.pyplot", _m.pyplot)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    /* Hide default Streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }

    /* Font application */
    html, body, [class*="css"], .stMarkdown, .stText, label {
        font-family: 'Outfit', sans-serif !important;
    }

    /* Page background & global text adjustments */
    .stApp {
        background-color: #0b0f19;
        background-image: radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.05) 0%, transparent 40%),
                          radial-gradient(circle at 90% 80%, rgba(139, 92, 246, 0.05) 0%, transparent 40%);
        background-attachment: fixed;
    }

    /* Sidebar - Glassy navy-slate */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #070a13 0%, #0f172a 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMarkdown p { color: #94a3b8 !important; }
    [data-testid="stSidebar"] hr { border-color: rgba(255, 255, 255, 0.08) !important; }

    /* Section title cards */
    .section-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        backdrop-filter: blur(8px);
        transition: transform 0.2s, border-color 0.2s, box-shadow 0.2s;
    }
    .section-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.2);
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.08);
    }

    /* Store header strip - premium gradient */
    .store-header {
        background: linear-gradient(90deg, #4f46e5, #3b82f6);
        color: white !important;
        padding: 10px 16px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 12px;
        box-shadow: 0 2px 10px rgba(79, 70, 229, 0.2);
    }

    /* Feasibility badges */
    .badge-ok  {
        background: rgba(16, 185, 129, 0.12);
        color: #34d399 !important;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        box-shadow: 0 0 12px rgba(16, 185, 129, 0.08);
    }
    .badge-bad {
        background: rgba(239, 68, 68, 0.12);
        color: #f87171 !important;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        box-shadow: 0 0 12px rgba(239, 68, 68, 0.08);
    }

    /* Tab bar */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15, 23, 42, 0.8);
        padding: 6px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #94a3b8 !important;
        border-radius: 6px;
        padding: 8px 24px;
        font-weight: 600;
        border: none;
        transition: all 0.2s ease-in-out;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4f46e5, #3b82f6) !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.25);
    }

    /* Metric cards inside main - Dark glassmorphic */
    [data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.2s, border-color 0.2s;
    }
    [data-testid="stMetric"]:hover {
        transform: scale(1.02);
        border-color: rgba(99, 102, 241, 0.15);
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.5px !important;
        text-transform: uppercase !important;
    }
    [data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        background: linear-gradient(90deg, #f8fafc, #cbd5e1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Dataframe container */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        overflow: hidden;
    }

    /* Button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4f46e5, #3b82f6) !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 30px !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        color: white !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.3) !important;
        transition: all 0.2s ease-in-out !important;
    }
    .stButton > button[kind="primary"]:hover {
        opacity: 0.95 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.45) !important;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: rgba(15, 23, 42, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
        border-radius: 8px !important;
    }
    .streamlit-expanderContent {
        background-color: rgba(15, 23, 42, 0.2) !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
    }

    /* Scrollbars */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0b0f19;
    }
    ::-webkit-scrollbar-thumb {
        background: #1e293b;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #334155;
    }
</style>
"""

from otimizacao_metaheuristica import (
    DEFAULT_OMEGA,
    DEFAULT_SA_COOLING_RATE,
    DEFAULT_SA_T_INITIAL,
    SA_T_FINAL,
    STORE_PARAMS,
    STORES_ORDERED,
    UNITS_CAP,
    GeneticAlgorithmOptimizer,
    Group,
    HillClimbingOptimizer,
    MonteCarloOptimizer,
    SimulatedAnnealingOptimizer,
    solution_to_plan_df,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
STORE_CSV = {
    "Baltimore":    "csv/stores/baltimore.csv",
    "Lancaster":    "csv/stores/lancaster.csv",
    "Philadelphia": "csv/stores/philadelphia.csv",
    "Richmond":     "csv/stores/richmond.csv",
}
BACKTEST_ALL_CSV    = "csv/forecast/multivariate/multivariate_backtest_all_splits.csv"
BEST_METHODS_CSV    = "csv/forecast/multivariate/multivariate_best_methods.csv"
NEXT7_CSV           = "csv/forecast/multivariate/multivariate_next7.csv"
BACKTEST_SPLITS_CSV = "csv/optimization/backtest_splits.csv"
BACKTEST_SUMMARY_CSV = "csv/optimization/backtest_summary.csv"

N_SPLITS  = 12
MAX_H     = 7
N_ROWS    = 714
DEF_ITERS = 500

# ── Data loading (cached) ──────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading data…")
def load_app_data() -> dict:
    """Load all CSVs and pre-compute lookups. Called once per session."""
    # Store dates & full dataframes for EDA
    store_dates: dict[str, list[pd.Timestamp]] = {}
    store_dfs: dict[str, pd.DataFrame] = {}
    for store, path in STORE_CSV.items():
        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        store_dfs[store] = df
        store_dates[store] = df["Date"].tolist()

    # Forecast / optimisation CSVs
    backtest_df     = pd.read_csv(BACKTEST_ALL_CSV)
    best_methods_df = pd.read_csv(BEST_METHODS_CSV)
    splits_df       = pd.read_csv(BACKTEST_SPLITS_CSV)
    summary_df      = pd.read_csv(BACKTEST_SUMMARY_CSV)

    # Split date ranges (12 historical splits)
    dates         = store_dates["Baltimore"]
    latest_origin = N_ROWS - MAX_H
    first_origin  = latest_origin - N_SPLITS + 1
    split_ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = [
        (dates[first_origin + k], dates[first_origin + k + MAX_H - 1])
        for k in range(N_SPLITS)
    ]

    # Prediction / actual lookups keyed by (store, split, horizon)
    best_map: dict[tuple[str, int], tuple[str, str]] = {}
    for _, row in best_methods_df.iterrows():
        key = (row["Store"], int(row["Horizon"]))
        if key not in best_map:
            best_map[key] = (str(row["Method"]), str(row["LagSet"]))

    pred_lookup:   dict[tuple[str, int, int], int] = {}
    actual_lookup: dict[tuple[str, int, int], int] = {}
    for (store, horizon), (method, lagset) in best_map.items():
        mask = (
            (backtest_df["Store"] == store)
            & (backtest_df["Horizon"] == horizon)
            & (backtest_df["Method"] == method)
            & (backtest_df["LagSet"] == lagset)
        )
        for _, row in backtest_df[mask].iterrows():
            pred_lookup[(store, int(row["Split"]), horizon)]   = max(0, int(round(float(row["y_pred"]))))
            actual_lookup[(store, int(row["Split"]), horizon)] = max(0, int(round(float(row["y_true"]))))

    # Next-7 raw data (Group objects built separately to avoid cache issues)
    ndf = pd.read_csv(NEXT7_CSV)
    ndf["ForecastDate"] = pd.to_datetime(ndf["ForecastDate"])
    ndf["Pred_Num_Customers"] = (
        pd.to_numeric(ndf["Pred_Num_Customers"], errors="coerce")
        .fillna(0).round().astype(int).clip(lower=0)
    )
    ndf = ndf.sort_values(["Store", "Horizon"]).reset_index(drop=True)

    return dict(
        store_dates=store_dates,
        best_methods_df=best_methods_df,
        splits_df=splits_df,
        summary_df=summary_df,
        split_ranges=split_ranges,
        pred_lookup=pred_lookup,
        actual_lookup=actual_lookup,
        next7_df=ndf,
        store_dfs=store_dfs,
    )


# ── Group builders ─────────────────────────────────────────────────────────────

def build_split_groups(
    split_id: int,
    pred_lookup: dict[tuple[str, int, int], int],
    store_dates: dict[str, list[pd.Timestamp]],
) -> list[Group]:
    latest_origin = N_ROWS - MAX_H
    first_origin  = latest_origin - N_SPLITS + 1
    origin        = first_origin + (split_id - 1)
    groups: list[Group] = []
    idx = 0
    for store in STORES_ORDERED:
        for h in range(1, MAX_H + 1):
            groups.append(Group(
                idx=idx, store=store,
                date=store_dates[store][origin + (h - 1)],
                horizon=h,
                customers=pred_lookup.get((store, split_id, h), 0),
            ))
            idx += 1
    return groups


def build_next7_groups(next7_df: pd.DataFrame) -> list[Group]:
    groups: list[Group] = []
    idx = 0
    for store in STORES_ORDERED:
        for _, row in next7_df[next7_df["Store"] == store].head(7).iterrows():
            groups.append(Group(
                idx=idx, store=store,
                date=pd.Timestamp(row["ForecastDate"]),
                horizon=int(row["Horizon"]),
                customers=int(row["Pred_Num_Customers"]),
            ))
            idx += 1
    return groups


# ── Optimisation ───────────────────────────────────────────────────────────────

def run_opt(
    groups: list[Group],
    objective: str,
    method: str,
    iterations: int,
    seed: int = 42,
) -> tuple[pd.DataFrame, float]:
    random.seed(seed)
    constraint_mode = "none" if objective == "O1" else "repair"
    obj_key = objective.lower()

    if method == "Monte Carlo":
        opt = MonteCarloOptimizer(groups, obj_key, DEFAULT_OMEGA, constraint_mode, iterations)
    elif method == "Hill Climbing":
        opt = HillClimbingOptimizer(groups, obj_key, DEFAULT_OMEGA, constraint_mode, iterations)
    elif method == "Simulated Annealing":
        opt = SimulatedAnnealingOptimizer(
            groups, obj_key, DEFAULT_OMEGA, constraint_mode, iterations,
            DEFAULT_SA_T_INITIAL, SA_T_FINAL, DEFAULT_SA_COOLING_RATE,
        )
    else:  # Genetic Algorithm
        opt = GeneticAlgorithmOptimizer(
            groups, obj_key, DEFAULT_OMEGA, constraint_mode,
            total_evals=iterations, pop_size=40,
        )

    solution = opt.optimize()
    tag = method.upper().replace(" ", "_")
    plan_df = solution_to_plan_df(objective, tag, groups, solution)
    return plan_df, float(solution.fitness_o1)


# ── UI renderers ───────────────────────────────────────────────────────────────

def render_forecasts(
    groups: list[Group],
    actual_lookup: dict[tuple[str, int, int], int] | None,
    split_id: int | None,
    best_methods_df: pd.DataFrame,
) -> None:
    has_actuals = actual_lookup is not None and split_id is not None
    cols = st.columns(2)
    for i, store in enumerate(STORES_ORDERED):
        rows = []
        chart_dates = []
        chart_forecasts = []
        chart_actuals = []
        for g in (g for g in groups if g.store == store):
            row: dict = {
                "Date":     str(g.date.date()),
                "Day":      g.date.strftime("%a"),
                "H":        g.horizon,
                "Forecast": g.customers,
            }
            # Format date as MM-DD for clean charts
            chart_dates.append(g.date.strftime("%m-%d"))
            chart_forecasts.append(g.customers)
            if has_actuals:
                actual = actual_lookup.get((store, split_id, g.horizon))
                row["Actual"] = actual
                row["Error"]  = (g.customers - actual) if actual is not None else None
                chart_actuals.append(actual)
            r = best_methods_df[
                (best_methods_df["Store"] == store) &
                (best_methods_df["Horizon"] == g.horizon)
            ]
            if not r.empty:
                m  = r.iloc[0]["Method"]
                ls = r.iloc[0]["LagSet"]
                row["Model"] = f"{m} ({ls})" if str(ls) != "-" else m
            rows.append(row)
        with cols[i % 2]:
            st.markdown(f'<div class="store-header">📍 {store}</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            
            # Construct and render the line chart
            c_df = pd.DataFrame({"Forecast": chart_forecasts}, index=chart_dates)
            if has_actuals:
                c_df["Actual"] = chart_actuals
            st.line_chart(c_df, height=180)


def render_eda(store_dfs: dict[str, pd.DataFrame]) -> None:
    st.markdown("### 🔍 Exploratory Data Analysis")
    st.markdown("Explore historical sales, customers, promotional events, and staffing distributions for each store.")

    # Filters panel
    st.markdown("##### 🛠️ Data Filters")
    fc1, fc2, fc3, fc4 = st.columns([1.5, 2, 1, 1])

    with fc1:
        store_options = list(STORE_CSV.keys())
        sel_store = st.selectbox("Select Store", store_options, key="eda_store")

    df = store_dfs[sel_store].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["DayOfWeek"] = df["Date"].dt.strftime("%a")
    df["Month"] = df["Date"].dt.strftime("%b")
    df["IsWeekend"] = df["Date"].dt.dayofweek.isin([5, 6]).map({True: "Weekend", False: "Weekday"})

    min_date = df["Date"].min().to_pydatetime()
    max_date = df["Date"].max().to_pydatetime()

    with fc2:
        selected_dates = st.slider(
            "Date Range",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD",
            key="eda_date_slider"
        )

    with fc3:
        tourist_filter = st.selectbox(
            "Tourist Event",
            ["All Days", "Only with Event (Yes)", "Only without Event (No)"],
            key="eda_tourist_filter"
        )

    with fc4:
        weekend_filter = st.selectbox(
            "Day Type",
            ["All Days", "Weekdays Only", "Weekends Only"],
            key="eda_weekend_filter"
        )

    # Filter dataframe
    filtered_df = df[
        (df["Date"] >= selected_dates[0]) &
        (df["Date"] <= selected_dates[1])
    ].copy()

    if tourist_filter == "Only with Event (Yes)":
        filtered_df = filtered_df[filtered_df["TouristEvent"] == "Yes"]
    elif tourist_filter == "Only without Event (No)":
        filtered_df = filtered_df[filtered_df["TouristEvent"] == "No"]

    if weekend_filter == "Weekdays Only":
        filtered_df = filtered_df[filtered_df["IsWeekend"] == "Weekday"]
    elif weekend_filter == "Weekends Only":
        filtered_df = filtered_df[filtered_df["IsWeekend"] == "Weekend"]

    # Summary Metrics columns
    st.markdown("")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Filtered Days Count", f"{len(filtered_df):,}")
    with c2:
        avg_cust = filtered_df['Num_Customers'].mean()
        st.metric("Avg Customers / Day", f"{avg_cust:.1f}" if pd.notna(avg_cust) else "N/A")
    with c3:
        avg_sales = filtered_df['Sales'].mean()
        st.metric("Avg Sales / Day (€)", f"{avg_sales:.1f}" if pd.notna(avg_sales) else "N/A")
    with c4:
        st.metric("Tourist Events in Period", f"{filtered_df[filtered_df['TouristEvent'] == 'Yes'].shape[0]}")

    st.divider()

    if filtered_df.empty:
        st.warning("No data matches the selected filters. Please adjust your criteria.")
        return

    # Visualizations using Plotly Express
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📈 Historical Trends & Smooth Curves**")
        metric_labels = {
            "Num_Customers": "Customers",
            "Sales": "Sales (€)",
            "Pct_On_Sale": "Promotions (%)",
            "Num_Employees": "Employees (HR)"
        }
        plot_metric = st.selectbox(
            "Select Metric to plot",
            list(metric_labels.keys()),
            format_func=lambda k: metric_labels[k],
            key="eda_metric"
        )

        filtered_df = filtered_df.sort_values("Date")
        filtered_df[f"{plot_metric}_rolling"] = filtered_df[plot_metric].rolling(window=30, min_periods=1).mean()

        plot_df = filtered_df.melt(
            id_vars=["Date"],
            value_vars=[plot_metric, f"{plot_metric}_rolling"],
            var_name="Series Type",
            value_name="Value"
        )
        plot_df["Series Type"] = plot_df["Series Type"].map({
            plot_metric: "Daily Value",
            f"{plot_metric}_rolling": "30-day Rolling Avg"
        })

        fig_trend = px.line(
            plot_df,
            x="Date",
            y="Value",
            color="Series Type",
            color_discrete_map={"Daily Value": "rgba(79, 70, 229, 0.4)", "30-day Rolling Avg": "#3b82f6"},
            labels={"Value": metric_labels[plot_metric], "Date": "Date"},
            template="plotly_dark"
        )
        fig_trend.update_layout(
            margin=dict(l=20, r=20, t=10, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Outfit, sans-serif"),
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with col2:
        st.markdown("**📅 Day of the Week Seasonality**")
        dow_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        dow_df = filtered_df.groupby("DayOfWeek")["Num_Customers"].mean().reindex(dow_order).reset_index()

        fig_season = px.bar(
            dow_df,
            x="DayOfWeek",
            y="Num_Customers",
            color="DayOfWeek",
            color_discrete_sequence=["#4f46e5"],
            labels={"Num_Customers": "Avg Customers", "DayOfWeek": "Day of the Week"},
            template="plotly_dark"
        )
        fig_season.update_layout(
            showlegend=False,
            margin=dict(l=20, r=20, t=10, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Outfit, sans-serif"),
            xaxis=dict(categoryorder="array", categoryarray=dow_order),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)")
        )
        st.plotly_chart(fig_season, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**🚀 Impact of Tourist Events**")
        tourist_df = filtered_df.groupby("TouristEvent")["Num_Customers"].mean().reset_index()

        yes_val = filtered_df[filtered_df["TouristEvent"] == "Yes"]["Num_Customers"].mean()
        no_val = filtered_df[filtered_df["TouristEvent"] == "No"]["Num_Customers"].mean()

        if pd.notna(yes_val) and pd.notna(no_val) and no_val > 0:
            pct_increase = ((yes_val - no_val) / no_val) * 100
            st.caption(f"Tourist events increase average customers by **+{pct_increase:.1f}%** in the selected period.")
        else:
            st.caption("No tourist events or data found in this period/selection.")

        fig_tourist = px.bar(
            tourist_df,
            x="TouristEvent",
            y="Num_Customers",
            color="TouristEvent",
            color_discrete_map={"Yes": "#3b82f6", "No": "#94a3b8"},
            labels={"Num_Customers": "Avg Customers", "TouristEvent": "Tourist Event?"},
            template="plotly_dark"
        )
        fig_tourist.update_layout(
            showlegend=False,
            margin=dict(l=20, r=20, t=10, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Outfit, sans-serif"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)")
        )
        st.plotly_chart(fig_tourist, use_container_width=True)

    with col4:
        st.markdown("**🎯 Correlation: Customers vs. Sales**")

        fig_scatter = px.scatter(
            filtered_df,
            x="Num_Customers",
            y="Sales",
            color="IsWeekend",
            color_discrete_map={"Weekend": "#8b5cf6", "Weekday": "#3b82f6"},
            labels={"Num_Customers": "Customers", "Sales": "Sales (€)"},
            template="plotly_dark"
        )
        fig_scatter.update_layout(
            margin=dict(l=20, r=20, t=10, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Outfit, sans-serif"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)


def render_backtest_summary(
    splits_df: pd.DataFrame,
    split_id: int,
    summary_df: pd.DataFrame,
) -> None:
    split_data = splits_df[splits_df["Split"] == split_id]
    if split_data.empty:
        st.warning("No pre-computed data for this split.")
        return

    best_rows = []
    for obj in sorted(split_data["Objective"].unique()):
        sub  = split_data[split_data["Objective"] == obj]
        best = sub.loc[sub["Profit"].idxmax()]
        best_rows.append({
            "Objective":  obj,
            "Method":     best["Method"],
            "Profit (€)": int(best["Profit"]),
            "Units":      int(best["Units"]),
            "HR":         int(best["HR"]),
            "Feasible":   "✅" if bool(best["Feasible"]) else "❌",
        })

    st.markdown(f"**Best result per objective — Split {split_id}**")
    st.dataframe(pd.DataFrame(best_rows), use_container_width=True, hide_index=True)

    st.markdown("**Median performance across all 12 splits**")
    cols  = ["Objective", "Method", "Median_Profit", "Median_Units", "Median_HR"]
    avail = [c for c in cols if c in summary_df.columns]
    st.dataframe(summary_df[avail], use_container_width=True, hide_index=True)


def render_plan(plan_df: pd.DataFrame, objective: str, method: str) -> None:
    total_profit = 0
    total_units  = int(plan_df["Units_Total"].sum())
    total_hr     = int(plan_df["Daily_HR_Total"].sum())

    # Per-store expanders in 2-column grid
    store_results = []
    for store in STORES_ORDERED:
        sub   = plan_df[plan_df["Store"] == store].copy()
        ws    = STORE_PARAMS[store]["Ws"]
        gross = int(sub["Daily_Profit"].sum())
        net   = gross - ws
        total_profit += net
        store_results.append((store, sub, ws, gross, net))

    cols = st.columns(2)
    for idx, (store, sub, ws, gross, net) in enumerate(store_results):
        with cols[idx % 2]:
            with st.expander(f"📍 {store}  —  net {net:,} €", expanded=True):
                sub = sub.copy()
                if "Sales_X" in sub.columns and "Sales_J" in sub.columns:
                    sub["Sales (€)"]   = sub["Sales_X"].astype(int) + sub["Sales_J"].astype(int)
                if "HR_Cost_X" in sub.columns and "HR_Cost_J" in sub.columns:
                    sub["HR Cost (€)"] = sub["HR_Cost_X"].astype(int) + sub["HR_Cost_J"].astype(int)

                keep = ["Date", "Pred_Customers", "PR", "X", "J",
                        "Units_Total", "Sales (€)", "HR Cost (€)", "Daily_Profit", "Daily_HR_Total"]
                keep = [c for c in keep if c in sub.columns]
                disp = sub[keep].copy()
                disp.rename(columns={
                    "Pred_Customers": "Clients",
                    "Units_Total":    "Units",
                    "Daily_Profit":   "Profit (€)",
                    "Daily_HR_Total": "HR",
                }, inplace=True)
                disp["Date"] = disp["Date"].astype(str).str[:10]
                disp["PR"]   = disp["PR"].round(2)
                st.dataframe(disp, use_container_width=True, hide_index=True)

                c1, c2, c3 = st.columns(3)
                c1.metric("Gross profit",    f"{gross:,} €")
                c2.metric("Fixed cost (Ws)", f"−{ws:,} €")
                c3.metric("Net profit",      f"{net:,} €")

    # ── Visualizations ──
    st.markdown("### 📊 Performance Visualizations")
    v_col1, v_col2 = st.columns(2)
    with v_col1:
        # Net Profit per Store Bar Chart
        store_profits = pd.DataFrame([
            {"Store": store, "Net Profit (€)": net}
            for store, _, _, _, net in store_results
        ])
        st.markdown("**Net Profit per Store**")
        st.bar_chart(store_profits.set_index("Store"), height=250)
        
    with v_col2:
        # Daily distribution of units sold (combined for all stores)
        daily_units = plan_df.groupby("Date")["Units_Total"].sum().reset_index()
        daily_units["Date"] = pd.to_datetime(daily_units["Date"]).dt.strftime("%m-%d")
        st.markdown("**Daily Total Units Sold (vs Cap)**")
        st.line_chart(daily_units.set_index("Date"), height=250)

    st.divider()

    # Summary row
    feasible = total_units <= UNITS_CAP
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total net profit",        f"{total_profit:,} €")
    c2.metric("Total units sold",        f"{total_units:,}")
    c3.metric("Total HR (employee-days)", f"{total_hr:,}")
    badge = '<span class="badge-ok">✓ Within cap</span>' if feasible else f'<span class="badge-bad">⚠ Exceeds cap ({UNITS_CAP:,})</span>'
    c4.markdown(f"<br>{badge}", unsafe_allow_html=True)

    st.download_button(
        "⬇ Download plan CSV",
        data=plan_df.to_csv(index=False).encode(),
        file_name=f"dss_plan_{objective}_{method.replace(' ', '_')}.csv",
        mime="text/csv",
    )


def render_opt_panel(groups: list[Group], week_key: str) -> None:
    OBJ_LABELS = [
        "O1 — Maximize profit (no cap)",
        "O2 — Maximize profit (cap ≤ 10,000 units)",
        "O3 — Maximize profit & minimize HR",
    ]
    OBJ_MAP = {
        "O1 — Maximize profit (no cap)":             "O1",
        "O2 — Maximize profit (cap ≤ 10,000 units)": "O2",
        "O3 — Maximize profit & minimize HR":         "O3_WEIGHTED",
    }
    METHODS = ["Genetic Algorithm", "Hill Climbing", "Simulated Annealing", "Monte Carlo"]

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        obj_label  = st.selectbox("Objective", OBJ_LABELS, key=f"obj_{week_key}")
    with c2:
        method     = st.selectbox("Method", METHODS, key=f"mth_{week_key}")
    with c3:
        iterations = st.number_input("Iterations", min_value=100, max_value=5000,
                                     value=DEF_ITERS, step=100, key=f"itr_{week_key}")

    objective  = OBJ_MAP[obj_label]
    result_key = f"res_{week_key}_{objective}_{method}_{iterations}"

    st.markdown("")
    if st.button("▶  Run Optimization", type="primary", key=f"run_{week_key}"):
        with st.spinner(f"Running {method} · {iterations} iterations…"):
            plan_df, fitness = run_opt(groups, objective, method, iterations)
        st.session_state[result_key] = {"plan_df": plan_df, "fitness": fitness}

    if result_key in st.session_state:
        res = st.session_state[result_key]
        st.info(f"**Best fitness:** {res['fitness']:.2f}", icon="🎯")
        st.divider()
        render_plan(res["plan_df"], objective, method)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="DSS — Retail Planning",
        page_icon="🏪",
        layout="wide",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    data = load_app_data()
    store_dates     = data["store_dates"]
    best_methods_df = data["best_methods_df"]
    splits_df       = data["splits_df"]
    summary_df      = data["summary_df"]
    split_ranges    = data["split_ranges"]
    pred_lookup     = data["pred_lookup"]
    actual_lookup   = data["actual_lookup"]
    next7_groups    = build_next7_groups(data["next7_df"])

    # ── SIDEBAR ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🏪 Retail DSS")
        st.markdown("*Decision Support System*")
        st.divider()

        nx_s = next7_groups[0].date.date()
        nx_e = next7_groups[-1].date.date()
        week_labels: list[str] = [
            f"Week {k}  ({s.date()} → {e.date()})"
            for k, (s, e) in enumerate(split_ranges, start=1)
        ]
        week_labels.append(f"Next week  ({nx_s} → {nx_e})")

        st.markdown("**Select week**")
        sel_idx = st.selectbox(
            "week",
            range(len(week_labels)),
            format_func=lambda i: week_labels[i],
            index=len(week_labels) - 1,
            label_visibility="collapsed",
        )

        is_past  = sel_idx < N_SPLITS
        week_key = f"w{sel_idx}"

        if is_past:
            split_id = sel_idx + 1
            s, e = split_ranges[sel_idx]
            st.caption(f"📅 Historical split {split_id}")
            st.caption(f"{s.date()} → {e.date()}")
            st.caption("Actuals available ✅")
        else:
            split_id = None
            st.caption(f"📅 Next week (forecast)")
            st.caption(f"{nx_s} → {nx_e}")
            st.caption("No actuals ℹ️")

        st.divider()
        st.markdown("**Store parameters**")
        param_df = pd.DataFrame([
            {"Store": s, "Fx": v["Fx"], "Fj": v["Fj"], "Ws (€)": v["Ws"]}
            for s, v in STORE_PARAMS.items()
        ])
        st.dataframe(param_df, use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("**Unit cap (O2/O3)**")
        st.markdown(f"`{UNITS_CAP:,}` units / week")

    # ── MAIN AREA ──────────────────────────────────────────────────────────────
    week_label = week_labels[sel_idx]
    st.markdown(f"# 🏪 Decision Support System")
    st.markdown(f"**Week:** {week_label}")

    if is_past:
        groups = build_split_groups(split_id, pred_lookup, store_dates)
        tab_names = [
            "🔍 Exploratory Data Analysis (EDA)",
            "📊 Forecasts & Actuals",
            "⚙️ Optimization",
            "📈 Backtest Summary",
        ]
        tab0, tab1, tab2, tab3 = st.tabs(tab_names)
    else:
        groups = next7_groups
        tab_names = [
            "🔍 Exploratory Data Analysis (EDA)",
            "📊 Forecasts",
            "⚙️ Optimization",
        ]
        tab0, tab1, tab2 = st.tabs(tab_names)
        tab3 = None

    with tab0:
        render_eda(data["store_dfs"])

    with tab1:
        st.markdown("### Customer forecasts" + (" & actuals" if is_past else ""))
        render_forecasts(
            groups,
            actual_lookup if is_past else None,
            split_id,
            best_methods_df,
        )

    with tab2:
        st.markdown("### Run optimization")
        render_opt_panel(groups, week_key)

    if tab3 is not None:
        with tab3:
            st.markdown("### Pre-computed backtest results")
            render_backtest_summary(splits_df, split_id, summary_df)


if __name__ == "__main__":
    main()

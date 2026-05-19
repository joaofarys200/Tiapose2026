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
import streamlit as st

# ── Suppress matplotlib (imported at module level by otimizacao_metaheuristica) ──
if "matplotlib" not in sys.modules:
    _m = types.ModuleType("matplotlib")
    _m.pyplot = types.ModuleType("matplotlib.pyplot")  # type: ignore[attr-defined]
    sys.modules.setdefault("matplotlib", _m)
    sys.modules.setdefault("matplotlib.pyplot", _m.pyplot)

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
    # Store dates
    store_dates: dict[str, list[pd.Timestamp]] = {}
    for store, path in STORE_CSV.items():
        df = pd.read_csv(path, usecols=["Date"])
        df["Date"] = pd.to_datetime(df["Date"])
        store_dates[store] = df.sort_values("Date")["Date"].tolist()

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
    for store in STORES_ORDERED:
        rows = []
        for g in (g for g in groups if g.store == store):
            row: dict = {
                "Date": str(g.date.date()),
                "Day":  g.date.strftime("%a"),
                "H":    g.horizon,
                "Forecast": g.customers,
            }
            if has_actuals:
                actual = actual_lookup.get((store, split_id, g.horizon))
                row["Actual"] = actual
                row["Error"]  = (g.customers - actual) if actual is not None else None
            r = best_methods_df[
                (best_methods_df["Store"] == store) &
                (best_methods_df["Horizon"] == g.horizon)
            ]
            if not r.empty:
                m  = r.iloc[0]["Method"]
                ls = r.iloc[0]["LagSet"]
                row["Method"] = f"{m} ({ls})" if str(ls) != "-" else m
            rows.append(row)
        st.markdown(f"**{store}**")
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


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
            "Objective": obj,
            "Method":    best["Method"],
            "Profit (€)": int(best["Profit"]),
            "Units":      int(best["Units"]),
            "HR":         int(best["HR"]),
            "Feasible":   bool(best["Feasible"]),
        })

    st.markdown(f"**Best result per objective — split {split_id}**")
    st.dataframe(pd.DataFrame(best_rows), width="stretch", hide_index=True)

    st.markdown("**Backtest medians across all 12 splits**")
    cols  = ["Objective", "Method", "Median_Profit", "Median_Units", "Median_HR"]
    avail = [c for c in cols if c in summary_df.columns]
    st.dataframe(summary_df[avail], width="stretch", hide_index=True)


def render_plan(plan_df: pd.DataFrame, objective: str, method: str) -> None:
    total_profit = 0
    total_units  = int(plan_df["Units_Total"].sum())
    total_hr     = int(plan_df["Daily_HR_Total"].sum())

    for store in STORES_ORDERED:
        sub   = plan_df[plan_df["Store"] == store].copy()
        ws    = STORE_PARAMS[store]["Ws"]
        gross = int(sub["Daily_Profit"].sum())
        net   = gross - ws
        total_profit += net

        with st.expander(f"**{store}** — net weekly profit: {net:,} €", expanded=True):
            # Compute sales and HR cost per day
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
            st.dataframe(disp, width="stretch", hide_index=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Gross profit",    f"{gross:,} €")
            c2.metric("Fixed cost (Ws)", f"−{ws:,} €")
            c3.metric("Net profit",      f"{net:,} €")

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total net profit",         f"{total_profit:,} €")
    c2.metric("Total units sold",         f"{total_units:,}")
    c3.metric("Total HR (employee-days)",  f"{total_hr:,}")

    if total_units > UNITS_CAP:
        st.warning(f"⚠ Units ({total_units:,}) exceed cap ({UNITS_CAP:,})")
    else:
        st.success(f"✓ Units within cap ({UNITS_CAP:,})")

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

    c1, c2, c3 = st.columns(3)
    with c1:
        obj_label  = st.selectbox("Objective", OBJ_LABELS, key=f"obj_{week_key}")
    with c2:
        method     = st.selectbox("Method", METHODS, key=f"mth_{week_key}")
    with c3:
        iterations = st.slider("Iterations", 100, 2000, DEF_ITERS, 100, key=f"itr_{week_key}")

    objective  = OBJ_MAP[obj_label]
    result_key = f"res_{week_key}_{objective}_{method}_{iterations}"

    if st.button("▶  Run Optimization", type="primary", key=f"run_{week_key}"):
        with st.spinner(f"Running {method} · {iterations} iterations…"):
            plan_df, fitness = run_opt(groups, objective, method, iterations)
        st.session_state[result_key] = {"plan_df": plan_df, "fitness": fitness}

    if result_key in st.session_state:
        res = st.session_state[result_key]
        st.caption(f"Best fitness: {res['fitness']:.2f}")
        render_plan(res["plan_df"], objective, method)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="DSS — Retail Planning",
        page_icon="🏪",
        layout="wide",
    )

    data = load_app_data()
    store_dates     = data["store_dates"]
    best_methods_df = data["best_methods_df"]
    splits_df       = data["splits_df"]
    summary_df      = data["summary_df"]
    split_ranges    = data["split_ranges"]
    pred_lookup     = data["pred_lookup"]
    actual_lookup   = data["actual_lookup"]
    next7_groups    = build_next7_groups(data["next7_df"])

    st.title("🏪 Decision Support System")
    st.caption("Retail Store Planning — Baltimore · Lancaster · Philadelphia · Richmond")
    st.divider()

    # ── STEP 1 — Choose week ───────────────────────────────────────────────────
    st.subheader("① Choose a week")

    nx_s = next7_groups[0].date.date()
    nx_e = next7_groups[-1].date.date()
    week_labels: list[str] = [
        f"Week {k}  ({s.date()} → {e.date()})  [historical]"
        for k, (s, e) in enumerate(split_ranges, start=1)
    ]
    week_labels.append(f"Next week  ({nx_s} → {nx_e})  [forecast]")

    sel_idx = st.selectbox(
        "Select week",
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
        st.info(f"📅 Week {split_id}: **{s.date()} → {e.date()}** — historical split (actuals available)")
        groups = build_split_groups(split_id, pred_lookup, store_dates)
    else:
        split_id = None
        st.info(f"📅 Next week: **{nx_s} → {nx_e}** — forecast only (no actuals)")
        groups = next7_groups

    st.divider()

    # ── STEP 2 — Customer forecasts ────────────────────────────────────────────
    st.subheader("② Customer forecasts" + (" & actuals" if is_past else ""))
    render_forecasts(
        groups,
        actual_lookup if is_past else None,
        split_id,
        best_methods_df,
    )

    if is_past:
        with st.expander("📋 Pre-computed optimization summary (backtest)", expanded=False):
            render_backtest_summary(splits_df, split_id, summary_df)

    st.divider()

    # ── STEP 3 — Optimized plan ────────────────────────────────────────────────
    st.subheader("③ Optimized plan")
    render_opt_panel(groups, week_key)


if __name__ == "__main__":
    main()

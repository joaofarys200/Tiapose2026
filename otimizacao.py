import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


STORE_PARAMS = {
    "Baltimore": {"Fj": 1.00, "Fx": 1.15, "Ws": 700},
    "Lancaster": {"Fj": 1.05, "Fx": 1.20, "Ws": 730},
    "Philadelphia": {"Fj": 1.10, "Fx": 1.15, "Ws": 760},
    "Richmond": {"Fj": 1.15, "Fx": 1.25, "Ws": 800},
}

PR_VALUES = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


@dataclass(frozen=True)
class Group:
    idx: int
    store: str
    date: pd.Timestamp
    horizon: int
    customers: int


@dataclass(frozen=True)
class Option:
    pr: float
    x: int
    j: int
    assisted_x: int
    assisted_j: int
    units_x: int
    units_j: int
    units_total: int
    sales_x: int
    sales_j: int
    hr_cost_x: int
    hr_cost_j: int
    daily_profit: int
    hr_total: int


def is_weekend(date_value: pd.Timestamp) -> bool:
    return date_value.dayofweek >= 5


def wage_x(date_value: pd.Timestamp) -> int:
    return 95 if is_weekend(date_value) else 80


def wage_j(date_value: pd.Timestamp) -> int:
    return 70 if is_weekend(date_value) else 60


def round_units(factor: float, pr: float) -> int:
    return int(round(factor * 10.0 / math.log(2.0 - pr)))


def build_day_option(group: Group, pr: float, x: int, j: int) -> Option:
    params = STORE_PARAMS[group.store]

    assisted_x = min(7 * x, group.customers)
    rem = group.customers - assisted_x
    assisted_j = min(6 * j, rem)

    units_per_x = round_units(params["Fx"], pr)
    units_per_j = round_units(params["Fj"], pr)

    units_x = assisted_x * units_per_x
    units_j = assisted_j * units_per_j
    units_total = units_x + units_j

    # Match professor debug slides: round after daily aggregation by worker type.
    sales_x = int(round(units_x * (1.0 - pr) * 1.07))
    sales_j = int(round(units_j * (1.0 - pr) * 1.07))

    hr_cost_x = x * wage_x(group.date)
    hr_cost_j = j * wage_j(group.date)

    daily_profit = sales_x + sales_j - hr_cost_x - hr_cost_j

    return Option(
        pr=pr,
        x=x,
        j=j,
        assisted_x=assisted_x,
        assisted_j=assisted_j,
        units_x=units_x,
        units_j=units_j,
        units_total=units_total,
        sales_x=sales_x,
        sales_j=sales_j,
        hr_cost_x=hr_cost_x,
        hr_cost_j=hr_cost_j,
        daily_profit=daily_profit,
        hr_total=x + j,
    )


def compress_by_units(options: list[Option]) -> list[Option]:
    best_by_units: dict[int, Option] = {}
    for opt in options:
        cur = best_by_units.get(opt.units_total)
        if cur is None:
            best_by_units[opt.units_total] = opt
            continue
        if (opt.daily_profit > cur.daily_profit) or (
            opt.daily_profit == cur.daily_profit and opt.hr_total < cur.hr_total
        ):
            best_by_units[opt.units_total] = opt
    return sorted(best_by_units.values(), key=lambda o: o.units_total)


def pareto_profit_frontier(options: list[Option]) -> list[Option]:
    frontier: list[Option] = []
    best_profit = -10**18
    for opt in options:
        if opt.daily_profit > best_profit:
            frontier.append(opt)
            best_profit = opt.daily_profit
    return frontier


def pareto_profit_hr_frontier(options: list[Option]) -> list[Option]:
    frontier: list[Option] = []
    for opt in options:
        dominated = False
        for kept in frontier:
            if (
                kept.units_total <= opt.units_total
                and kept.daily_profit >= opt.daily_profit
                and kept.hr_total <= opt.hr_total
                and (
                    kept.units_total < opt.units_total
                    or kept.daily_profit > opt.daily_profit
                    or kept.hr_total < opt.hr_total
                )
            ):
                dominated = True
                break
        if dominated:
            continue

        new_frontier: list[Option] = []
        for kept in frontier:
            if (
                opt.units_total <= kept.units_total
                and opt.daily_profit >= kept.daily_profit
                and opt.hr_total <= kept.hr_total
                and (
                    opt.units_total < kept.units_total
                    or opt.daily_profit > kept.daily_profit
                    or opt.hr_total < kept.hr_total
                )
            ):
                continue
            new_frontier.append(kept)
        new_frontier.append(opt)
        frontier = new_frontier

    return sorted(frontier, key=lambda o: (o.units_total, -o.daily_profit, o.hr_total))


def generate_group_options(group: Group, units_cap: int) -> tuple[list[Option], list[Option], Option]:
    max_x = math.ceil(group.customers / 7) if group.customers > 0 else 0
    max_j = math.ceil(group.customers / 6) if group.customers > 0 else 0

    options_raw: list[Option] = []
    for pr in PR_VALUES:
        for x in range(max_x + 1):
            for j in range(max_j + 1):
                opt = build_day_option(group, pr, x, j)
                options_raw.append(opt)

    best_o1 = max(options_raw, key=lambda o: (o.daily_profit, -o.hr_total, -o.units_total))

    reduced = compress_by_units(options_raw)
    reduced_cap = [o for o in reduced if o.units_total <= units_cap]

    if not reduced_cap:
        zero_opt = build_day_option(group, 0.0, 0, 0)
        reduced_cap = [zero_opt]

    options_o2 = pareto_profit_frontier(reduced_cap)
    options_o3 = pareto_profit_hr_frontier(reduced_cap)

    return options_o2, options_o3, best_o1


def load_forecast_groups(forecast_file: Path) -> list[Group]:
    df = pd.read_csv(forecast_file)
    needed = {"Store", "Horizon", "ForecastDate", "Pred_Num_Customers"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Forecast file missing columns: {sorted(missing)}")

    df = df.copy()
    df["ForecastDate"] = pd.to_datetime(df["ForecastDate"])
    df["Pred_Num_Customers"] = (
        pd.to_numeric(df["Pred_Num_Customers"], errors="coerce").fillna(0.0).round().astype(int)
    )
    df["Pred_Num_Customers"] = df["Pred_Num_Customers"].clip(lower=0)

    df = df[df["Store"].isin(STORE_PARAMS.keys())]
    df = df.sort_values(["Store", "Horizon"]).reset_index(drop=True)

    groups: list[Group] = []
    idx = 0
    for store, grp in df.groupby("Store", sort=True):
        if len(grp) < 7:
            raise ValueError(f"Store {store} has fewer than 7 forecast rows.")
        first7 = grp.head(7)
        for _, row in first7.iterrows():
            groups.append(
                Group(
                    idx=idx,
                    store=store,
                    date=row["ForecastDate"],
                    horizon=int(row["Horizon"]),
                    customers=int(row["Pred_Num_Customers"]),
                )
            )
            idx += 1

    if len(groups) != 28:
        raise ValueError(f"Expected 28 rows (4 stores x 7 days). Got {len(groups)}.")

    return groups


def solve_o2(groups: list[Group], options_by_group: dict[int, list[Option]], units_cap: int) -> dict[int, Option]:
    states: dict[int, int] = {0: 0}
    backtrack: list[dict[int, tuple[int, Option]]] = []

    for g in groups:
        next_states: dict[int, int] = {}
        next_choice: dict[int, tuple[int, Option]] = {}
        options = options_by_group[g.idx]

        for used_units, score_profit in states.items():
            for opt in options:
                new_units = used_units + opt.units_total
                if new_units > units_cap:
                    continue
                new_profit = score_profit + opt.daily_profit
                cur = next_states.get(new_units)
                if cur is None or new_profit > cur:
                    next_states[new_units] = new_profit
                    next_choice[new_units] = (used_units, opt)

        states = next_states
        backtrack.append(next_choice)

    if not states:
        raise RuntimeError("No feasible O2 solution found.")

    best_units = max(states.keys(), key=lambda u: states[u])

    chosen: dict[int, Option] = {}
    cur_units = best_units
    for i in range(len(groups) - 1, -1, -1):
        prev_units, opt = backtrack[i][cur_units]
        chosen[groups[i].idx] = opt
        cur_units = prev_units

    return chosen


def solve_o3(groups: list[Group], options_by_group: dict[int, list[Option]], units_cap: int) -> dict[int, Option]:
    states: dict[int, tuple[int, int]] = {0: (0, 0)}
    backtrack: list[dict[int, tuple[int, Option]]] = []

    for g in groups:
        next_states: dict[int, tuple[int, int]] = {}
        next_choice: dict[int, tuple[int, Option]] = {}
        options = options_by_group[g.idx]

        for used_units, (score_profit, score_hr) in states.items():
            for opt in options:
                new_units = used_units + opt.units_total
                if new_units > units_cap:
                    continue

                new_profit = score_profit + opt.daily_profit
                new_hr = score_hr + opt.hr_total

                cur = next_states.get(new_units)
                if cur is None or (new_profit > cur[0]) or (new_profit == cur[0] and new_hr < cur[1]):
                    next_states[new_units] = (new_profit, new_hr)
                    next_choice[new_units] = (used_units, opt)

        states = next_states
        backtrack.append(next_choice)

    if not states:
        raise RuntimeError("No feasible O3 solution found.")

    best_units = max(states.keys(), key=lambda u: (states[u][0], -states[u][1]))

    chosen: dict[int, Option] = {}
    cur_units = best_units
    for i in range(len(groups) - 1, -1, -1):
        prev_units, opt = backtrack[i][cur_units]
        chosen[groups[i].idx] = opt
        cur_units = prev_units

    return chosen


def build_plan_df(objective: str, groups: list[Group], selected: dict[int, Option]) -> pd.DataFrame:
    rows = []
    for g in groups:
        opt = selected[g.idx]
        rows.append(
            {
                "Objective": objective,
                "Store": g.store,
                "Date": g.date.date().isoformat(),
                "Horizon": g.horizon,
                "Pred_Customers": g.customers,
                "PR": opt.pr,
                "X": opt.x,
                "J": opt.j,
                "Assisted_X": opt.assisted_x,
                "Assisted_J": opt.assisted_j,
                "Units_X": opt.units_x,
                "Units_J": opt.units_j,
                "Units_Total": opt.units_total,
                "Sales_X": opt.sales_x,
                "Sales_J": opt.sales_j,
                "HR_Cost_X": opt.hr_cost_x,
                "HR_Cost_J": opt.hr_cost_j,
                "Daily_Profit": opt.daily_profit,
                "Daily_HR_Total": opt.hr_total,
                "IsWeekend": int(is_weekend(g.date)),
            }
        )

    df = pd.DataFrame(rows).sort_values(["Store", "Horizon"]).reset_index(drop=True)

    # Apply fixed weekly store cost once per store.
    store_summary = (
        df.groupby("Store", as_index=False)
        .agg(
            Week_Units=("Units_Total", "sum"),
            Week_DailyProfit_NoFixed=("Daily_Profit", "sum"),
            Week_HR_Total=("Daily_HR_Total", "sum"),
        )
        .sort_values("Store")
    )
    store_summary["Fixed_Week_Cost_Ws"] = store_summary["Store"].map(lambda s: STORE_PARAMS[s]["Ws"])
    store_summary["Week_Profit"] = store_summary["Week_DailyProfit_NoFixed"] - store_summary["Fixed_Week_Cost_Ws"]
    store_summary["Objective"] = objective

    return df, store_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimization module (O1/O2/O3) for TIAPOSE project.")
    parser.add_argument(
        "--forecast-file",
        default="multivariate_next7.csv",
        help="CSV file with 7-day customer forecasts (default: multivariate_next7.csv)",
    )
    parser.add_argument(
        "--units-cap",
        type=int,
        default=10000,
        help="Hard cap for O2/O3 total sold units across all stores and days.",
    )
    parser.add_argument(
        "--output-prefix",
        default="optimization",
        help="Prefix for output CSV files.",
    )
    args = parser.parse_args()

    forecast_file = Path(args.forecast_file)
    if not forecast_file.exists():
        raise FileNotFoundError(f"Forecast file not found: {forecast_file}")

    groups = load_forecast_groups(forecast_file)

    options_o2: dict[int, list[Option]] = {}
    options_o3: dict[int, list[Option]] = {}
    selected_o1: dict[int, Option] = {}

    for g in groups:
        g_o2, g_o3, g_o1 = generate_group_options(g, args.units_cap)
        options_o2[g.idx] = g_o2
        options_o3[g.idx] = g_o3
        selected_o1[g.idx] = g_o1

    selected_o2 = solve_o2(groups, options_o2, args.units_cap)
    selected_o3 = solve_o3(groups, options_o3, args.units_cap)

    plan_o1, summary_o1 = build_plan_df("O1", groups, selected_o1)
    plan_o2, summary_o2 = build_plan_df("O2", groups, selected_o2)
    plan_o3, summary_o3 = build_plan_df("O3", groups, selected_o3)

    summary_all = pd.concat([summary_o1, summary_o2, summary_o3], ignore_index=True)
    global_summary = (
        summary_all.groupby("Objective", as_index=False)
        .agg(
            Total_Week_Units=("Week_Units", "sum"),
            Total_Week_HR=("Week_HR_Total", "sum"),
            Total_Week_Profit=("Week_Profit", "sum"),
        )
        .sort_values("Objective")
    )

    out_prefix = args.output_prefix
    plan_o1.to_csv(f"{out_prefix}_o1_plan.csv", index=False)
    plan_o2.to_csv(f"{out_prefix}_o2_plan.csv", index=False)
    plan_o3.to_csv(f"{out_prefix}_o3_plan.csv", index=False)
    summary_all.to_csv(f"{out_prefix}_store_summary.csv", index=False)
    global_summary.to_csv(f"{out_prefix}_global_summary.csv", index=False)

    print("Optimization completed.")
    print(f"Forecast source: {forecast_file}")
    print(f"Units cap (O2/O3): {args.units_cap}")
    print()
    print(global_summary.to_string(index=False))
    print()
    print("Created files:")
    print(f" - {out_prefix}_o1_plan.csv")
    print(f" - {out_prefix}_o2_plan.csv")
    print(f" - {out_prefix}_o3_plan.csv")
    print(f" - {out_prefix}_store_summary.csv")
    print(f" - {out_prefix}_global_summary.csv")


if __name__ == "__main__":
    main()

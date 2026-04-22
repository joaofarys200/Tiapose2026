import pandas as pd
from pathlib import Path

from otimizacao_metaheuristica import Group, STORE_PARAMS, build_day_option

PR = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
X = [4, 0, 8, 20, 0, 4, 3]
J = [0, 10, 4, 0, 5, 5, 4]

# Valores da semana do slide
C_B = [97, 61, 65, 71, 65, 89, 125]
C_P = [230, 144, 154, 168, 154, 211, 298]


def eval_week(store: str, customers: list[int]) -> tuple[pd.DataFrame, dict]:
    rows = []
    start = pd.Timestamp("2014-06-15")  # Sunday

    for d in range(7):
        group = Group(
            idx=d,
            store=store,
            date=start + pd.Timedelta(days=d),
            horizon=d + 1,
            customers=customers[d],
        )
        opt = build_day_option(group, PR[d], X[d], J[d])
        rows.append(
            {
                "Store": store,
                "Day": d + 1,
                "Date": group.date.date().isoformat(),
                "PR": PR[d],
                "X": X[d],
                "J": J[d],
                "Customers": customers[d],
                "Assisted_X": opt.assisted_x,
                "Assisted_J": opt.assisted_j,
                "Units_X": opt.units_x,
                "Units_J": opt.units_j,
                "Sales_X": opt.sales_x,
                "Sales_J": opt.sales_j,
                "HR_Cost_X": opt.hr_cost_x,
                "HR_Cost_J": opt.hr_cost_j,
                "Daily_Profit": opt.daily_profit,
            }
        )

    df = pd.DataFrame(rows)
    week = {
        "Store": store,
        "Tot_HR_X": int(df["X"].sum()),
        "Tot_HR_J": int(df["J"].sum()),
        "Tot_HR": int((df["X"] + df["J"]).sum()),
        "Xunits": int(df["Units_X"].sum()),
        "Junits": int(df["Units_J"].sum()),
        "Xsales": int(df["Sales_X"].sum()),
        "Jsales": int(df["Sales_J"].sum()),
        "XHRc": int(df["HR_Cost_X"].sum()),
        "JHRc": int(df["HR_Cost_J"].sum()),
        "Ws": STORE_PARAMS[store]["Ws"],
        "Week_Profit": int(df["Daily_Profit"].sum() - STORE_PARAMS[store]["Ws"]),
    }
    return df, week


def main() -> None:
    b_day, b_sum = eval_week("Baltimore", C_B)
    p_day, p_sum = eval_week("Philadelphia", C_P)

    daily = pd.concat([b_day, p_day], ignore_index=True)
    summary = pd.DataFrame([b_sum, p_sum])

    out_dir = Path("csv/optimization")
    out_dir.mkdir(parents=True, exist_ok=True)

    daily_file = out_dir / "validation_slide_daily.csv"
    summary_file = out_dir / "validation_slide_summary.csv"

    daily.to_csv(daily_file, index=False)
    summary.to_csv(summary_file, index=False)

    print(f"OK: {daily_file} e {summary_file} gerados")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

"""
Validação de soluções metaheurísticas contra exemplos dos slides.
Auto-detecta os ficheiros atuais e suporta validação por objetivo.
"""

import argparse
from pathlib import Path

import pandas as pd

from otimizacao_metaheuristica import STORE_PARAMS, Group, build_day_option


def validate_plan_df_against_slides(
    df: pd.DataFrame, expected_results: dict[str, int | None]
) -> dict:
    """Valida um DataFrame de plano contra valores esperados dos slides."""
    validation_rows = []
    store_profits = {}
    
    for store in ["Baltimore", "Lancaster", "Philadelphia", "Richmond"]:
        store_profits[store] = 0
    
    for _, row in df.iterrows():
        store = row["Store"]
        date_str = row["Date"]
        pred_customers = int(row["Pred_Customers"])
        pr_actual = float(row["PR"])
        x_actual = int(row["X"])
        j_actual = int(row["J"])
        
        # Reconstruir Group
        group = Group(
            idx=0,
            store=store,
            date=pd.to_datetime(date_str),
            horizon=int(row["Horizon"]),
            customers=pred_customers,
        )
        
        # Recalcular Option
        opt = build_day_option(group, pr_actual, x_actual, j_actual)
        
        # Verificar se valores coincidem
        values_match = (
            opt.units_total == int(row["Units_Total"])
            and opt.sales_x == int(row["Sales_X"])
            and opt.sales_j == int(row["Sales_J"])
            and opt.daily_profit == int(row["Daily_Profit"])
        )
        
        validation_rows.append(
            {
                "Store": store,
                "Date": date_str,
                "Customers": pred_customers,
                "PR": pr_actual,
                "X": x_actual,
                "J": j_actual,
                "Recalc_Units": opt.units_total,
                "Recalc_Sales": opt.sales_x + opt.sales_j,
                "Recalc_Profit": opt.daily_profit,
                "CSV_Units": int(row["Units_Total"]),
                "CSV_Sales": int(row["Sales_X"]) + int(row["Sales_J"]),
                "CSV_Profit": int(row["Daily_Profit"]),
                "Match": values_match,
            }
        )
        
        store_profits[store] += int(row["Daily_Profit"])
    
    # Aplicar custo fixo por loja
    summary = {}
    for store in ["Baltimore", "Lancaster", "Philadelphia", "Richmond"]:
        ws = STORE_PARAMS[store]["Ws"]
        week_profit = store_profits[store] - ws
        expected = expected_results.get(store)
        match = week_profit == expected if expected else None
        
        summary[store] = {
            "Daily_Profit_Sum": store_profits[store],
            "Fixed_Cost_Ws": ws,
            "Week_Profit": week_profit,
            "Expected": expected,
            "Match": match,
        }
    
    return {
        "validation_rows": validation_rows,
        "summary": summary,
    }


def discover_test_files(prefix: str) -> list[Path]:
    """Descobre automaticamente os ficheiros de plano atuais."""
    root = Path(".")
    best_files = sorted(root.glob(f"{prefix}*_best_plans.csv"))
    if best_files:
        return best_files
    return sorted(root.glob(f"{prefix}*_plan.csv"))


def print_validation_result(label: str, result: dict) -> None:
    print("\nResumo por Loja:")
    all_match = True
    for store, data in result["summary"].items():
        week_profit = data["Week_Profit"]
        exp = data["Expected"]
        match = data["Match"]

        if exp is not None:
            status = "✓ MATCH" if match else f"✗ MISMATCH (expected {exp})"
            print(f"  {store:15} Week_Profit={week_profit:6d}  {status}")
            if not match:
                all_match = False
        else:
            print(f"  {store:15} Week_Profit={week_profit:6d}  (sem validação)")

    print(f"\nResultado Final [{label}]: {'✅ VÁLIDO' if all_match else '❌ INVÁLIDO'}")


def main():
    parser = argparse.ArgumentParser(description="Validação metaheurística contra slides")
    parser.add_argument(
        "--prefix",
        default="csv/optimization/metaheuristica_clean",
        help="Prefixo dos ficheiros a validar (default: csv/optimization/metaheuristica_clean)",
    )
    parser.add_argument(
        "--file",
        default="",
        help="Ficheiro específico para validar em vez do prefixo.",
    )
    args = parser.parse_args()

    expected = {
        "Baltimore": 146,
        "Lancaster": None,
        "Philadelphia": 1728,
        "Richmond": None,
    }

    test_files = [Path(args.file)] if args.file else discover_test_files(args.prefix)

    print("=" * 80)
    print("VALIDAÇÃO METAHEURÍSTICA CONTRA SLIDES")
    print("=" * 80)

    out_dir = Path("csv/optimization")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not test_files:
        print(f"\n❌ Nenhum ficheiro encontrado para o prefixo: {args.prefix}")
        print("Dica: execute primeiro a otimização ou use --file com um CSV específico.")
        print(f"\n{'=' * 80}")
        print("Validação Completa")
        print(f"{'=' * 80}")
        return

    processed = 0
    for path in test_files:
        if not path.exists():
            print(f"\n❌ {path.name} not found, skipping...")
            continue

        print(f"\n{'─' * 80}")
        print(f"Validando: {path.name}")
        print(f"{'─' * 80}")

        df = pd.read_csv(path)
        if "Objective" in df.columns and df["Objective"].nunique() > 1:
            for objective, part_df in df.groupby("Objective"):
                result = validate_plan_df_against_slides(part_df.copy(), expected)
                print_validation_result(str(objective), result)
                output_file = out_dir / f"validacao_{path.stem}_{str(objective).lower()}.csv"
                pd.DataFrame(result["validation_rows"]).to_csv(output_file, index=False)
                print(f"Detalhes guardados em: {output_file}")
        else:
            result = validate_plan_df_against_slides(df, expected)
            label = str(df["Objective"].iloc[0]) if "Objective" in df.columns else path.stem
            print_validation_result(label, result)
            output_file = out_dir / f"validacao_{path.stem}.csv"
            pd.DataFrame(result["validation_rows"]).to_csv(output_file, index=False)
            print(f"Detalhes guardados em: {output_file}")

        processed += 1

    print(f"\nFicheiros processados com sucesso: {processed}")
    print(f"\n{'=' * 80}")
    print("Validação Completa")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()

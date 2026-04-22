"""
Validação de Coerência: Verificar que build_day_option calcula correctamente
para as soluções metaheurísticas geradas
"""

import pandas as pd
from pathlib import Path

from otimizacao_metaheuristica import build_day_option, Group, is_weekend


def validate_calculation_consistency(plan_csv: str) -> dict:
    """
    Valida se os cálculos em plan_csv coincidem com build_day_option
    
    Retorna:
        {
            "total_rows": int,
            "rows_match": int,
            "rows_mismatch": int,
            "mismatches": [...]
        }
    """
    df = pd.read_csv(plan_csv)
    
    mismatches = []
    rows_match = 0
    rows_mismatch = 0
    
    for idx, row in df.iterrows():
        store = row["Store"]
        date_str = row["Date"]
        customers = int(row["Pred_Customers"])
        pr = float(row["PR"])
        x = int(row["X"])
        j = int(row["J"])
        
        # Reconstruir Group e recalcular
        group = Group(
            idx=idx,
            store=store,
            date=pd.to_datetime(date_str),
            horizon=int(row["Horizon"]),
            customers=customers,
        )
        
        opt = build_day_option(group, pr, x, j)
        
        # Comparar com CSV
        csv_units = int(row["Units_Total"])
        csv_sales_x = int(row["Sales_X"])
        csv_sales_j = int(row["Sales_J"])
        csv_profit = int(row["Daily_Profit"])
        csv_hr = int(row["Daily_HR_Total"])
        
        match = (
            opt.units_total == csv_units
            and opt.sales_x == csv_sales_x
            and opt.sales_j == csv_sales_j
            and opt.daily_profit == csv_profit
            and opt.hr_total == csv_hr
        )
        
        if match:
            rows_match += 1
        else:
            rows_mismatch += 1
            mismatches.append({
                "Row": idx,
                "Store": store,
                "Date": date_str,
                "PR": pr,
                "X": x,
                "J": j,
                "Units_CSV": csv_units,
                "Units_Calc": opt.units_total,
                "Sales_X_CSV": csv_sales_x,
                "Sales_X_Calc": opt.sales_x,
                "Sales_J_CSV": csv_sales_j,
                "Sales_J_Calc": opt.sales_j,
                "Profit_CSV": csv_profit,
                "Profit_Calc": opt.daily_profit,
                "HR_CSV": csv_hr,
                "HR_Calc": opt.hr_total,
            })
    
    return {
        "total_rows": len(df),
        "rows_match": rows_match,
        "rows_mismatch": rows_mismatch,
        "mismatches": mismatches,
    }


def main():
    print("=" * 90)
    print("VALIDAÇÃO DE COERÊNCIA: build_day_option ↔ CSVs Metaheurísticas")
    print("=" * 90)
    
    # Testar um ficheiro de cada método
    test_files = [
        "metaheuristica_v2_o1_monte_carlo_plan.csv",
        "metaheuristica_v2_o1_hill_climbing_plan.csv",
        "metaheuristica_v2_o1_simulated_annealing_plan.csv",
    ]
    
    overall_match = True
    
    for test_file in test_files:
        path = Path(test_file)
        if not path.exists():
            print(f"\n❌ {test_file} not found")
            continue
        
        print(f"\n{'─' * 90}")
        print(f"Validando: {test_file}")
        print(f"{'─' * 90}")
        
        result = validate_calculation_consistency(test_file)
        
        total = result["total_rows"]
        match = result["rows_match"]
        mismatch = result["rows_mismatch"]
        
        pct_match = 100.0 * match / total if total > 0 else 0
        
        print(f"\n✓ Linhas OK:     {match:3d}/{total} ({pct_match:.1f}%)")
        print(f"✗ Linhas ERRO:   {mismatch:3d}/{total}")
        
        if mismatch == 0:
            print(f"\n✅ VÁLIDO - Todos os cálculos coincidem!")
        else:
            print(f"\n❌ INVÁLIDO - {mismatch} linhas com discrepâncias")
            overall_match = False
            
            # Mostrar primeiras 3 discrepâncias
            print(f"\nPrimeiras discrepâncias (max 3):")
            for i, mm in enumerate(result["mismatches"][:3]):
                print(f"\n  [{i+1}] Row {mm['Row']}: {mm['Store']} {mm['Date']}")
                print(f"      Units: CSV={mm['Units_CSV']} vs Calc={mm['Units_Calc']}")
                print(f"      Profit: CSV={mm['Profit_CSV']} vs Calc={mm['Profit_Calc']}")
    
    print(f"\n{'=' * 90}")
    if overall_match:
        print("✅ CONCLUSÃO: Todos os ficheiros são coherentes. build_day_option funciona correctamente!")
    else:
        print("⚠️  CONCLUSÃO: Algumas discrepâncias encontradas. Ver detalhes acima.")
    print(f"{'=' * 90}\n")


if __name__ == "__main__":
    main()

"""
Validação rápida da função de avaliação com valores dos slides.

Uso típico (ajusta para os valores reais do slide):
python project/validar_eval_slide.py --store Baltimore --date 2014-06-15 --customers 97 --pr 0.02 --x 11 --j 3

Se quiseres validar contra resultados esperados do slide:
python project/validar_eval_slide.py --store Baltimore --date 2014-06-15 --customers 97 --pr 0.02 --x 11 --j 3 \
  --expected assisted_x=77 --expected assisted_j=18 --expected units_total=1579 --expected daily_profit=401
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict

import pandas as pd

from otimizacao_metaheuristica import Group, build_day_option


def parse_expected(pairs: list[str]) -> dict[str, float]:
    expected: dict[str, float] = {}
    for raw in pairs:
        if "=" not in raw:
            raise ValueError(f"Formato invalido em --expected: '{raw}'. Usa campo=valor.")
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Campo vazio em --expected: '{raw}'.")
        expected[key] = float(value)
    return expected


def almost_equal(a: float, b: float, atol: float = 1e-9) -> bool:
    return math.isclose(a, b, rel_tol=0.0, abs_tol=atol)


def main() -> None:
    parser = argparse.ArgumentParser(description="Valida a eval com os valores dos slides.")
    parser.add_argument("--store", required=True, help="Loja (ex: Baltimore)")
    parser.add_argument("--date", required=True, help="Data no formato YYYY-MM-DD")
    parser.add_argument("--customers", required=True, type=int, help="Clientes previstos")
    parser.add_argument("--pr", required=True, type=float, help="PR em [0, 0.3]")
    parser.add_argument("--x", required=True, type=int, help="Numero de trabalhadores tipo X")
    parser.add_argument("--j", required=True, type=int, help="Numero de trabalhadores tipo J")
    parser.add_argument(
        "--expected",
        action="append",
        default=[],
        help=(
            "Valor esperado no formato campo=valor. Pode repetir este argumento. "
            "Campos validos incluem assisted_x, assisted_j, units_x, units_j, units_total, "
            "sales_x, sales_j, hr_cost_x, hr_cost_j, daily_profit, hr_total."
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Tolerancia absoluta para comparar esperados (default: 1e-6).",
    )

    args = parser.parse_args()

    group = Group(
        idx=0,
        store=args.store,
        date=pd.Timestamp(args.date),
        horizon=1,
        customers=args.customers,
    )

    option = build_day_option(group, args.pr, args.x, args.j)
    result = asdict(option)

    print("=== Resultado da eval (build_day_option) ===")
    print(f"store={args.store} date={args.date} customers={args.customers} pr={args.pr} x={args.x} j={args.j}")
    for key in [
        "assisted_x",
        "assisted_j",
        "units_x",
        "units_j",
        "units_total",
        "sales_x",
        "sales_j",
        "hr_cost_x",
        "hr_cost_j",
        "daily_profit",
        "hr_total",
    ]:
        print(f"{key}: {result[key]}")

    if not args.expected:
        return

    expected = parse_expected(args.expected)
    print("\n=== Validação contra esperados ===")

    failures = 0
    for key, exp in expected.items():
        if key not in result:
            print(f"[ERRO] campo desconhecido: {key}")
            failures += 1
            continue

        got = float(result[key])
        ok = almost_equal(got, exp, atol=args.tolerance)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {key}: esperado={exp} obtido={got}")
        if not ok:
            failures += 1

    if failures > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse


def process_backtest(csv_path: Path, out_root: Path, max_stores: int, all_stores: bool):
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return
    df = pd.read_csv(csv_path)
    if 'LagSet' in df.columns:
        df['MethodLabel'] = df.apply(
            lambda r: f"{r['Method']}_{r['LagSet']}" if pd.notna(r['LagSet']) and str(r['LagSet']) != '-' else r['Method'],
            axis=1,
        )
    else:
        df['MethodLabel'] = df['Method']

    stores = sorted(df['Store'].unique())
    if not all_stores:
        stores = stores[:max_stores]

    out_root.mkdir(parents=True, exist_ok=True)

    for store in stores:
        df_s = df[df['Store'] == store]
        methods = sorted(df_s['MethodLabel'].unique())
        store_out = out_root / store
        store_out.mkdir(parents=True, exist_ok=True)

        # Individual method plots
        for method in methods:
            sub = df_s[df_s['MethodLabel'] == method]
            # aggregate across splits (median) per horizon
            grp = sub.groupby('Horizon').median(numeric_only=True).reset_index()
            if grp.empty:
                continue
            horizons = grp['Horizon'].astype(int)
            y_true = grp['y_true'] if 'y_true' in grp.columns else None
            y_pred = grp['y_pred'] if 'y_pred' in grp.columns else (grp['Pred_Num_Customers'] if 'Pred_Num_Customers' in grp.columns else None)

            fig, ax = plt.subplots(figsize=(4, 3))
            if y_true is not None:
                ax.plot(horizons, y_true, '-o', color='k', label='real', markerfacecolor='white')
            if y_pred is not None:
                ax.plot(horizons, y_pred, '-o', color='green', label='previsto')
            ax.set_xticks(horizons)
            ax.set_xlabel('Horizon')
            ax.set_title(f"{store} — {method}")
            ax.grid(True, linestyle=':', linewidth=0.5)
            ax.legend()
            fname = store_out / f"{method}.png"
            fig.savefig(fname, dpi=100, bbox_inches='tight')
            plt.close(fig)

        # Panel with all methods for the store
        n = len(methods)
        if n == 0:
            continue
        cols = 3
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        axes = np.array(axes).reshape(-1)
        for i, method in enumerate(methods):
            ax = axes[i]
            sub = df_s[df_s['MethodLabel'] == method]
            grp = sub.groupby('Horizon').median(numeric_only=True).reset_index()
            if grp.empty:
                ax.set_visible(False)
                continue
            horizons = grp['Horizon'].astype(int)
            if 'y_true' in grp.columns:
                ax.plot(horizons, grp['y_true'], '-o', color='k', markerfacecolor='white')
            ypred = grp['y_pred'] if 'y_pred' in grp.columns else (grp['Pred_Num_Customers'] if 'Pred_Num_Customers' in grp.columns else None)
            if ypred is not None:
                ax.plot(horizons, ypred, '-o', color='green')
            ax.set_title(method)
            ax.set_xticks(horizons)
            ax.grid(True, linestyle=':', linewidth=0.5)

        # remove extra axes
        for j in range(i + 1, rows * cols):
            try:
                fig.delaxes(axes[j])
            except Exception:
                pass

        fig.suptitle(f"Store: {store}")
        outp = store_out / "all_methods.png"
        fig.savefig(outp, dpi=100, bbox_inches='tight')
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Generate forecasting plots from backtest CSVs')
    parser.add_argument('--all', action='store_true', help='Process all stores (default: only first N)')
    parser.add_argument('--max-stores', type=int, default=2, help='Max stores to process when --all not set')
    parser.add_argument('--out', type=str, default='csv/analysis/forecasting', help='Output root folder')
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    out_root = root / args.out

    # Univariate
    uni_path = root / 'csv' / 'forecast' / 'univariate' / 'univariate_backtest_all_splits.csv'
    process_backtest(uni_path, out_root / 'univariate', args.max_stores, args.all)

    # Multivariate
    mv_path = root / 'csv' / 'forecast' / 'multivariate' / 'multivariate_backtest_all_splits.csv'
    process_backtest(mv_path, out_root / 'multivariate', args.max_stores, args.all)


if __name__ == '__main__':
    main()

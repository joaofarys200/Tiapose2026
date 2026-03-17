import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.titleweight'] = 'bold'
sns.set_style("whitegrid")

STORE_FILES = {
    'Baltimore': 'baltimore.csv',
    'Lancaster': 'lancaster.csv',
    'Philadelphia': 'philadelphia.csv',
    'Richmond': 'richmond.csv'
}


def load_data():
    """Carrega os CSVs e devolve DataFrame combinado para análises multi-loja."""
    frames = []

    for store_name, file_name in STORE_FILES.items():
        df = pd.read_csv(file_name)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Store'] = store_name
        df['DayName'] = df['Date'].dt.day_name()
        df['Month'] = df['Date'].dt.month
        df['MonthName'] = df['Date'].dt.month_name()
        df['IsWeekend'] = df['Date'].dt.dayofweek.isin([5, 6]).astype(int)
        df['TouristEvent_bin'] = (df['TouristEvent'].str.strip().str.lower() == 'yes').astype(int)
        frames.append(df)

        print(f"[{store_name.upper()}] Shape: {df.shape} | Datas: {df['Date'].min().date()} a {df['Date'].max().date()}")

    return pd.concat(frames, ignore_index=True)


def print_data_quality(df_all):
    print("\n\n=== QUALIDADE DE DADOS (MULTI-LOJA) ===\n")

    cols_num = ['Num_Employees', 'Num_Customers', 'Pct_On_Sale', 'Sales']
    miss = df_all.groupby('Store')[cols_num].apply(lambda x: x.isnull().sum())
    print("Valores em falta por loja e variável:")
    print(miss)

    print("\nOutliers em Num_Customers (Z-score > 3) por loja:")
    for store, grp in df_all.groupby('Store'):
        z = np.abs(stats.zscore(grp['Num_Customers'].dropna()))
        print(f"  {store:15s}: {(z > 3).sum()}")


def print_store_kpis(df_all):
    print("\n\n=== KPI COMPARATIVO ENTRE LOJAS ===\n")

    kpis = (
        df_all.groupby('Store')
        .agg(
            Dias=('Date', 'count'),
            Clientes_Medios=('Num_Customers', 'mean'),
            Sales_Media=('Sales', 'mean'),
            Sales_Total=('Sales', 'sum'),
            Empregados_Medios=('Num_Employees', 'mean'),
            Pct_On_Sale_Medio=('Pct_On_Sale', 'mean')
        )
        .round(2)
        .sort_values('Sales_Total', ascending=False)
    )

    kpis['Sales_por_Cliente'] = (
        df_all.groupby('Store')['Sales'].sum() / df_all.groupby('Store')['Num_Customers'].sum()
    ).round(2)

    print(kpis)


def plot_total_trend(df_all):
    daily = (
        df_all.groupby('Date')[['Num_Customers', 'Sales']]
        .sum()
        .sort_index()
    )
    daily['Customers_MA7'] = daily['Num_Customers'].rolling(7, min_periods=1).mean()
    daily['Sales_MA7'] = daily['Sales'].rolling(7, min_periods=1).mean()

    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    axes[0].plot(daily.index, daily['Num_Customers'], color='#1E88E5', alpha=0.35, linewidth=0.9, label='Clientes (total diário)')
    axes[0].plot(daily.index, daily['Customers_MA7'], color='#0D47A1', linewidth=2, label='Média móvel 7d')
    axes[0].set_title('Total Multi-Loja - Clientes por Dia')
    axes[0].set_ylabel('Clientes')
    axes[0].legend()

    axes[1].plot(daily.index, daily['Sales'], color='#43A047', alpha=0.35, linewidth=0.9, label='Sales (total diário)')
    axes[1].plot(daily.index, daily['Sales_MA7'], color='#1B5E20', linewidth=2, label='Média móvel 7d')
    axes[1].set_title('Total Multi-Loja - Sales por Dia')
    axes[1].set_ylabel('Sales')
    axes[1].legend()

    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


def plot_cross_store_comparisons(df_all):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    sns.boxplot(data=df_all, x='Store', y='Num_Customers', ax=axes[0, 0], palette='Set2')
    axes[0, 0].set_title('Distribuição de Clientes por Loja')
    axes[0, 0].tick_params(axis='x', rotation=20)

    sns.boxplot(data=df_all, x='Store', y='Sales', ax=axes[0, 1], palette='Set3')
    axes[0, 1].set_title('Distribuição de Sales por Loja')
    axes[0, 1].tick_params(axis='x', rotation=20)

    weekend = (
        df_all.groupby(['Store', 'IsWeekend'])['Num_Customers']
        .mean()
        .reset_index()
    )
    weekend['Tipo'] = weekend['IsWeekend'].map({0: 'Dia Semana', 1: 'Fim Semana'})
    sns.barplot(data=weekend, x='Store', y='Num_Customers', hue='Tipo', ax=axes[1, 0], palette='Paired')
    axes[1, 0].set_title('Clientes Médios: Semana vs Fim de Semana')
    axes[1, 0].tick_params(axis='x', rotation=20)

    monthly = (
        df_all.groupby(['MonthName', 'Store'])['Num_Customers']
        .mean()
        .reset_index()
    )
    month_order = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ]
    monthly['MonthName'] = pd.Categorical(monthly['MonthName'], categories=month_order, ordered=True)
    monthly = monthly.sort_values('MonthName')

    sns.lineplot(data=monthly, x='MonthName', y='Num_Customers', hue='Store', marker='o', ax=axes[1, 1])
    axes[1, 1].set_title('Sazonalidade Mensal de Clientes por Loja')
    axes[1, 1].tick_params(axis='x', rotation=35)

    plt.tight_layout()
    plt.show()


def print_and_plot_tourist_event_impact(df_all):
    print("\n\n=== IMPACTO DE TOURISTEVENT (MULTI-LOJA) ===\n")
    for store, grp in df_all.groupby('Store'):
        base = grp[grp['TouristEvent_bin'] == 0]['Num_Customers'].dropna()
        event = grp[grp['TouristEvent_bin'] == 1]['Num_Customers'].dropna()
        if len(base) > 3 and len(event) > 3:
            _, p_val = stats.ttest_ind(event, base, equal_var=False)
            diff_pct = ((event.mean() - base.mean()) / base.mean()) * 100
            print(
                f"{store:15s} | Sem evento={base.mean():8.1f} | "
                f"Com evento={event.mean():8.1f} | Delta={diff_pct:7.2f}% | p-valor={p_val:.4f}"
            )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df_plot = df_all.copy()
    df_plot['TouristEvent_lbl'] = df_plot['TouristEvent_bin'].map({0: 'No', 1: 'Yes'})

    sns.boxplot(data=df_plot, x='Store', y='Num_Customers', hue='TouristEvent_lbl', ax=axes[0], palette='pastel')
    axes[0].set_title('TouristEvent vs Clientes (por Loja)')
    axes[0].tick_params(axis='x', rotation=20)

    effect = (
        df_plot.groupby(['Store', 'TouristEvent_lbl'])['Num_Customers']
        .mean()
        .reset_index()
    )
    sns.barplot(data=effect, x='Store', y='Num_Customers', hue='TouristEvent_lbl', ax=axes[1], palette='muted')
    axes[1].set_title('Clientes Médios com/sem TouristEvent')
    axes[1].tick_params(axis='x', rotation=20)

    plt.tight_layout()
    plt.show()


def plot_correlations(df_all):
    print("\n\n=== CORRELAÇÕES GLOBAIS (POOL MULTI-LOJA) ===\n")
    corr_cols = ['Num_Employees', 'Pct_On_Sale', 'TouristEvent_bin', 'Num_Customers', 'Sales']
    corr_global = df_all[corr_cols].corr().round(3)
    print(corr_global['Sales'].sort_values(ascending=False))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(corr_global, vmin=-1, vmax=1, center=0, annot=True, fmt='.2f', cmap='RdYlBu_r', square=True, ax=axes[0])
    axes[0].set_title('Heatmap Global - Todas as Lojas')

    corr_store = (
        df_all.groupby('Store')[corr_cols]
        .corr()
        .reset_index()
    )
    corr_sales = corr_store[corr_store['level_1'] == 'Sales'][['Store', 'Num_Employees', 'Pct_On_Sale', 'TouristEvent_bin', 'Num_Customers']]
    corr_sales = corr_sales.set_index('Store')
    sns.heatmap(corr_sales, vmin=-1, vmax=1, center=0, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1])
    axes[1].set_title('Correlação com Sales por Loja')

    plt.tight_layout()
    plt.show()


def main():
    print("=" * 68)
    print("  EDA MESTRE - Análises Multi-Loja (Cross-Store)")
    print("=" * 68)

    df_all = load_data()
    print_data_quality(df_all)
    print_store_kpis(df_all)
    plot_total_trend(df_all)
    plot_cross_store_comparisons(df_all)
    print_and_plot_tourist_event_impact(df_all)
    plot_correlations(df_all)

    print("\n\n=== EDA MESTRE CONCLUÍDA ===")
    print("Este script cobre apenas análises com múltiplas lojas. Para análises individuais, usa os scripts por loja.")


if __name__ == '__main__':
    main()


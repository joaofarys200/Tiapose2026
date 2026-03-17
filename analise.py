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

def load_data():
    """Carrega os 4 CSVs e faz pré-processamento inicial."""
    stores = ['baltimore', 'lancaster', 'philadelphia', 'richmond']
    dfs = {}

    for store in stores:
        df = pd.read_csv(f'{store}.csv', sep=',')
        df['Date'] = pd.to_datetime(df['Date'])
        df['Store'] = store.capitalize()

        # Features temporais
        df['DayOfWeek']  = df['Date'].dt.dayofweek          # 0=Mon, 6=Sun
        df['DayName']    = df['Date'].dt.day_name()
        df['Month']      = df['Date'].dt.month
        df['MonthName']  = df['Date'].dt.month_name()
        df['Year']       = df['Date'].dt.year
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
        df['IsWeekend']  = df['DayOfWeek'].isin([5, 6]).astype(int)

        # TouristEvent para binário
        df['TouristEvent_bin'] = (df['TouristEvent'].str.strip().str.lower() == 'yes').astype(int)

        dfs[store] = df
        print(f"[{store.upper()}] Shape: {df.shape} | "
              f"Datas: {df['Date'].min().date()} a {df['Date'].max().date()}")

    return dfs

print("=" * 60)
print("  EDA - USA Stores")
print("=" * 60)
dfs = load_data()

# DataFrame combinado
df_all = pd.concat(dfs.values(), ignore_index=True)


print("\n\n=== ESTATÍSTICAS DESCRITIVAS POR LOJA ===\n")

cols_num = ['Num_Employees', 'Num_Customers', 'Pct_On_Sale', 'Sales']

for store, df in dfs.items():
    print(f"\n--- {store.upper()} ---")
    desc = df[cols_num].describe().round(2)
    print(desc)
    missing = df.isnull().sum()
    if missing.any():
        print(f"  Valores em falta:\n{missing[missing > 0]}")
    else:
        print("  Sem valores em falta.")


print("\n\n=== ANÁLISE DE VALORES EM FALTA ===\n")
for store, df in dfs.items():
    total_miss = df[cols_num].isnull().sum().sum()
    print(f"  {store.capitalize():15s}: {total_miss} valores em falta")

print("\n\n=== OUTLIERS (Z-score > 3) - Num_Customers ===\n")
for store, df in dfs.items():
    z = np.abs(stats.zscore(df['Num_Customers'].dropna()))
    n_out = (z > 3).sum()
    print(f"  {store.capitalize():15s}: {n_out} outliers detectados")

# Serie temporal por loja
fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=False)
colors = ['#2196F3', '#4CAF50', '#FF5722', '#9C27B0']
store_list = list(dfs.keys())

for i, (store, df) in enumerate(dfs.items()):
    axes[i].plot(df['Date'], df['Num_Customers'],
                 color=colors[i], linewidth=0.8, alpha=0.9)
    axes[i].set_title(f'{store.capitalize()} — Número Diário de Clientes')
    axes[i].set_ylabel('Clientes')
    axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[i].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(axes[i].xaxis.get_majorticklabels(), rotation=30)

plt.suptitle('Série Temporal — Num_Customers por Loja', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, (store, df) in enumerate(dfs.items()):
    ax = axes[i]
    sns.histplot(df['Num_Customers'], kde=True, ax=ax,
                 color=colors[i], bins=40, alpha=0.7)
    ax.axvline(df['Num_Customers'].mean(), color='red',
               linestyle='--', linewidth=1.5, label=f"Média: {df['Num_Customers'].mean():.0f}")
    ax.axvline(df['Num_Customers'].median(), color='orange',
               linestyle='--', linewidth=1.5, label=f"Mediana: {df['Num_Customers'].median():.0f}")
    ax.set_title(f'{store.capitalize()}')
    ax.set_xlabel('Num_Customers')
    ax.legend(fontsize=9)

plt.suptitle('Distribuição de Num_Customers por Loja', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()


day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, (store, df) in enumerate(dfs.items()):
    ax = axes[i]
    avg_day = df.groupby('DayName')['Num_Customers'].mean().reindex(day_order)
    bars = ax.bar(range(7), avg_day.values, color=colors[i], alpha=0.8, edgecolor='white')
    ax.set_xticks(range(7))
    ax.set_xticklabels(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
    ax.set_title(f'{store.capitalize()}')
    ax.set_ylabel('Média de Clientes')

    # Destacar fim de semana
    for j in [5, 6]:
        bars[j].set_edgecolor('red')
        bars[j].set_linewidth(2)

plt.suptitle('Média de Clientes por Dia da Semana\n(vermelho = fim de semana)',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()

month_order = ['January','February','March','April','May','June',
               'July','August','September','October','November','December']

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for i, (store, df) in enumerate(dfs.items()):
    ax = axes[i]
    avg_month = df.groupby('MonthName')['Num_Customers'].mean().reindex(month_order)
    ax.plot(range(12), avg_month.values, marker='o', color=colors[i],
            linewidth=2, markersize=7)
    ax.fill_between(range(12), avg_month.values, alpha=0.15, color=colors[i])
    ax.set_xticks(range(12))
    ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun',
                        'Jul','Aug','Sep','Oct','Nov','Dec'], rotation=30)
    ax.set_title(f'{store.capitalize()}')
    ax.set_ylabel('Média de Clientes')

plt.suptitle('Média de Clientes por Mês', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, (store, df) in enumerate(dfs.items()):
    ax = axes[i]
    df_plot = df.copy()
    df_plot['Tipo'] = df_plot['IsWeekend'].map({0: 'Dia de Semana', 1: 'Fim de Semana'})
    sns.boxplot(data=df_plot, x='Tipo', y='Num_Customers',
                palette=['#64B5F6', '#EF9A9A'], ax=ax)
    ax.set_title(f'{store.capitalize()}')
    ax.set_xlabel('')
    ax.set_ylabel('Num_Customers')

plt.suptitle('Clientes: Dia de Semana vs Fim de Semana', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, (store, df) in enumerate(dfs.items()):
    ax = axes[i]
    ax.scatter(df['Pct_On_Sale'], df['Num_Customers'],
               alpha=0.3, color=colors[i], s=15)

    # Linha de tendência
    valid = df[['Pct_On_Sale', 'Num_Customers']].dropna()
    if len(valid) > 10:
        m, b, r, p, _ = stats.linregress(valid['Pct_On_Sale'], valid['Num_Customers'])
        x_line = np.linspace(valid['Pct_On_Sale'].min(), valid['Pct_On_Sale'].max(), 100)
        ax.plot(x_line, m * x_line + b, color='red', linewidth=2,
                label=f'r={r:.2f}, p={p:.3f}')
        ax.legend(fontsize=9)

    ax.set_title(f'{store.capitalize()}')
    ax.set_xlabel('Pct_On_Sale')
    ax.set_ylabel('Num_Customers')

plt.suptitle('Promoções (Pct_On_Sale) vs Número de Clientes', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()


print("\n\n=== COMPARATIVO GERAL ENTRE LOJAS ===\n")

summary = (
    df_all.groupby('Store')
    .agg(
        Dias=('Date', 'count'),
        Clientes_Medios=('Num_Customers', 'mean'),
        Clientes_Mediana=('Num_Customers', 'median'),
        Sales_Media=('Sales', 'mean'),
        Sales_Total=('Sales', 'sum'),
        Empregados_Medios=('Num_Employees', 'mean'),
        Pct_On_Sale_Medio=('Pct_On_Sale', 'mean')
    )
)

summary['Sales_por_Cliente'] = summary['Sales_Total'] / (
    df_all.groupby('Store')['Num_Customers'].sum()
)

summary = summary.sort_values('Sales_Total', ascending=False).round(2)
print(summary)


print("\n\n=== IMPACTO DE TOURISTEVENT EM CLIENTES ===\n")

for store, df in dfs.items():
    base = df[df['TouristEvent_bin'] == 0]['Num_Customers'].dropna()
    event = df[df['TouristEvent_bin'] == 1]['Num_Customers'].dropna()

    if len(base) > 3 and len(event) > 3:
        t_stat, p_val = stats.ttest_ind(event, base, equal_var=False)
        diff_pct = ((event.mean() - base.mean()) / base.mean()) * 100
        print(
            f"{store.capitalize():15s} | "
            f"Sem evento={base.mean():8.1f} | "
            f"Com evento={event.mean():8.1f} | "
            f"Delta={diff_pct:7.2f}% | p-valor={p_val:.4f}"
        )


fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, (store, df) in enumerate(dfs.items()):
    ax = axes[i]
    df_plot = df.copy()
    df_plot['TouristEvent_lbl'] = df_plot['TouristEvent_bin'].map({0: 'No', 1: 'Yes'})
    sns.boxplot(
        data=df_plot,
        x='TouristEvent_lbl',
        y='Num_Customers',
        palette=['#B0BEC5', '#FFCC80'],
        ax=ax
    )
    ax.set_title(f'{store.capitalize()}')
    ax.set_xlabel('TouristEvent')
    ax.set_ylabel('Num_Customers')

plt.suptitle('Impacto de TouristEvent no Número de Clientes', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()


print("\n\n=== CORRELAÇÕES (PEARSON) POR LOJA ===\n")

for store, df in dfs.items():
    corr = df[['Num_Employees', 'Pct_On_Sale', 'TouristEvent_bin', 'Num_Customers', 'Sales']].corr()
    print(f"\n--- {store.upper()} ---")
    print(corr['Sales'].sort_values(ascending=False).round(3))


fig, axes = plt.subplots(2, 2, figsize=(15, 11))
axes = axes.flatten()

for i, (store, df) in enumerate(dfs.items()):
    ax = axes[i]
    corr = df[['Num_Employees', 'Pct_On_Sale', 'TouristEvent_bin', 'Num_Customers', 'Sales']].corr()
    sns.heatmap(
        corr,
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt='.2f',
        cmap='RdYlBu_r',
        square=True,
        ax=ax,
        cbar=False
    )
    ax.set_title(f'{store.capitalize()}')

plt.suptitle('Heatmap de Correlações por Loja', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
axes = axes.flatten()

for i, (store, df) in enumerate(dfs.items()):
    ax = axes[i]
    dft = df.sort_values('Date').copy()
    dft['MA_7'] = dft['Num_Customers'].rolling(7, min_periods=1).mean()
    dft['MA_30'] = dft['Num_Customers'].rolling(30, min_periods=1).mean()

    ax.plot(dft['Date'], dft['Num_Customers'], color=colors[i], alpha=0.22, linewidth=0.9, label='Diário')
    ax.plot(dft['Date'], dft['MA_7'], color='#F44336', linewidth=1.5, label='Média móvel 7d')
    ax.plot(dft['Date'], dft['MA_30'], color='#212121', linewidth=1.8, label='Média móvel 30d')
    ax.set_title(f'{store.capitalize()}')
    ax.set_ylabel('Num_Customers')
    ax.legend(fontsize=8, loc='upper left')

plt.suptitle('Tendência de Clientes (Série + Médias Móveis)', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
axes = axes.flatten()

for i, (store, df) in enumerate(dfs.items()):
    ax = axes[i]
    dft = df.sort_values('Date').copy()
    dft['RollingStd_30'] = dft['Num_Customers'].rolling(30, min_periods=7).std()

    ax.plot(dft['Date'], dft['RollingStd_30'], color=colors[i], linewidth=1.7)
    ax.set_title(f'{store.capitalize()}')
    ax.set_ylabel('Desvio padrão 30d')

plt.suptitle('Volatilidade de Clientes (Rolling Std 30d)', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for i, (store, df) in enumerate(dfs.items()):
    ax = axes[i]
    tmp = df.copy()
    tmp['YearMonth'] = tmp['Date'].dt.to_period('M').astype(str)
    pivot = tmp.pivot_table(index='DayOfWeek', columns='Month', values='Num_Customers', aggfunc='mean')

    sns.heatmap(
        pivot,
        cmap='YlGnBu',
        annot=True,
        fmt='.0f',
        ax=ax,
        cbar=False
    )

    ax.set_title(f'{store.capitalize()}')
    ax.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], rotation=0)
    ax.set_xlabel('Mês')
    ax.set_ylabel('Dia da Semana')

plt.suptitle('Heatmap Sazonal: Dia da Semana x Mês (Média de Clientes)', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()


print("\n\n=== EDA CONCLUÍDA (FASE EXPLORATÓRIA) ===")
print("Próximo passo recomendado: modelação (forecasting/regressão) com base nas variáveis mais relevantes.")


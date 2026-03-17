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
    """Carrega o CSV da loja Richmond e cria features temporais."""
    df = pd.read_csv('richmond.csv', sep=',')
    df['Date'] = pd.to_datetime(df['Date'])
    df['Store'] = 'Richmond'

    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['DayName'] = df['Date'].dt.day_name()
    df['Month'] = df['Date'].dt.month
    df['MonthName'] = df['Date'].dt.month_name()
    df['Year'] = df['Date'].dt.year
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    df['TouristEvent_bin'] = (df['TouristEvent'].str.strip().str.lower() == 'yes').astype(int)

    print(f"[RICHMOND] Shape: {df.shape} | Datas: {df['Date'].min().date()} a {df['Date'].max().date()}")
    return df


print("=" * 60)
print("  EDA - Richmond")
print("=" * 60)

df = load_data()
cols_num = ['Num_Employees', 'Num_Customers', 'Pct_On_Sale', 'Sales']

print("\n=== ESTATÍSTICAS DESCRITIVAS ===\n")
print(df[cols_num].describe().round(2))

missing = df.isnull().sum()
print("\n=== VALORES EM FALTA ===\n")
if missing.any():
    print(missing[missing > 0])
else:
    print("Sem valores em falta.")

z = np.abs(stats.zscore(df['Num_Customers'].dropna()))
print(f"\n=== OUTLIERS (Z-score > 3) ===\nRichmond: {(z > 3).sum()} outliers")

fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(df['Date'], df['Num_Customers'], color='#9C27B0', linewidth=0.9)
ax.set_title('Richmond - Série Temporal de Clientes')
ax.set_ylabel('Clientes')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(df['Num_Customers'], kde=True, ax=ax, color='#9C27B0', bins=40)
ax.axvline(df['Num_Customers'].mean(), color='red', linestyle='--', linewidth=1.4,
           label=f"Média: {df['Num_Customers'].mean():.0f}")
ax.axvline(df['Num_Customers'].median(), color='orange', linestyle='--', linewidth=1.4,
           label=f"Mediana: {df['Num_Customers'].median():.0f}")
ax.legend()
ax.set_title('Richmond - Distribuição de Num_Customers')
plt.tight_layout()
plt.show()

day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
avg_day = df.groupby('DayName')['Num_Customers'].mean().reindex(day_order)

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(range(7), avg_day.values, color='#9C27B0', alpha=0.85)
for i in [5, 6]:
    bars[i].set_edgecolor('red')
    bars[i].set_linewidth(2)
ax.set_xticks(range(7))
ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
ax.set_title('Richmond - Média de Clientes por Dia da Semana')
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 5))
df_plot = df.copy()
df_plot['Tipo'] = df_plot['IsWeekend'].map({0: 'Dia de Semana', 1: 'Fim de Semana'})
sns.boxplot(data=df_plot, x='Tipo', y='Num_Customers', palette=['#64B5F6', '#EF9A9A'], ax=ax)
ax.set_title('Richmond - Clientes: Dia de Semana vs Fim de Semana')
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(df['Pct_On_Sale'], df['Num_Customers'], alpha=0.35, color='#9C27B0', s=18)
valid = df[['Pct_On_Sale', 'Num_Customers']].dropna()
if len(valid) > 10:
    m, b, r, p, _ = stats.linregress(valid['Pct_On_Sale'], valid['Num_Customers'])
    x_line = np.linspace(valid['Pct_On_Sale'].min(), valid['Pct_On_Sale'].max(), 100)
    ax.plot(x_line, m * x_line + b, color='red', linewidth=2, label=f'r={r:.2f}, p={p:.3f}')
    ax.legend()
ax.set_title('Richmond - Promoções vs Clientes')
ax.set_xlabel('Pct_On_Sale')
ax.set_ylabel('Num_Customers')
plt.tight_layout()
plt.show()

print("\n=== IMPACTO DE TOURISTEVENT ===\n")
base = df[df['TouristEvent_bin'] == 0]['Num_Customers'].dropna()
event = df[df['TouristEvent_bin'] == 1]['Num_Customers'].dropna()
if len(base) > 3 and len(event) > 3:
    _, p_val = stats.ttest_ind(event, base, equal_var=False)
    diff_pct = ((event.mean() - base.mean()) / base.mean()) * 100
    print(f"Sem evento={base.mean():.1f} | Com evento={event.mean():.1f} | Delta={diff_pct:.2f}% | p-valor={p_val:.4f}")

print("\n=== CORRELAÇÕES (PEARSON) ===\n")
corr = df[['Num_Employees', 'Pct_On_Sale', 'TouristEvent_bin', 'Num_Customers', 'Sales']].corr()
print(corr['Sales'].sort_values(ascending=False).round(3))

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, vmin=-1, vmax=1, center=0, annot=True, fmt='.2f', cmap='RdYlBu_r', square=True, ax=ax)
ax.set_title('Richmond - Heatmap de Correlações')
plt.tight_layout()
plt.show()

print("\nEDA Richmond concluída.")

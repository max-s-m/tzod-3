import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

np.random.seed()
syn_norm = np.random.normal(loc=300, scale=50, size=2000)
syn_exp = np.random.exponential(scale=100, size=2000)

df = pd.read_csv('france_house_06_09.csv', sep=';', low_memory=False)
df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
real_data = df['Global_active_power'].dropna().values
real_sample = np.random.choice(real_data, size=5000, replace=False)

mu_real, std_real = stats.norm.fit(real_sample)
loc_real, scale_real = stats.expon.fit(real_sample)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.set_style("whitegrid")

sns.histplot(syn_norm, bins=30, stat='density', color='blue', alpha=0.4, ax=axes[0])
x_n = np.linspace(syn_norm.min(), syn_norm.max(), 100)
axes[0].plot(x_n, stats.norm.pdf(x_n, 300, 50), 'b-', lw=2)
axes[0].set_title('Normal dist')
axes[0].set_xlabel('Energy (kwh)')

sns.histplot(syn_exp, bins=30, stat='density', color='green', alpha=0.4, ax=axes[1])
x_e = np.linspace(syn_exp.min(), syn_exp.max(), 100)
axes[1].plot(x_e, stats.expon.pdf(x_e, scale=100), 'g-', lw=2)
axes[1].set_title('Exp dist')
axes[1].set_xlabel('Energy (kwh)')

sns.histplot(real_sample, bins=50, stat='density', color='gray', alpha=0.5, ax=axes[2])
x_r = np.linspace(real_sample.min(), real_sample.max(), 100)
axes[2].plot(x_r, stats.norm.pdf(x_r, mu_real, std_real), 'b--', lw=2, label='Normal fit')
axes[2].plot(x_r, stats.expon.pdf(x_r, loc_real, scale_real), 'g--', lw=2, label='Exp fit')
axes[2].set_title('House data (global_active_power)')
axes[2].set_xlabel('Power (kw)')
axes[2].legend()

plt.tight_layout()
plt.show()

print(f"House avg: {mu_real:.2f}, Deviation: {std_real:.2f}")
print(f"Exponent scale: {scale_real:.2f}")
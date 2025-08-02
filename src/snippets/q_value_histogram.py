import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

# Set file paths
Q_SOURCE_FILE = "../../data/processed/Q Table - Source.csv"
Q_TARGET_FILE = "../../data/processed/Q Table - Target.csv"

# Load Q-tables
q_source = pd.read_csv(Q_SOURCE_FILE, index_col=0)
q_target = pd.read_csv(Q_TARGET_FILE, index_col=0)

# Flatten and filter non-zero entries
qs = q_source.values.flatten()
qs = qs[qs != 0]
qt = q_target.values.flatten()
qt = qt[qt != 0]

# Prepare KDE curves
xs = np.linspace(min(qs.min(), qt.min()), max(qs.max(), qt.max()), 200)
kde_s = gaussian_kde(qs)
kde_t = gaussian_kde(qt)

# Plot
fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Source Q-table
axs[0].hist(qs, bins=12, color='#34699A', edgecolor='black', alpha=0.7, density=True, label='Histogram')
axs[0].plot(xs, kde_s(xs), color='#113F67', linewidth=2, label='KDE')
axs[0].set_title('Source Q-table')
axs[0].set_xlabel('Q-value')
axs[0].set_ylabel('Density')
axs[0].grid(axis='y', linestyle='--', alpha=0.6)
axs[0].legend()

# Target Q-table
axs[1].hist(qt, bins=12, color='#D92C54', edgecolor='black', alpha=0.7, density=True, label='Histogram')
axs[1].plot(xs, kde_t(xs), color='#932F67', linewidth=2, label='KDE')
axs[1].set_title('Target Q-table')
axs[1].set_xlabel('Q-value')
axs[1].grid(axis='y', linestyle='--', alpha=0.6)
axs[1].legend()

plt.suptitle('Distribution of Non-zero Q-values (Source vs Target Q-table)')
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig('../../results/plots/q_value_histogram_side_by_side.png', dpi=300)
plt.show()

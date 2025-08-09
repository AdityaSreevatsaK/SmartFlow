import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from vanilla_python.constants import Q_SOURCE_FILE

# Load Q-table (Source)
q = pd.read_csv(Q_SOURCE_FILE, index_col=0)

plt.figure(figsize=(8, 6))
sns.heatmap(q, annot=True, fmt=".1f", cmap="mako", linewidths=0.5, cbar_kws={"label": "Q-value"})
plt.title("Q-table Heatmap (Source Table)")
plt.xlabel("Target Station")
plt.ylabel("Source Station")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("../../results/plots/q_table_heatmap_source.png", dpi=300)
plt.show()

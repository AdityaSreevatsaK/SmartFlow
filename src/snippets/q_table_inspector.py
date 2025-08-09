import pandas as pd

from vanilla_python.constants import Q_SOURCE_FILE, Q_TARGET_FILE


def inspect_q_table(path, label):
    q = pd.read_csv(path, index_col=0)
    print(f"--- {label} Q-Table ---")
    print("Shape:", q.shape)
    print("Min Q-value:", q.values.min())
    print("Max Q-value:", q.values.max())
    print("Mean Q-value:", q.values.mean())
    print("Non-zero entries:", (q.values != 0).sum())
    print("Top 5 Q-values:")
    print(q.stack().sort_values(ascending=False).head())


if __name__ == "__main__":
    inspect_q_table(Q_SOURCE_FILE, "Source")
    inspect_q_table(Q_TARGET_FILE, "Target")

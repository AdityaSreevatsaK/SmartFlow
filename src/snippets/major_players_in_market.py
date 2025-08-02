import matplotlib.pyplot as plt
import numpy as np

providers = [
    "Lime (global)",
    "Nextbike (global)",
    "Vélib’ (Paris)",
    "Citi Bike (NYC)",
    "Bicing (Barcelona)",
    "Divvy (Chicago)",
    "Bike Share Toronto",
    "Capital Bikeshare (DC)",
    "Bluebikes (Boston)"
]
rides_millions = np.array([200, 51, 49.3, 44, 18, 11, 7, 6.1, 4.7])

# Sort descending
order = np.argsort(-rides_millions)
providers_sorted = [providers[i] for i in order]
rides_sorted = rides_millions[order]

# Cumulative share
cum_percent = np.cumsum(rides_sorted) / rides_sorted.sum() * 100

fig, ax1 = plt.subplots(figsize=(10, 6))

bar_color = "royalblue"
line_color = "darkorange"

bars = ax1.bar(np.arange(len(providers_sorted)), rides_sorted, color=bar_color)
ax1.set_ylabel("Annual trips (millions)")
ax1.set_xticks(np.arange(len(providers_sorted)))
ax1.set_xticklabels(providers_sorted, rotation=45, ha='right')

# Annotate bars
for rect, value in zip(bars, rides_sorted):
    height = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width() / 2, height + 1.5, f"{value:.1f}",
             ha='center', va='bottom', fontsize=9)

# Pareto line
ax2 = ax1.twinx()
ax2.plot(np.arange(len(providers_sorted)), cum_percent, marker='o', color=line_color, linewidth=2)
ax2.set_ylabel("Cumulative share of trips (%)", color=line_color)
ax2.tick_params(axis='y', labelcolor=line_color)
ax2.set_ylim(0, 110)

# 80% reference
ax2.axhline(80, linestyle='--', linewidth=1, color='gray')

fig.suptitle("Major Bike-Share Operators – 2024 Ridership & Cumulative Share")
fig.tight_layout(rect=(0, 0.03, 1, 0.95))
plt.savefig('../../images/bike_share_2024_ridership.png')
plt.show()

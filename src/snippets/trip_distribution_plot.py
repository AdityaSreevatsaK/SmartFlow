import matplotlib.pyplot as plt
import pandas as pd

from vanilla_python.constants import TRIPS_FILE

# Use your target date for filtering
TARGET_DATE = pd.to_datetime("2016-07-01").date()

# Load and filter trip data
df = pd.read_csv(TRIPS_FILE)
df = df.dropna(subset=["Start Station Name"])
df["Start Time"] = pd.to_datetime(df["Start Time"], errors="coerce")
df = df[df["Start Time"].dt.date == TARGET_DATE]

# Count trips by start station
trip_counts = df['Start Station Name'].value_counts().sort_values(ascending=False)

# Plot as bar and line graph
plt.figure(figsize=(14, 7))
ax = trip_counts.plot(kind='bar', color='#7FFF00', label='Trip Count')
trip_counts.plot(kind='line', color='darkblue', marker='o', linewidth=2, label='Trend', ax=ax)

plt.xlabel('Station')
plt.ylabel('Number of Trips')
plt.title('Trip Count Distribution Across Stations (July 1, 2016)')
plt.tight_layout()
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()

# Add this to give more space for x-labels
plt.subplots_adjust(bottom=0.30)

plt.savefig('../../results/plots/trip_count_histogram.png', dpi=300)
plt.show()

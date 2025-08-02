from smartflow import calculate_thresholds, get_top_stations, load_and_filter
from constants import TARGET_DATE

if __name__ == "__main__":
    # Load and filter data
    df = load_and_filter(TARGET_DATE)
    print("Trips on target date:", len(df))

    # Get top stations and counts
    top_stations = get_top_stations(df)
    trip_counts = df['Start Station Name'].value_counts()
    print("Trip counts for all stations:\n", trip_counts.head(20))
    print("Top stations:", top_stations)

    # Thresholds for each station
    thresholds = calculate_thresholds(trip_counts)
    print("Thresholds for stations:\n", thresholds)

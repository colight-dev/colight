# | colight: mostly-prose

# Building a Data Analysis Workflow with Colight
#
# When working with data, you often need to explore, transform, and visualize it in various ways. Colight provides a seamless workflow that combines data manipulation with interactive visualization. Let's walk through a real-world example.

import numpy as np
import colight.plot as Plot
from datetime import datetime, timedelta

# ## Loading and Exploring Data
#
# For this example, we'll simulate some time-series data representing daily website traffic with seasonal patterns:

# Generate dates for one year
start_date = datetime(2023, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(365)]

# Simulate traffic with weekly and seasonal patterns
base_traffic = 1000
weekly_pattern = np.array([0.8, 0.9, 1.0, 1.0, 1.1, 1.3, 0.7])  # Mon-Sun
seasonal_pattern = np.sin(np.linspace(0, 2 * np.pi, 365)) * 200 + base_traffic

# Add weekly variations
daily_traffic = []
for i, date in enumerate(dates):
    weekly_factor = weekly_pattern[date.weekday()]
    seasonal_value = seasonal_pattern[i]
    noise = np.random.normal(0, 50)
    daily_traffic.append(seasonal_value * weekly_factor + noise)

daily_traffic = np.array(daily_traffic)

# Let's visualize the raw data:
Plot.line(
    {"x": dates, "y": daily_traffic},
    {"stroke": "steelblue", "strokeWidth": 1, "opacity": 0.7},
) + {"title": "Daily Website Traffic (2023)", "xlabel": "Date", "ylabel": "Visitors"}

# ## Identifying Patterns
#
# The raw data is noisy. Let's add a moving average to see the underlying trend:

# Calculate 7-day moving average
window = 7
moving_avg = np.convolve(daily_traffic, np.ones(window) / window, mode="valid")
ma_dates = dates[window - 1 :]

# Visualize both raw and smoothed data
traffic_plot = (
    Plot.line(
        {"x": dates, "y": daily_traffic},
        {"stroke": "lightblue", "strokeWidth": 1, "opacity": 0.5},
    )
    + Plot.line(
        {"x": ma_dates, "y": moving_avg}, {"stroke": "darkblue", "strokeWidth": 2}
    )
    + {"title": "Traffic with 7-day Moving Average"}
)

traffic_plot

# ## Interactive Analysis
#
# Let's add controls to explore different time windows:

window_slider = Plot.Slider(
    key="window",
    label="Moving Average Window:",
    showValue=True,
    range=[3, 30],
    step=1,
    init=7,
)

# Create a plot that updates based on the slider
interactive_analysis = (
    Plot.line(
        {"x": dates, "y": daily_traffic},
        {"stroke": "lightblue", "strokeWidth": 1, "opacity": 0.5},
    )
    + Plot.js("""
    // Calculate moving average dynamically
    const window = $state.window;
    const ma_data = [];
    
    for (let i = window - 1; i < data.length; i++) {
        let sum = 0;
        for (let j = 0; j < window; j++) {
            sum += data[i - j].y;
        }
        ma_data.push({x: data[i].x, y: sum / window});
    }
    
    return Plot.line(ma_data, {stroke: "darkblue", strokeWidth: 2}).plot();
""")
    + {"title": "Interactive Moving Average Analysis"}
)

interactive_analysis | window_slider

# ## Analyzing Weekly Patterns
#
# Let's aggregate by day of week to see if there are consistent patterns:

# Group by day of week
weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
weekday_traffic = {day: [] for day in weekday_names}

for i, date in enumerate(dates):
    weekday = weekday_names[date.weekday()]
    weekday_traffic[weekday].append(daily_traffic[i])

# Calculate averages
avg_by_weekday = [(day, np.mean(traffic)) for day, traffic in weekday_traffic.items()]

# Create a bar chart
Plot.barY(avg_by_weekday, {"x": 0, "y": 1, "fill": "steelblue"}) + {
    "title": "Average Traffic by Day of Week",
    "xlabel": "Day",
    "ylabel": "Average Visitors",
}

# ## Monthly Trends
#
# Finally, let's look at monthly aggregations to see seasonal patterns:

# Group by month
monthly_traffic = {}
for i, date in enumerate(dates):
    month = date.strftime("%B")
    if month not in monthly_traffic:
        monthly_traffic[month] = []
    monthly_traffic[month].append(daily_traffic[i])

# Calculate monthly averages
month_order = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
monthly_avg = [
    (month, np.mean(monthly_traffic[month]))
    for month in month_order
    if month in monthly_traffic
]

# Create a line chart for monthly trends
Plot.line(
    monthly_avg,
    {"x": 0, "y": 1, "stroke": "darkgreen", "strokeWidth": 2, "marker": True},
) + {"title": "Monthly Traffic Trends", "xlabel": "Month", "ylabel": "Average Visitors"}

# ## Key Insights
#
# From our analysis, we can see:
#
# 1. **Weekly Pattern**: Traffic peaks on Fridays and Saturdays, with a significant drop on Sundays
# 2. **Seasonal Trend**: There's a clear seasonal pattern with higher traffic in summer months
# 3. **Smoothing Effect**: The moving average helps identify trends by filtering out daily noise
#
# This workflow demonstrates how Colight enables rapid data exploration with immediate visual feedback. The combination of Python's data manipulation capabilities with interactive visualizations makes it ideal for iterative analysis.
#
# Try adjusting the moving average window to see how it affects pattern detection!

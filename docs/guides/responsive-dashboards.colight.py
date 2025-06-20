# | colight: mostly-prose

# Creating Responsive Dashboards
#
# Learn how to build interactive dashboards that adapt to different screen sizes and update in real-time.

import numpy as np
import colight.plot as Plot

# ## Dashboard Basics
#
# A dashboard combines multiple visualizations into a cohesive interface. Colight's layout system makes it easy to create responsive designs.

# Generate sample metrics data
np.random.seed(42)
hours = list(range(24))
cpu_usage = 40 + 20 * np.sin(np.linspace(0, 2 * np.pi, 24)) + np.random.normal(0, 5, 24)
memory_usage = (
    60 + 10 * np.cos(np.linspace(0, 2 * np.pi, 24)) + np.random.normal(0, 3, 24)
)
requests_per_hour = (
    1000 + 500 * np.sin(np.linspace(0, 4 * np.pi, 24)) + np.random.normal(0, 100, 24)
)

# Create individual metric cards
cpu_chart = (
    Plot.line({"x": hours, "y": cpu_usage}, {"stroke": "red", "strokeWidth": 2})
    + Plot.ruleY([50], {"stroke": "orange", "strokeDasharray": "5,5"})
    + {"title": "CPU Usage (%)", "height": 200}
)

memory_chart = (
    Plot.line({"x": hours, "y": memory_usage}, {"stroke": "blue", "strokeWidth": 2})
    + Plot.ruleY([70], {"stroke": "orange", "strokeDasharray": "5,5"})
    + {"title": "Memory Usage (%)", "height": 200}
)

requests_chart = Plot.barY(
    {"x": hours, "y": requests_per_hour}, {"fill": "green", "opacity": 0.7}
) + {"title": "Requests per Hour", "height": 200}

# Compose into a dashboard layout
(cpu_chart | memory_chart) & requests_chart

# ## Real-time Updates
#
# Dashboards often need to display live data. Use sliders with fps to simulate real-time updates:

time_slider = Plot.Slider(
    key="time",
    label="Current Hour:",
    showValue=True,
    range=[0, 23],
    step=1,
    fps=2,  # Update every 0.5 seconds
    init=0,
)

# Create a real-time monitoring view
realtime_dashboard = (
    Plot.initialState({"cpu_usage": cpu_usage})
    + Plot.line(
        {"x": hours, "y": cpu_usage},
        {"stroke": "red", "strokeWidth": 2, "opacity": 0.3},
    )
    + Plot.dot(
        {
            "x": [Plot.js("$state.time")],
            "y": [Plot.js("$state.cpu_usage[Math.floor($state.time)]")],
        },
        {"fill": "red", "r": 5},
    )
    + Plot.text(
        [
            {
                "x": Plot.js("$state.time"),
                "y": Plot.js("$state.cpu_usage[Math.floor($state.time)]"),
            }
        ],
        {
            "text": Plot.js(
                "$state.cpu_usage[Math.floor($state.time)].toFixed(1) + '%'"
            ),
            "dy": -10,
            "fontSize": 12,
        },
    )
    + {"title": "Live CPU Monitoring", "height": 250}
)

realtime_dashboard | time_slider

# ## Responsive Grid Layouts
#
# For more complex dashboards, you can create grid-like layouts that adapt to screen size:

# Create metric summary cards
current_cpu = cpu_usage[-1]
current_memory = memory_usage[-1]
total_requests = np.sum(requests_per_hour)

# Status indicators with color coding
cpu_status = Plot.barY(
    [{"metric": "CPU", "value": current_cpu}],
    {"x": "metric", "y": "value", "fill": "red" if current_cpu > 80 else "green"},
) + {"title": f"CPU: {current_cpu:.1f}%", "height": 100}

memory_status = Plot.barY(
    [{"metric": "Memory", "value": current_memory}],
    {"x": "metric", "y": "value", "fill": "orange" if current_memory > 70 else "green"},
) + {"title": f"Memory: {current_memory:.1f}%", "height": 100}

requests_status = Plot.text(
    [{"text": f"{total_requests:,.0f}", "x": 0.5, "y": 0.5}],
    {"fontSize": 24, "fontWeight": "bold", "textAnchor": "middle"},
) + {"title": "Total Requests (24h)", "height": 100, "domain": [[0, 1], [0, 1]]}

# Compose the status cards
status_row = cpu_status | memory_status | requests_status

# ## Interactive Filtering
#
# Add controls to filter and explore your data:

metric_selector = Plot.Slider(
    key="metric",
    label="Select Metric:",
    showValue=False,
    range=[0, 2],
    step=1,
    init=0,
    ticks=[0, 1, 2],
    tickFormat=["CPU", "Memory", "Requests"],
)

# Create a plot that changes based on selection
filtered_view = Plot.js("""
    const metrics = [
        {data: ${JSON.stringify(cpu_usage)}, color: "red", title: "CPU Usage (%)"},
        {data: ${JSON.stringify(memory_usage)}, color: "blue", title: "Memory Usage (%)"},
        {data: ${JSON.stringify(requests_per_hour)}, color: "green", title: "Requests per Hour"}
    ];
    
    const selected = metrics[Math.floor($state.metric)];
    
    return Plot.line(
        selected.data.map((y, x) => ({x, y})),
        {stroke: selected.color, strokeWidth: 2}
    ).plot({title: selected.title, height: 300});
""")

# Combine status overview with filtered detail view
status_row & (filtered_view | metric_selector)

# ## Best Practices
#
# When building dashboards with Colight:
#
# 1. **Use consistent scales** - Make comparisons easier by aligning axes
# 2. **Add context** - Include reference lines, thresholds, or historical ranges
# 3. **Progressive disclosure** - Start with overview, allow drilling into details
# 4. **Responsive design** - Test layouts at different sizes using `|` and `&`
# 5. **Performance** - For large datasets, consider aggregation and sampling
#
# ## Next Steps
#
# - Explore [real-time data sources](../interactivity/tail.py) for live updates
# - Learn about [custom themes](../learn-more/color.py) for consistent styling
# - Check out [export options](../export/images_and_video.py) for sharing dashboards

# | colight: mostly-prose

# Getting Started with Colight
#
# Colight is a powerful data visualization library that combines the expressiveness of Python with the interactivity of JavaScript. This guide will walk you through creating your first interactive visualizations.

import numpy as np
import colight.plot as Plot

# ## Your First Plot
#
# Let's start with a simple line plot. In Colight, we use the Observable Plot API through Python:

x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

# Create a line plot
Plot.line({"x": x, "y": y}, {"stroke": "steelblue", "strokeWidth": 2})

# ## Adding Interactivity
#
# One of Colight's strengths is its ability to create interactive visualizations. Let's add a slider to control the amplitude:

slider = Plot.Slider(
    key="amplitude",
    label="Amplitude:",
    showValue=True,
    range=[0.1, 2],
    step=0.1,
    init=1,
)

# Now we can use this slider value in our plot:
interactive_plot = Plot.line(
    {"x": x},
    {"y": Plot.js("(d, i) => $state.amplitude * Math.sin(d.x)")},
) + {"height": 300}

interactive_plot | slider

# ## Composing Multiple Visualizations
#
# Colight makes it easy to compose multiple plots. You can arrange them horizontally with `|` or vertically with `&`:

# Create two different wave forms
sine_wave = Plot.line({"x": x, "y": np.sin(x)}, {"stroke": "blue"}) + {
    "title": "Sine Wave"
}
cosine_wave = Plot.line({"x": x, "y": np.cos(x)}, {"stroke": "red"}) + {
    "title": "Cosine Wave"
}

# Display them side by side
sine_wave | cosine_wave

# ## Working with Real Data
#
# Let's create a more realistic example using random data to simulate stock prices:

np.random.seed(42)
days = 100
dates = np.arange(days)
prices = 100 + np.cumsum(np.random.randn(days) * 2)

# Create a stock price chart with area fill
stock_chart = (
    Plot.area({"x": dates, "y": prices}, {"fill": "lightblue", "fillOpacity": 0.3})
    + Plot.line({"x": dates, "y": prices}, {"stroke": "darkblue", "strokeWidth": 2})
    + Plot.ruleY([100], {"stroke": "gray", "strokeDasharray": "5,5"})
    + {"title": "Stock Price Simulation", "xlabel": "Days", "ylabel": "Price ($)"}
)

stock_chart

# ## Next Steps
#
# Now that you've seen the basics, you can:
#
# - Explore more [mark types](../learn-more/plot-options.py) like scatter plots, bar charts, and heatmaps
# - Learn about [advanced interactivity](../interactivity/events.py) with mouse events
# - Dive into [3D visualizations](../scene3d.py) with Scene3D
# - Check out the [API documentation](../api/plot.md) for a complete reference
#
# Happy visualizing! 🎨

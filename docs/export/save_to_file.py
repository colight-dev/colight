# %% [markdown]

# ## Embedding Visuals
#
# Colight supports exporting visuals for embedding in websites:
#
# ### HTML Export
#
# You can export any visualization as a standalone HTML file:

# %%
import colight.plot as Plot

# Create a simple visualization
data = [
    {"category": "A", "value": 10},
    {"category": "B", "value": 20},
    {"category": "C", "value": 15},
    {"category": "D", "value": 25},
]

p = Plot.barY(data, {"x": "category", "y": "value", "fill": "category"})

# Save as HTML file
p.save_html("my_visual.html")

# %% [markdown]
#
# ### Data Export (.colight)
#
# For more efficient embedding with binary data support, you can export visualizations as `.colight` files:

# %%
import numpy as np

# Create a visualization with binary data
raster_data = np.random.rand(50, 50)
p = Plot.raster(raster_data)

# Save as .colight file
p.save_file("my_visual.colight")

# %% [markdown]
#
# ### Embedding in Websites
#
# To embed a `.colight` file in your website, you have several options:
#
# **Option 1: Simple embedding with data-src attribute**
#
# ```html
# <!-- Import the Colight embed script -->
# <script type="module" src="https://cdn.jsdelivr.net/npm/@colight/core/embed.js"></script>
#
# <!-- Embed the visualization -->
# <div class="colight-embed" data-src="./my_visualization.colight"></div>
# ```
#
# **Option 2: Programmatic embedding**
#
# ```html
# <script type="module">
#   import { loadVisual } from "https://cdn.jsdelivr.net/npm/@colight/core/embed.js";
#   loadVisual("#my-container", "./my_visualization.colight");
# </script>
# <div id="my-container"></div>
# ```
#
# The embed script automatically discovers and loads all visualizations with the `colight-embed` class, making it easy to embed multiple visualizations on a single page.

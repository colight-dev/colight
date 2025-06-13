# %% [markdown]

# The `colight.plot` module is an interface to the [Observable Plot](https://observablehq.com/plot/getting-started) library, with 100% coverage and a straightforward mapping between what you write in Python and how plots are typically written in JavaScript. The Python code you write produces a structured representation which is serialized and rendered in a browser environment. To use this library effectively, you will want to frequently refer to the [Observable Plot documentation](https://observablehq.com/plot/quickstart) to understand the API surface we're targeting.

# ## Marks
#
# [Marks](https://observablehq.com/plot/features/marks) are the basic visual elements used to represent data. Common marks include `line`, `dot`, `bar`, and `text`.
# Each mark type has its own set of properties that control its appearance and behavior. For example, with `line`, we can control the stroke, stroke width, and curve:

# %%
import colight.plot as Plot

six_points = [[1, 1], [2, 4], [1.5, 7], [3, 10], [2, 13], [4, 15]]

Plot.line(
    six_points,
    {
        "stroke": "steelblue",  # Set the line color
        "strokeWidth": 3,  # Set the line thickness
        "curve": "natural",  # Set the curve type
    },
)

# %% [markdown]

#
# ## Composition
#
# We can layer multiple marks and add options to plots using the `+` operator. For example, here we compose a [line mark](bylight?match=Plot.line(...\)) with a [dot mark](bylight?match=Plot.dot(...\)), then add a [frame](bylight?match=Plot.frame(\)):

# %%
line_plot = Plot.line(six_points, {"stroke": "pink", "strokeWidth": 10})
dot_plot = Plot.dot(six_points, {"fill": "purple"})

line_plot + dot_plot + Plot.frame()

# %% [markdown]
# ## Layouts
# To show more than one plot, we can compose layouts using `&` (for rows) and `|` (for columns).

# %%

line_plot & dot_plot

# %% [markdown]
# For more advanced layout options, including grids and responsive layouts, see the [Layouts guide](system-guide/layouts.py).
#
# ## Supplying Data
#
# To render a mark we must (1) supply some data, and (2) indicate how that data maps to visual properties (or _channels_, in Observable lingo). Common channels include:
#
# - `x` and `y` for 2d coordinates
# - `fill` and `stroke` for colors
# - `opacity` for alpha blending
#
# The documentation for each mark will indicate what channels are available or required.
#
# There are a few ways to supply data and map it to channels. Shorthand syntax exists to make common cases faster to specify; this tends to be appreciated by advanced users but can be tricky when getting started.
#
# Below is an example of the "base case" that Observable Plot is designed around. In this mode of working, data arrives as a list of objects, and our task is to specify how each object's properties should map to the necessary channels.

# %%
object_data = [
    {"X": 1, "Y": 2, "CATEGORY": "A"},
    {"X": 2, "Y": 4, "CATEGORY": "B"},
    {"X": 1.5, "Y": 7, "CATEGORY": "C"},
    {"X": 3, "Y": 10, "CATEGORY": "D"},
    {"X": 2, "Y": 13, "CATEGORY": "E"},
    {"X": 4, "Y": 15, "CATEGORY": "F"},
]

# %% [markdown]
#
# We always pass data as the first argument to a mark, followed by options (which may be a dictionary or keyword args). For each required or optional channel, we specify "where to find" that channel's data in each entry. Let's start with the simple case of using strings, which are simply used as keys to look up a property in each object.
# %%

Plot.dot(object_data, x="X", y="Y", fill="CATEGORY", r=20)

# %% [markdown]
# A mark takes [data](bylight?match=object_data) followed by an options dictionary, which specifies how [channel names](bylight?match="x","y","stroke","strokeWidth","r","fill") get their values.
#
# There are a few ways to specify channel values in Observable Plot:
#
# 1. A [string](bylight?match="X","Y","CATEGORY") is used to specify a property name in the data object. If it matches, that property's value is used. Otherwise, it's treated as a literal value.
# 2. A [function](bylight?match=Plot.js(...\)) will receive two arguments, `(data, index)`, and should return the desired value for the channel. We use `Plot.js` to insert a JavaScript source string - this function is evaluated within the rendering environment, and not in python.
# 3. An [array](bylight?match=[...]) provides explicit values for each data point. It should have the same length as the list passed in the first (data) position.
# 4. [Other values](bylight?match=8,None) will be used as a constant for all data points.

# %%
Plot.dot(
    object_data,
    {
        "x": "X",
        "y": "Y",
        "stroke": Plot.js("(data, index) => data.CATEGORY"),
        "strokeWidth": [1, 2, 3, 4, 5, 6],
        "r": 8,
        "fill": None,
    },
)
# %% [markdown]
#
# There are a couple of special cases to be aware of.
#
# 1. If all of your data is in **columnar** format (ie. each channel's values are in their own arrays), we can pass them in dictionary format in the first (data) position, eg. `Plot.dot({"x": [...], "y": [...]})`.
# 2. Some marks, like `Plot.dot` and `Plot.line`, which expect `x/y` coordinates, will accept an array of arrays without any need to manually map channel names. eg/ `Plot.line([[x1, y1], [x2, y2], ...])` .

# %% [markdown]
# ## Data Serialization
#
# Data is passed to the JavaScript runtime as JSON with binary buffer support. The serialization process handles various data types:
#
# | Data Type | Conversion |
# |-----------|------------|
# | Basic types (str, int, bool) | Direct JSON serialization |
# | Binary data (bytes, bytearray, memoryview) | Stored in binary buffers with reference |
# | NumPy/JAX arrays | Converted to binary buffers with dtype and shape metadata |
# | Objects with `for_json` method | `object.for_json()` result is serialized |
# | Datetime objects | Converted to JavaScript `Date` |
# | Iterables (list, tuple) | Recursively serialized |
# | Callable objects | Converted to JavaScript callback functions (widget mode only) |
#
# Binary data is handled efficiently by storing the raw bytes in separate buffers rather than base64 encoding in JSON. This is particularly important for large numeric arrays and binary data.
#
# There is a 100mb limit on the size of initial data and subsequent messages (per message).
#
# The serialization process also handles state management for interactive widgets, collecting initial state and synced keys to enable bidirectional updates between Python and JavaScript. For more details on state management, see the [State guide](system-guide/state.py).


# %% [markdown]
# ## Widgets vs HTML
#
# Colight offers two rendering modes:
#
# 1. **HTML mode**: Renders visuals as standalone HTML, ideal for embedding in web pages or exporting. Plots persist across kernel restarts.
#
# 2. **Widget mode**: Renders visuals as interactive Jupyter widgets. Enables bidirectional communication between Python and JavaScript.
#
# You can choose the rendering mode in two ways:
#
# 1. Globally, using `Plot.configure()`:

# %%
Plot.configure(display_as="widget")  # Set global rendering mode to widget

# %% [markdown]
# 2. Using a plot's [.display_as(...)](bylight) method:

# %%
categorical_data = [
    {"category": "A", "value": 10},
    {"category": "B", "value": 20},
    {"category": "C", "value": 15},
    {"category": "D", "value": 25},
]
(
    Plot.dot(categorical_data, {"x": "value", "y": "category", "fill": "category"})
    + Plot.colorLegend()
).display_as("html")

# %% [markdown]
# The global setting affects all subsequent plots unless overridden by `.display_as()`.
# You can switch between modes as needed for different use cases.

import colight.plot as Plot


def inspect(v):
    # TODO
    # handle default visualizations of python types
    return Plot.html([Plot.js("colight.api.inspect"), {"value": v}])

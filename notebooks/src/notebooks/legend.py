import colight.plot as Plot
import numpy as np

x = np.linspace(0, 10, 100)
fps_plot = (
    Plot.line([[x, y] for x, y in zip(x, np.sin(x))], stroke=Plot.constantly("Gen3D"))
    + Plot.line(
        [[x, y] for x, y in zip(x, np.sin(x + 0.5))],
        stroke=Plot.constantly("Gaussian Splatting"),
    )
    + Plot.line(
        [[x, y] for x, y in zip(x, np.cos(x + 0.5))],
        stroke=Plot.constantly("FoundationPose"),
    )
    + {
        "x": {"label": "Frames per second"},
        "y": {"label": "Gaussian count"},
        "height": 160,
        "onPlotCreate": Plot.js(
            "(plot) => $state.update({'legend': plot.legend('color')})"
        ),
    }
)

(legend_example := (fps_plot | ["div.flex.justify-center", Plot.js("$state.legend")]))

# legend_example.save_pdf('./scratch/legend.pdf')

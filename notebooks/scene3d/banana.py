from colight.scene3d import Ellipsoid
import pickle
import colight.plot as Plot

with open("./notebooks/scene3d/banana_gaussians.pkl", "rb") as f:
    banana_gaussians = pickle.load(f)


def render_gaussians(bananas):
    """
    Renders a Gen3D state's Gaussian ellipsoids using colight.scene3d.

    Parameters:
        state: A Gen3D state object containing Gaussian parameters

    Returns:
        A colight Scene3D containing the rendered ellipsoids
    """

    # Convert covariances to ellipsoid parameters using gen3d's function

    return Ellipsoid(
        centers=bananas["xyz"],
        half_sizes=bananas["half_sizes"] * 2.5,
        quaternions=bananas["quaternions"],
        colors=bananas["colors"],
        fill_mode="MajorWireframe",
    ) + {
        "camera": Plot.ref(
            {
                "position": [0.106653, 0.085635, 0.135901],
                "target": [0.000000, 0.000000, 0.000000],
                "up": [0.000000, 1.000000, 0.000000],
                "fov": 45,
            },
            "camera",
        ),
        "onCameraChange": Plot.js("(camera) => $state.update({camera})"),
    }


Plot.Row(*[render_gaussians(g) for g in banana_gaussians])

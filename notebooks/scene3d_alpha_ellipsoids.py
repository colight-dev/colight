import numpy as np
from genstudio.scene3d import Ellipsoid
import genstudio.plot as Plot
import math


def create_gaussian_ellipsoids_scene():
    """Create a scene with clusters of gaussian-distributed ellipsoids."""

    # Create 15 cluster center positions
    n_clusters = 15
    cluster_centers = np.random.uniform(low=-3, high=3, size=(n_clusters, 3))

    # Generate 30 ellipsoids around each center with gaussian distribution
    n_ellipsoids_per_cluster = 30
    centers = []
    colors = []
    alphas = []
    radii = []

    for center in cluster_centers:
        # Generate positions with gaussian distribution around center
        positions = np.random.normal(
            loc=center, scale=0.5, size=(n_ellipsoids_per_cluster, 3)
        )
        centers.extend(positions)

        # Random colors for this cluster with some variation
        base_color = np.random.random(3)
        cluster_colors = np.clip(
            np.random.normal(
                loc=base_color, scale=0.1, size=(n_ellipsoids_per_cluster, 3)
            ),
            0,
            1,
        )
        colors.extend(cluster_colors)

        # Random alpha values that decrease with distance from center
        distances = np.linalg.norm(positions - center, axis=1)
        cluster_alphas = np.clip(1.0 - distances / 2, 0.1, 0.8)
        alphas.extend(cluster_alphas)

        # Random radii that decrease with distance from center
        base_radius = np.random.uniform(0.1, 0.2)
        cluster_radii = np.array(
            [[base_radius, base_radius, base_radius]] * n_ellipsoids_per_cluster
        )
        cluster_radii *= np.clip(1.0 - distances[:, np.newaxis] / 3, 0.3, 1.0)
        radii.extend(cluster_radii)

    # Create the scene
    scene = Ellipsoid(
        centers=np.array(centers),
        radii=np.array(radii),
        colors=np.array(colors),
        alphas=np.array(alphas),
    ) | Plot.initialState(
        {
            "camera": {
                "position": [8, 8, 8],
                "target": [0, 0, 0],
                "up": [0, 1, 0],
                "fov": math.degrees(math.pi / 3),
                "near": 0.01,
                "far": 100.0,
            }
        }
    )

    return scene


create_gaussian_ellipsoids_scene()

import numpy as np
from colight.scene3d import Ellipsoid, Cuboid, deco, Scene
import colight.plot as Plot
import math
from colight.plot import js


def generate_cluster_centers(n_clusters, bounds=(-3, 3)):
    """Generate random cluster center positions."""
    return np.random.uniform(low=bounds[0], high=bounds[1], size=(n_clusters, 3))


def generate_cluster_positions(center, n_items, spread=0.5):
    """Generate gaussian-distributed positions around a center."""
    return np.random.normal(loc=center, scale=spread, size=(n_items, 3))


def generate_cluster_colors(n_items, base_color=None, variation=0.1):
    """Generate colors for a cluster with variation around a base color."""
    if base_color is None:
        base_color = np.random.random(3).astype(np.float32)
    return np.clip(
        np.random.normal(loc=base_color, scale=variation, size=(n_items, 3)),
        0,
        1,
    ).astype(np.float32)


def calculate_distance_based_values(
    positions, center, alpha_range=(0.01, 0.8), scale_range=(0.3, 1.0)
):
    """Calculate alpha and scale values based on distance from center."""
    distances = np.linalg.norm(positions - center, axis=1)
    alphas = np.clip(1.0 - distances / 2, alpha_range[0], alpha_range[1])
    scales = np.clip(1.0 - distances / 3, scale_range[0], scale_range[1])
    return alphas, scales


def get_default_camera():
    """Return default camera settings."""
    return {
        "camera": {
            "position": [8, 8, 8],
            "target": [0, 0, 0],
            "up": [0, 1, 0],
            "fov": math.degrees(math.pi / 3),
            "near": 0.01,
            "far": 100.0,
        }
    }


def create_gaussian_ellipsoids_scene():
    """Create a scene with clusters of gaussian-distributed ellipsoids."""
    n_clusters = 15
    n_ellipsoids_per_cluster = 30

    cluster_centers = generate_cluster_centers(n_clusters)

    centers = []
    colors = []
    alphas = []
    half_sizes = []

    for center in cluster_centers:
        positions = generate_cluster_positions(center, n_ellipsoids_per_cluster)
        centers.extend(positions)

        cluster_colors = generate_cluster_colors(n_ellipsoids_per_cluster)
        colors.extend(cluster_colors)

        cluster_alphas, scales = calculate_distance_based_values(positions, center)
        alphas.extend(cluster_alphas)

        base_half_size = np.random.uniform(0.05, 0.1)
        cluster_half_sizes = np.array(
            [[base_half_size, base_half_size, base_half_size]]
            * n_ellipsoids_per_cluster
        )
        cluster_half_sizes *= scales[:, np.newaxis]
        half_sizes.extend(cluster_half_sizes)

    return Ellipsoid(
        centers=np.array(centers),
        half_sizes=np.array(half_sizes),
        colors=np.array(colors),
        alphas=np.array(alphas),
    ) | Plot.initialState(get_default_camera())


def create_gaussian_cuboids_scene():
    """Create a scene with clusters of gaussian-distributed cuboids."""
    n_clusters = 12
    n_cuboids_per_cluster = 25

    cluster_centers = generate_cluster_centers(n_clusters)

    centers = []
    colors = []
    alphas = []
    half_sizes = []
    quaternions = []

    for center in cluster_centers:
        positions = generate_cluster_positions(
            center, n_cuboids_per_cluster, spread=0.6
        )
        centers.extend(positions)

        cluster_colors = generate_cluster_colors(n_cuboids_per_cluster)
        colors.extend(cluster_colors)

        cluster_alphas, scales = calculate_distance_based_values(
            positions, center, alpha_range=(0.15, 0.85), scale_range=(0.4, 1.0)
        )
        alphas.extend(cluster_alphas)

        # Random base half_size for this cluster
        base_half_size = np.random.uniform(0.08, 0.16)
        cluster_half_sizes = np.array(
            [[base_half_size, base_half_size, base_half_size]] * n_cuboids_per_cluster
        )
        cluster_half_sizes *= scales[:, np.newaxis]
        half_sizes.extend(cluster_half_sizes)

        # Generate random quaternions for each cuboid
        # Format: [x,y,z,w] where w is cos(theta/2) and x,y,z is the normalized axis * sin(theta/2)
        angles = np.random.uniform(0, 2 * np.pi, size=n_cuboids_per_cluster)
        axes = np.random.normal(0, 1, size=(n_cuboids_per_cluster, 3))
        axes = axes / np.linalg.norm(axes, axis=1)[:, np.newaxis]  # Normalize axes

        cluster_quaternions = np.zeros((n_cuboids_per_cluster, 4))
        cluster_quaternions[:, :3] = axes * np.sin(angles / 2)[:, np.newaxis]
        cluster_quaternions[:, 3] = np.cos(angles / 2)  # w component
        quaternions.extend(cluster_quaternions)
    return Cuboid(
        centers=np.array(centers),
        half_sizes=np.array(half_sizes),
        colors=np.array(colors),
        alphas=np.array(alphas),
        quaternions=np.array(quaternions),
    ) | Plot.initialState(get_default_camera())


# Create and display both scenes
ellipsoid_scene = create_gaussian_ellipsoids_scene()
cuboid_scene = create_gaussian_cuboids_scene()

# Display ellipsoid scene
ellipsoid_scene
# %%
# Display cuboid scene
cuboid_scene
# %%


def create_animated_clusters_scene(
    n_frames=10, n_clusters=15, n_ellipsoids_per_cluster=5000
):
    """Create an animated scene where cluster centers stay fixed but members regenerate each frame.
    Each cluster is rendered as a separate Ellipsoid component.

    Returns a Plot layout with animation controls.
    """
    # Fixed cluster centers that won't change between frames
    cluster_centers = generate_cluster_centers(n_clusters)

    # Generate one fixed random color per cluster
    cluster_fixed_colors = generate_cluster_colors(n_clusters)

    # Pre-generate all frame data for each cluster
    clusters_frames = []
    for cluster_idx in range(n_clusters):
        cluster_frames = {"centers": [], "colors": [], "alphas": [], "half_sizes": []}

        center = cluster_centers[cluster_idx]
        cluster_color = cluster_fixed_colors[cluster_idx]

        for frame in range(n_frames):
            # Generate new random positions/properties for this cluster
            positions = generate_cluster_positions(
                center, n_ellipsoids_per_cluster, spread=0.2
            )

            # Use the fixed color for this cluster
            cluster_colors = np.tile(cluster_color, (n_ellipsoids_per_cluster, 1))

            cluster_alphas, scales = calculate_distance_based_values(
                positions, center, alpha_range=(0.15, 0.85), scale_range=(0.4, 1.0)
            )

            base_half_size = np.random.uniform(0.08, 0.16)
            cluster_half_sizes = np.array(
                [[base_half_size, base_half_size, base_half_size]]
                * n_ellipsoids_per_cluster
            )
            cluster_half_sizes *= scales[:, np.newaxis]

            # Store arrays for this frame - ensure float32
            cluster_frames["centers"].append(
                np.array(positions, dtype=np.float32).flatten()
            )
            cluster_frames["colors"].append(
                np.array(cluster_colors, dtype=np.float32).flatten()
            )
            cluster_frames["alphas"].append(
                np.array(cluster_alphas, dtype=np.float32).flatten()
            )
            cluster_frames["half_sizes"].append(
                np.array(cluster_half_sizes, dtype=np.float32).flatten()
            )

        clusters_frames.append(cluster_frames)

    # Create the animated scene with multiple Ellipsoid components
    scene = (
        Scene(
            *[
                Ellipsoid(
                    centers=js(f"$state.clusters[{i}].centers[$state.frame]"),
                    colors=js(f"$state.clusters[{i}].colors[$state.frame]"),
                    alpha=0.5,
                    half_sizes=js(f"$state.clusters[{i}].half_sizes[$state.frame]"),
                    onHover=js(
                        f"(idx) => $state.update({{hovered: idx !== undefined ? [{i}, idx] : undefined}})"
                    ),
                    decorations=[
                        deco(
                            js(
                                f"$state.hovered && $state.hovered[0] === {i} ? [$state.hovered[1]] : []"
                            ),
                            color=[1, 0, 0],
                        )
                    ],
                )
                for i in range(n_clusters)
            ]
        )
        + {"controls": ["fps"]}
        | Plot.Slider("frame", 0, range=n_frames, fps="raf")
        | Plot.initialState(
            {
                "frame": 0,
                "clusters": clusters_frames,
                "camera": get_default_camera(),
            }
        )
    )

    return scene


# Display animated clusters scene
animated_scene = create_animated_clusters_scene()
animated_scene

# %% Test transparency sorting with overlapping ellipsoids
print("Testing transparency sorting with overlapping ellipsoids...")

test_scene = Ellipsoid(
    centers=np.array(
        [
            [0, 0, 0],  # Back ellipsoid
            [0, 0, 0.5],  # Middle ellipsoid
            [0, 0, 1.0],  # Front ellipsoid
        ]
    ),
    colors=np.array(
        [
            [0, 1, 0],  # Green for all
            [0, 1, 0],
            [0, 1, 0],
        ]
    ),
    alphas=np.array([0.9, 0.5, 0.2]),  # High to low alpha from back to front
    half_size=[0.5, 0.5, 0.5],  # Same size for all
) + (
    {
        "defaultCamera": {
            "position": [2, 2, 2],
            "target": [0, 0, 0.5],  # Center the view on middle of ellipsoids
            "up": [0, 0, 1],
            "fov": 45,
            "near": 0.1,
            "far": 100,
        }
    }
)

test_scene

# /// script
# dependencies = [
#   "scikit-learn",
# ]
# ///

# PEP 723 Example File with Colight Visualization
# This file demonstrates using inline script metadata to specify dependencies

from sklearn.datasets import load_iris

# Load and prepare the Iris dataset
iris = load_iris()

# Extract basic information about the iris dataset
feature_names = iris.feature_names
target_names = iris.target_names
n_samples, n_features = iris.data.shape

# Create a summary dictionary
{
    "dataset_name": "Iris Dataset",
    "n_samples": n_samples,
    "n_features": n_features,
    "feature_names": feature_names,
    "target_names": target_names,
    "target_counts": {
        name: sum(iris.target == i) for i, name in enumerate(target_names)
    },
}

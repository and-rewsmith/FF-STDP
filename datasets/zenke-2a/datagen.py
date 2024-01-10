import numpy as np
import pandas as pd


def generate_sequential_dataset(num_samples=10000, num_clusters=2, cluster_switch_prob=0.01, cluster_spread=0.5):
    """
    Generates a sequential dataset with rare abrupt switches between clusters.

    Parameters:
    num_samples (int): Total number of samples in the dataset.
    num_clusters (int): Number of clusters.
    cluster_switch_prob (float): Probability of switching to a different cluster.
    cluster_spread (float): Variability within the cluster.

    Returns:
    pd.DataFrame: DataFrame containing the sequential dataset.
    """
    # Initialize variables
    data = np.zeros((num_samples, 2))  # Two-dimensional data points
    current_cluster = 0

    for i in range(num_samples):
        # Decide whether to switch clusters
        if np.random.rand() < cluster_switch_prob:
            current_cluster = (current_cluster + 1) % num_clusters

        # Cluster center along the x-axis
        center_x = current_cluster * 2.0

        # Generate the data point
        data[i, 0] = np.random.normal(center_x, 0.1)  # x-coordinate
        data[i, 1] = np.random.normal(
            0, cluster_spread)  # y-coordinate with noise

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=['x', 'y'])

    return df


# Generate the dataset
sequential_data = generate_sequential_dataset()

# Save to CSV file
csv_file_path = '/mnt/data/sequential_data.csv'
sequential_data.to_csv(csv_file_path, index=False)

csv_file_path  # Return the file path for reference

import pandas as pd
import matplotlib.pyplot as plt

# Read the dataset from the CSV file
from datasets.src.zenke_2a.constants import TRAIN_DATA_PATH

if __name__ == "__main__":
    data = pd.read_csv(TRAIN_DATA_PATH)

    # Create a scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(data['x'], data['y'], alpha=0.5)
    plt.title('Scatter Plot of Sequential Data')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.grid(True)
    plt.show()

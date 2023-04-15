import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    df = pd.read_csv("covtype.data", header=None)
    
    labels = df.iloc[:, 54]
    labels = labels.values

    elevation = df.iloc[:, 0]

    # For each label, print the mean and standard deviation of elevation
    for i in range(1, 8):
        print("Label: ", i)
        print("Mean elevation: ", np.mean(elevation[labels == i]))
        print("Standard deviation of elevation: ", np.std(elevation[labels == i]))
        print("Median: ", np.median(elevation[labels == i]))

        # Plot the histogram of elevation for each label
        plt.hist(elevation[labels == i], bins=100)
        plt.title("Label: " + str(i))
        plt.xlabel("Elevation")
        plt.ylabel("Frequency")
        plt.show()
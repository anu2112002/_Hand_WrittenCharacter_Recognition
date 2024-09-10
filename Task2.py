import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Define the number of samples and features
n_samples = 10000
n_features = 784  # 28x28 images

# Generate synthetic data for handwritten characters
X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=784, n_redundant=0, n_repeated=0, n_classes=26, random_state=42)

# Reshape the data into 28x28 images
X = X.reshape(-1, 28, 28)

# Create a dataset for handwritten characters (A-Z)
characters = [chr(i) for i in range(65, 91)]  # A-Z

# Create a dictionary to map labels to characters
label_to_char = {i: char for i, char in enumerate(characters)}

# Print the first few samples
for i in range(5):
    plt.imshow(X[i], cmap='gray')
    plt.title(f"Label: {y[i]}, Character: {label_to_char[y[i]]}")
    plt.show()

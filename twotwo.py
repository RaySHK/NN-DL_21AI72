# Step 1: Import required libraries
import numpy as np
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
 
# Step 2: Define the AND, OR, XOR datasets
# Inputs
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
 
# AND Gate
y_and = np.array([0, 0, 0, 1])
 
# OR Gate
y_or = np.array([0, 1, 1, 1])
 
# XOR Gate
y_xor = np.array([0, 1, 1, 0])
 
# Function to train and plot the decision boundary
def train_and_plot(X, y, gate_name):
    # Step 3: Initialize the Perceptron model
    perceptron = Perceptron(max_iter=1000, tol=1e-3)
    
    # Step 4: Train the model
    perceptron.fit(X, y)
    
    # Step 5: Plot decision boundary
    plt.figure(figsize=(6, 6))
    
    # Create a mesh grid for plotting decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    
    Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.5, cmap='coolwarm')
    
    # Plot the data points
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', s=100, cmap='coolwarm')
    plt.title(f'Decision Boundary for {gate_name} Gate')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
 
# Step 6: Train and plot for AND, OR, XOR gates
train_and_plot(X, y_and, "AND")
train_and_plot(X, y_or, "OR")
train_and_plot(X, y_xor, "XOR")
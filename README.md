# Experiment 5: Cluster Analysis for Market Segmentation (K-Means Algorithm)

## Aim
To apply the K-Means Clustering algorithm to segment a dataset into distinct clusters and visualize how similar data points are grouped together for market segmentation.

## Dataset
The dataset consists of 16 manually entered customer data points representing two numerical features:
* **Feature 1 ($X$):** Annual Income (in Lakhs)
* **Feature 2 ($Y$):** Spending Score (`1–100`)

| Point | Annual Income ($X$) | Spending Score ($Y$) | Point | Annual Income ($X$) | Spending Score ($Y$) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 20 | 30 | **9** | 32 | 33 |
| **2** | 22 | 35 | **10** | 24 | 31 |
| **3** | 25 | 40 | **11** | 28 | 45 |
| **4** | 70 | 80 | **12** | 75 | 70 |
| **5** | 75 | 85 | **13** | 71 | 83 |
| **6** | 80 | 90 | **14** | 83 | 89 |
| **7** | 30 | 70 | **15** | 28 | 60 |
| **8** | 35 | 75 | **16** | 27 | 67 |

## Algorithm / Steps Followed
1. Import required Python libraries (`numpy`, `matplotlib`, `scikit-learn`).
2. Input data points as separate $X$ and $Y$ feature arrays.
3. Combine both variables into a single 2D feature matrix using `np.column_stack()`.
4. Initialize the `KMeans` model with $K = 3$ clusters.
5. Fit the model and assign each data point to a cluster using `fit_predict()`.
6. Retrieve cluster centroids (`cluster_centers_`) and assigned labels.
7. Visualize the clusters and their centroids using a Matplotlib scatter plot.
8. Interpret the resulting customer segments and centroid coordinates.

## Functions and Methods Used

| Category | Function / Method | Purpose / Description |
| :--- | :--- | :--- |
| **Data Handling** | `np.array()` | Creates arrays for storing numerical feature values ($X$ and $Y$) |
| | `np.column_stack()` | Combines multiple 1D arrays into a single 2D dataset for clustering |
| **Model Initialization** | `KMeans()` | Initializes the K-Means clustering model with $K$ clusters |
| **Model Training & Prediction** | `fit_predict()` | Fits the K-Means model to the dataset and assigns cluster labels |
| **Cluster Properties** | `cluster_centers_` | Retrieves the coordinates (centroids) of the cluster centers |
| | `labels_` | Stores the cluster labels assigned to each data point |
| **Visualization (Matplotlib)** | `plt.scatter()` | Plots the data points and centroids in a 2D scatter plot |
| | `plt.title()`, `plt.xlabel()`, `plt.ylabel()`, `plt.legend()`, `plt.grid()`, `plt.show()` | Configures labels, legend, gridlines, and displays the final graph |

## Program & Output

```python
# K-Means Clustering for Market Segmentation

# Step 1: Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Step 2: Input data points (Annual Income in Lakhs, Spending Score 1-100)
X = np.array([20, 22, 25, 70, 75, 80, 30, 35, 32, 24, 28, 75, 71, 83, 28, 27])
Y = np.array([30, 35, 40, 80, 85, 90, 70, 75, 33, 31, 45, 70, 83, 89, 60, 67])

# Step 3: Combine both variables into a single 2D feature matrix
data = np.column_stack((X, Y))
print("Feature matrix (Income, Spending Score):")
print(data)

# Step 4: Initialize the KMeans model with K = 3 clusters
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)

# Step 5: Fit the model and assign each point to a cluster
labels = kmeans.fit_predict(data)

# Step 6: Retrieve centroids and labels
centroids = kmeans.cluster_centers_
print("\nCluster labels for each point:")
for i, label in enumerate(labels, start=1):
    print(f"  Point {i:2d} ({X[i-1]:>2}, {Y[i-1]:>2}) -> Cluster {label}")

print("\nCluster centroids (Income, Spending Score):")
for i, c in enumerate(centroids):
    print(f"  Cluster {i}: ({c[0]:.2f}, {c[1]:.2f})  |  Points: {np.sum(labels == i)}")

print(f"\nInertia (within-cluster sum of squares): {kmeans.inertia_:.2f}")

# Step 7: Visualize clusters and centroids
colors = ["red", "green", "blue"]
plt.figure(figsize=(8, 6))
for i in range(3):
    plt.scatter(data[labels == i, 0], data[labels == i, 1],
                color=colors[i], s=70, label=f"Cluster {i}")
plt.scatter(centroids[:, 0], centroids[:, 1], color="black", marker="X",
            s=250, label="Centroids")
plt.title("K-Means Clustering: Customer Segmentation", fontsize=14, fontweight="bold")
plt.xlabel("Annual Income (Lakhs)")
plt.ylabel("Spending Score (1-100)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("images/kmeans_clusters.png", dpi=150)
plt.show()

Feature matrix (Income, Spending Score):
[[20 30]
 [22 35]
 [25 40]
 [70 80]
 [75 85]
 [80 90]
 [30 70]
 [35 75]
 [32 33]
 [24 31]
 [28 45]
 [75 70]
 [71 83]
 [83 89]
 [28 60]
 [27 67]]

Cluster labels for each point:
  Point  1 (20, 30) -> Cluster 1
  Point  2 (22, 35) -> Cluster 1
  Point  3 (25, 40) -> Cluster 1
  Point  4 (70, 80) -> Cluster 0
  Point  5 (75, 85) -> Cluster 0
  Point  6 (80, 90) -> Cluster 0
  Point  7 (30, 70) -> Cluster 2
  Point  8 (35, 75) -> Cluster 2
  Point  9 (32, 33) -> Cluster 1
  Point 10 (24, 31) -> Cluster 1
  Point 11 (28, 45) -> Cluster 1
  Point 12 (75, 70) -> Cluster 0
  Point 13 (71, 83) -> Cluster 0
  Point 14 (83, 89) -> Cluster 0
  Point 15 (28, 60) -> Cluster 2
  Point 16 (27, 67) -> Cluster 2

Cluster centroids (Income, Spending Score):
  Cluster 0: (75.67, 82.83)  |  Points: 6
  Cluster 1: (25.17, 35.67)  |  Points: 6
  Cluster 2: (30.00, 68.00)  |  Points: 4

Inertia (within-cluster sum of squares): 810.33

## Result
Thus, K-Means Clustering for market segmentation was executed successfully.

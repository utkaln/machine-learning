import numpy as np
import matplotlib.pyplot as plt

# K-Means basic steps 
# Step 1 : Initialize with random centroids
# Loop following two steps as long as Cost function reduces
# Step 2: Assign dataset items to closest centroid
# Step 3: Recompute centroid by finding average of assigned dataset items
# The converged solution may not provide globally optimal solution
# Hence this process is repeated in iterations to start with different initialization values

# Initialize with random centroids by randomizing the order of sample data and picking k of them
def initialize_random_centroids(dataset, K):
    # Create a random order
    rand_data = np.random.permutation(dataset.shape[0])
    
    # pick K count of points from dataset as centroids
    centroids = dataset[rand_data[:K]]
    print(f"Initial Random Centroids found => {centroids}")
    return centroids

# Assign dataset to closest centroid value
# X = input dataset, centroids = list of centroids
def closest_centroid(X, centroids):
    K = centroids.shape[0]
    datasize = X.shape[0]
    c_indices = np.zeros(X.shape[0], dtype=int)
    for i in range(datasize):
        distance = []
        for j in range(K):
            avg_dist = np.linalg.norm(X[i] - centroids[j])
            distance.append(avg_dist)
        c_indices[i] = np.argmin(distance)
    return c_indices


# Compute Centroid Means
def compute_centroids(X, c_indices, K):
    m,n = X.shape
    centroids = np.zeros((K,n))
    for i in range(K):
        data_points = X[c_indices == i]
        centroids[i] = np.mean(data_points, axis=0)
    return centroids


# Run K means method
def run_k_means(dataset, init_centroids, max_iter = 10):
    m,n = dataset.shape
    K = init_centroids.shape[0]
    centroids = init_centroids
    prev_centroids = centroids
    c_indices = np.zeros(m)

    for i in range(max_iter):
        print(f"Iteration {i+1} of {max_iter} ======>")
        c_indices = closest_centroid(dataset,centroids)
        centroids = compute_centroids(dataset, c_indices,K)
        print(f"centroids found => {centroids}")
    return centroids, c_indices
                    

# Execution with data
def load_data(filename):
    X = np.load(filename)
    return X

dataset = load_data("./data/kmeans-data.npy")
K = 10
max_iters = 10
initial_centroids = initialize_random_centroids(dataset,K)
centroids, c_indices = run_k_means(dataset, initial_centroids, max_iters)

print(f"FINAL centroids found => {centroids}\n c_indices found => {c_indices}")


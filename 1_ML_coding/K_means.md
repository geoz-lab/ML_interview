# K means

```python
import numpy as np

def K_means(X_input, K, max_iteration):
    np.random.seed(2)

    # X_input (n_samples, n_features)
    n_samples, n_features = len(X), len(X[0])

    # Setup random initial centroids
    random_idx = np.random.choice(n_samples, K, replace=False)
    centroids = X[random_idx] # (K, n_features)

    # update the centroids
    for _ in range(max_iteration):
        distances = np.zeros((n_samples, K))
        labels = np.zeros((n_samples))

        for i in range(n_samples):
            feature = X[i] # [x1, x2, ...]
            for j in range(K):
                centroid = centroids[j]
                distances[i][j] = np.linalg.norm(feature - centroid) # (n_samples, K)
            labels[i] = np.argmin(distances[i]) # (n_samples, 1)

        # new centroids
        for k in range(K):
            samples_in_cluster = X_input[labels==k] # (n_k, n_features)
 
        if len(samples_in_cluster) > 0:
            centroids[k] = samples_in_cluster.mean(axis=0)
        else:
            centroids[k] = X[np.random.choice(n_samples, 1)]

    return centroids, labels
```
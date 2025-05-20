# K means

```python
import numpy as np

def simple_kmeans(X, k=3, max_iters=10):
    # Step 1: Initialize centroids randomly
    n_samples = X.shape[0]
    centroids = X[np.random.choice(n_samples, k, replace=False)]

    for _ in range(max_iters):
        # Step 2: Assign clusters (nearest centroid)
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)  # shape: (n_samples, k)
        labels = np.argmin(distances, axis=1)

        # Step 3: Update centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        # Optional: print step
        print("Centroids:\n", new_centroids)

        centroids = new_centroids

    return centroids, labels
```
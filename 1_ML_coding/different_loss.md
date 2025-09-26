# Different Loss Functions

## Regression Loss
```python
def Mean_Squared_Error(y_hat, y):
    """
    range: [0, inf]
    y_hat: [n_batch, n_features]
    y: [n_batch, n_features]
    """
    mse_loss = np.mean((y_hat - y)**2.0)
    return mse_loss

def MAE_loss(y_hat, y):
    """
    Mean absolute error loss (L1 loss)
    range: [0, inf]
    y_hat: [n_batch, n_features]
    y: [n_batch, n_features]
    """
    return np.mean(np.abs(y_hat - y))

def RMSE_loss(y_hat, y):
    """
    Root mean squared error loss
    """
    return np.sqrt(np.mean((y_hat - y)**2.0))


def quantile_loss(y_hat, y, quantile = 0.5):
    """Quantile Loss (Pinball Loss)
    for quantile regression
    range: [0, inf]
    """
    residual = y - y_hat
    return np.mean(np.maximum(quantile * residual, (quantile - 1) * residual))


```

## BINARY CLASSIFICATION LOSS

```python
def binary_cross_entropy_loss(y_hat, y, epsilon=1e-15):
    """
    Binary classification
    y_hat [0, 1) and y {0, 1}
    likehood to log-likelihood: 
    P(y | ŷ) = ŷ^y × (1 - ŷ)^(1-y)
    L(θ) = ∏(i=1 to N) P(y_i | ŷ_i) = ∏(i=1 to N) ŷ_i^(y_i) × (1 - ŷ_i)^(1-y_i)
    """
    y_hat = np.clip(y_hat, epsilon, 1-epsilon) # clip to [epsilon, 1-epsilon]
    return -np.mean(y*np.log(y_hat) + (1-y)*np.log(1-y_hat))

def logistic_loss(y_hat, y):
    """
    use for binary classification
    range: [0, inf)
    """
    return np.mean(np.log(1+np.exp(-y*y_hat)))

```

## MULTI-CLASS CLASSIFICATION LOSS
```python
def categorical_cross_entropy_loss(y_hat, y, epsilon=1e-15):
    """
    Used for: Multi-class classification (one-hot encoded)
    Range: [0, ∞)
    """
    y_hat = np.clip(y_hat, epsilon, 1-epsilon)
    return -np.mean(np.sum(y*np.log(y_hat), axis = 1))

def sparse_categorical_cross_entropy_loss(y_hat, y, epsilon=1e-15):
    """
    Used for: Multi-class classification (integer labels)
    Range: [0, ∞)
    y_hat are probabilities, y are class indices
    # Predictions (probabilities for each class)
    for example: 
        y_hat = np.array([
            [0.1, 0.7, 0.1, 0.1],  # Sample 0: high prob for class 1 (dog)
            [0.8, 0.1, 0.05, 0.05], # Sample 1: high prob for class 0 (cat)  
            [0.2, 0.1, 0.6, 0.1]   # Sample 2: high prob for class 2 (bird)
        ])
    """
    y_hat = np.clip(y_hat, epsilon, 1-epsilon)

    return -np.mean(np.log(y_hat[np.arange(len(y)), y.astype(int)]))

def KL_divergence_loss(y_hat, y, epsilon=1e-15):
    """Kullback-Leibler Divergence Loss
    Used for: Probability distribution matching
    """
    y = np.clip(y, epsilon, 1-epsilon)
    y_hat = np.clip(y_hat, epsilon, 1-epsilon)
    return np.mean(np.sum(y * np.log(y/y_hat), axis = 1))

```

## ADVANCED/SPECIALIZED LOSS FUNCTIONS

---
def focal_loss(y_hat, y, alpha=1, gamma=2, epsilon=1e-15):
    """Focal Loss
    Used for: Class imbalanced problems
    Range: [0, ∞)
    Focuses learning on hard examples
    """
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
    ce = -y * np.log(y_hat)
    weight = alpha * ((1 - y_hat) ** gamma)
    return np.mean(weight * ce)

def dice_loss(y_hat, y, smooth=1):
    """Dice Loss
    Used for: Segmentation tasks
    Range: [0, 1]
    Based on Dice coefficient
    """
    intersection = np.sum(y_hat * y)
    union = np.sum(y_hat) + np.sum(y)
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice

def cosine_similarity_loss(y_hat, y):
    """Cosine Similarity Loss
    Used for: Similarity learning, embeddings
    Range: [0, 2]
    """
    dot_product = np.sum(y_hat * y, axis=1)
    norm_y_hat = np.linalg.norm(y_hat, axis=1)
    norm_y = np.linalg.norm(y, axis=1)
    cosine_sim = dot_product / (norm_y_hat * norm_y)
    return np.mean(1 - cosine_sim)

def contrastive_loss(y_hat, y, margin=1.0):
    """Contrastive Loss
    Used for: Siamese networks, learning embeddings
    Range: [0, ∞)
    Note: y should be 0 (similar) or 1 (dissimilar)
    """
    distance = np.sqrt(np.sum((y_hat[::2] - y_hat[1::2]) ** 2, axis=1))
    loss_similar = (1 - y) * distance ** 2
    loss_dissimilar = y * np.maximum(0, margin - distance) ** 2
    return np.mean(loss_similar + loss_dissimilar)

def triplet_loss(anchor, positive, negative, margin=1.0):
    """Triplet Loss
    Used for: Learning embeddings, face recognition
    Range: [0, ∞)
    """
    pos_dist = np.sum((anchor - positive) ** 2, axis=1)
    neg_dist = np.sum((anchor - negative) ** 2, axis=1)
    return np.mean(np.maximum(0, pos_dist - neg_dist + margin))

# =============================================================================
# ROBUST LOSS FUNCTIONS
# =============================================================================

def log_cosh_loss(y_hat, y):
    """Log-Cosh Loss
    Used for: Regression, robust to outliers
    Range: [0, ∞)
    Approximately MSE for small errors, MAE for large errors
    """
    def _log_cosh(x):
        return x + np.log(1 + np.exp(-2 * x)) - np.log(2)
    
    return np.mean(_log_cosh(y_hat - y))

def wing_loss(y_hat, y, omega=10, epsilon=2):
    """Wing Loss
    Used for: Facial landmark detection, robust regression
    Range: [0, ∞)
    """
    diff = np.abs(y_hat - y)
    condition = diff < omega
    linear = omega * np.log(1 + diff / epsilon)
    nonlinear = diff - omega + omega * np.log(1 + omega / epsilon)
    return np.mean(np.where(condition, linear, nonlinear))


# Simple CNN with Residual Connection

---

### 🧠 CNN with Residual Block — Markdown Math

---

#### **1. Notation**

$$
\begin{aligned}
X &\in \mathbb{R}^{m \times C \times H \times W} \quad \text{(Input images)} \\\\
W_1, W_2 &\quad \text{(Convolution filters)} \\\\
b_1, b_2 &\quad \text{(Bias terms)} \\\\
f(\cdot) &\quad \text{ReLU activation function} \\\\
\end{aligned}
$$

---

#### **2. Residual Block (Forward Pass)**

Input:

$$
Z^{[1]} = \text{Conv}(X, W_1) + b_1
$$

$$
A^{[1]} = f(Z^{[1]})
$$

Then:

$$
Z^{[2]} = \text{Conv}(A^{[1]}, W_2) + b_2
$$

$$
A^{[2]} = f(Z^{[2]} + X) \quad \text{(Residual connection)}
$$

---

#### **3. Loss Function**

For simplicity, mean squared error:

$$
\mathcal{L}(y, \hat{y}) = \frac{1}{m} \sum_{i=1}^{m} \| y^{(i)} - \hat{y}^{(i)} \|^2
$$

---

```python
import numpy as np

# --- Utility functions ---
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def conv2d(X, W, b, stride=1, padding=1):
    """
    Very simple 2D convolution with zero padding and stride.
    X: (C_in, H, W)
    W: (C_out, C_in, kH, kW)
    b: (C_out,)
    """
    C_out, C_in, kH, kW = W.shape
    _, H, W_in = X.shape

    # Pad input
    X_padded = np.pad(X, ((0,0), (padding,padding), (padding,padding)), mode='constant')
    H_out = (H + 2*padding - kH)//stride + 1
    W_out = (W_in + 2*padding - kW)//stride + 1

    out = np.zeros((C_out, H_out, W_out))
    for c in range(C_out):
        for i in range(0, H_out):
            for j in range(0, W_out):
                region = X_padded[:, i:i+kH, j:j+kW]
                out[c, i, j] = np.sum(region * W[c]) + b[c]
    return out

# --- Residual CNN Block ---
class ResidualBlock:
    def __init__(self, in_channels, out_channels, ksize=3):
        self.W1 = np.random.randn(out_channels, in_channels, ksize, ksize) * 0.01
        self.b1 = np.zeros(out_channels)
        self.W2 = np.random.randn(in_channels, out_channels, ksize, ksize) * 0.01
        self.b2 = np.zeros(in_channels)

    def forward(self, X):
        # First conv
        Z1 = conv2d(X, self.W1, self.b1)
        A1 = relu(Z1)

        # Second conv
        Z2 = conv2d(A1, self.W2, self.b2)

        # Residual connection
        out = relu(Z2 + X)
        return out

# --- Usage example ---
if __name__ == "__main__":
    X = np.random.randn(3, 8, 8)   # (channels, H, W)
    block = ResidualBlock(3, 3)
    Y = block.forward(X)
    print("Input shape:", X.shape, " Output shape:", Y.shape)
```

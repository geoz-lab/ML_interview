# Deep learning basics


---

**1. Describe the structure and principle of CNN**
CNNs use convolutional layers with filters to extract spatial features, followed by pooling and fully connected layers for classification.

**2. Compare the differences between RNN and LSTM**

* RNN: suffers from vanishing gradients, short memory.
* LSTM: uses gates (input, forget, output) to retain long-term dependencies.

**3. Explain the structure and principle of Transformer**
Uses self-attention and feedforward layers in parallelized encoder-decoder stacks; enables modeling long-range dependencies efficiently.

**4. The principle of Attention mechanism**
Computes weighted sums of values, where weights are based on the relevance (dot product) between queries and keys.

**5. The calculation process of Self-attention**

1. Compute Q, K, V from input
2. Attention = Softmax(QKᵀ / √d) × V
3. Outputs context-aware embeddings

**6. The role of Multi-head attention**
Allows the model to attend to information from different subspaces jointly, improving representational power.

**7. The role and implementation method of Positional encoding**
Adds sequence order info to input embeddings; implemented via sinusoidal or learned vectors.

**8. Encoder-decoder structure**
Encoder maps input to a context vector; decoder uses this to generate output sequentially (e.g., in translation).

**9. The principle and role of Dropout**
Randomly deactivates neurons during training to prevent overfitting and co-adaptation.

**10. Common weight initialization methods in neural networks**

* Xavier/Glorot: balances variance across layers (used with tanh)
* He: designed for ReLU activations
* Uniform/Normal: used with scaling factors

**11. Compare Adam and RMSprop**

* RMSprop: adjusts learning rate using moving average of squared gradients.
* Adam: combines RMSprop with momentum (first moment) for better convergence.

**12. What is the vanishing and exploding gradients problem? How to alleviate it?**
Occurs in deep/RNN networks during backpropagation. Solutions:

* Use ReLU or variants
* Normalize inputs (BatchNorm)
* Use LSTM/GRU for RNNs
* Gradient clipping

**13. Derive the mathematical formula of Backpropagation and describe the implementation steps**

1. Compute loss L
2. Compute gradients using chain rule: ∂L/∂w = ∂L/∂y × ∂y/∂x × ∂x/∂w
3. Propagate gradients backward layer by layer
4. Update weights: w ← w - η∂L/∂w

**14. Explain the principle and application scenarios of VAE**
VAE = probabilistic autoencoder that learns latent distributions.
Uses encoder to get μ, σ; decoder samples from latent space to reconstruct.
Used in image generation, anomaly detection, and representation learning.

**15. GAN vs VAE**
GAN:
- Generates very realistic and sharp images
- Good for high-resolution synthesis
- No need to define an explicit likelihood function
- Hard to train: mode collapse, unstable dynamics
- No interpretable latent space
- Sensitive to hyperparameters and architecture choices

VAE:
- Stable training (uses variational inference)
- Interpretable and smooth latent space
- Good for representation learning and interpolation
- Outputs are often blurry due to Gaussian assumptions
- Lower visual fidelity compared to GANs
- KL divergence term can dominate, hurting reconstruction

**16. What is Mel Spectrogram (MEL)**
A Mel spectrogram is a visual representation of audio, showing how the frequency content of a sound evolves over time, using a Mel scale that mimics human hearing.

It's essentially a 2D image where:
X-axis = Time
Y-axis = Frequency (Mel scale)
Color intensity = Energy (amplitude) at each frequency and time


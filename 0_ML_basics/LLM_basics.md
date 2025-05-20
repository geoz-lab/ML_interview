# LLM basics

---

**1. What are some ways to adapt LLMs to new tasks?**

* Fine-tuning (full or parameter-efficient like LoRA)
* Prompt engineering
* Instruction tuning
* In-context learning (few-shot or zero-shot)

**2. Explain the Transformer architecture, including encoder, decoder, QKV, and attention mechanisms.**

* **Encoder** processes input tokens into embeddings.
* **Decoder** generates outputs token by token.
* **QKV** (Query, Key, Value) are projections used in attention to compute token relevance.
* **Attention** scores each token's importance relative to others.


**3. Why do we choose decoder-only models instead of encoder-decoder models for certain tasks?**
Decoder-only models (e.g., GPT) are efficient for autoregressive generation. Encoder-decoder models are better for input-to-output mappings like translation.

**4. What is LoRA (Low-Rank Adaptation)?**
A fine-tuning technique that adds trainable low-rank matrices to frozen weight layers, enabling efficient training with fewer parameters.

**5. Differences: Continue training vs. LoRA**

* **Continue training**: updates all weights, requires more resources.
* **LoRA**: efficient, trains small parameter sets, suitable for customization.
* Trade-off: LoRA is lighter but may not reach the full potential of full fine-tuning.

**6. How to train an LLM to output a specific style?**

* Style-specific fine-tuning or LoRA
* Prompt tuning or prefix-tuning
* Few-shot examples demonstrating style

**7. Differences between BERT and vanilla Transformers?**
BERT uses only the **encoder** with masked language modeling; vanilla Transformers include both encoder and decoder.

**8. Differences in attention and position embeddings (OPT, LLaMA, etc.)**

* LLaMA: uses rotary positional embeddings (RoPE), efficient attention.
* OPT: uses learned absolute positional embeddings.
* Variants differ in performance and resource trade-offs.

**9. How to reduce LLM inference latency?**

* Quantization
* Model distillation
* KV cache reuse
* Efficient batch scheduling
* Model pruning or early exit

**10. Changes needed for online usage of LLMs**

* Add real-time APIs
* Use session caching
* Load balancing
* Ensure input sanitization and response time guarantees
* Guardrails for safety

**11. Tokenizer mechanism and types**
Tokenizers split text into model-readable units:

* **BPE (Byte Pair Encoding)**: merges frequent subwords.
* **WordPiece**: like BPE, used in BERT.
* **Unigram**: probabilistic approach.
* **SentencePiece**: language-agnostic, supports subword units.
* **Tiktoken**: optimized for GPT models.

**12. GPT vs. BERT vs. T5 and others**

* **GPT**: decoder-only, generative
* **BERT**: encoder-only, bidirectional
* **T5**: encoder-decoder, treats all tasks as text-to-text
* Others (e.g., LLaMA, PaLM): vary in architecture and efficiency

**13. Common tokenization methods and types**

* BPE, WordPiece, Unigram, SentencePiece
* Chosen based on model type and language requirements

**14. Common LLM fine-tuning methods**

* Full fine-tuning
* LoRA / QLoRA
* Prefix-tuning
* Adapter modules
* Instruction tuning

**15. Efficient training and inference techniques**

* Mixed-precision training (FP16/BF16)
* Gradient checkpointing
* Parameter-efficient fine-tuning
* Quantization (e.g., 4-bit)
* KV cache reuse

**16. Current limitations of LLMs**

* Hallucination
* Poor reasoning or math
* Bias and toxicity
* High compute and memory cost
* Lack of real-time world knowledge

**17. What is prompt engineering?**
Designing inputs (prompts) to guide LLM outputs effectively using examples, instructions, or formatting.

**18. Applications of LLMs in industry**

* Chatbots and customer support
* Code generation (e.g., Copilot)
* Document summarization
* Search and retrieval
* Data extraction and analysis
* Education and tutoring
* Healthcare (with supervision)


**19. Self-attention**
When encoding a specific token, it allows use the take a look at other tokens and weight the importance.
It can capture the contextual relationship between words regardless of the distance in the sequence.
$Q = XW^Q$, $K = XW^K$, $V = XW^V$
Compute Attention Scores:
$$
Scores(Q, K) = Q K^T
$$
Scale the scores:
$$
Scores = \frac{QK^T}{\sqrt{d_k}}
$$
Apply softmax to let each row sums to 1, giving weights to other tokens in the sequence.
$$
\text{Attention Weights} = \text{softmax} (\frac{QK^T}{\sqrt{d_k}})
$$
Compute Weighted Sum of Values:
$$
\text{Output} = \text{softmax} (\frac{QK^T}{\sqrt{d_k}}) V
$$

Instead of one self-attention, we use multiple heads:

![alt text](image.png)

**20. Why cross-attention in the decoder**
In sequence-to-sequence tasks (like translation), the decoder needs to:
(1) Understand what it's generated so far → via self-attention
(2) Access the encoded input sequence (e.g., the original sentence in French) → via cross-attention, to align and guide output generation (e.g., translate to English)

**21. When Transformers Struggle**
Small dataset; long sequences, quadratic time/memory complexity in attention; numerical and logical reasoning; Transformers can be data-hungry pattern matchers, not true reasoners; Sensitivity to Prompt/Tokenization

**22. How to improve transformers**
(1) Improve Accuracy and Generalization
Fine-Tuning and Instruction Tuning
Data Quality & Augmentation
Prompt Engineering
(2) Improve Efficiency
Model Compression
Optimized Architectures

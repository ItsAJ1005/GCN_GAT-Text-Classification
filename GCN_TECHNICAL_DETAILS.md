# GCN Technical Details - Normalization, Aggregation, Loss & Functions

## Overview

This document explains the technical components of the GCN (Graph Convolutional Network) implementation in this project.

---

## 1. 🔄 Normalization

### **A. Batch Normalization**

Used at multiple layers for training stability:

```python
# Input normalization
self.input_norm = nn.BatchNorm1d(hidden_dim)

# After each GCN layer
self.bn1 = nn.BatchNorm1d(hidden_dim)
self.bn2 = nn.BatchNorm1d(hidden_dim)
```

**Purpose:**
- Normalizes activations across the batch dimension
- Formula: `y = (x - μ) / √(σ² + ε) * γ + β`
  - μ: batch mean
  - σ²: batch variance
  - γ, β: learnable parameters
  - ε: small constant for numerical stability

**Benefits:**
- Faster convergence
- Reduces internal covariate shift
- Allows higher learning rates
- Acts as regularization

### **B. Graph Normalization (in GCNConv)**

The `GCNConv` layer from PyTorch Geometric implements symmetric normalization:

**Formula:**
```
H^(l+1) = σ(D̃^(-1/2) Ã D̃^(-1/2) H^(l) W^(l))
```

Where:
- **Ã = A + I**: Adjacency matrix with self-loops
- **D̃**: Degree matrix of Ã
- **D̃^(-1/2)**: Inverse square root of degree matrix
- **H^(l)**: Node features at layer l
- **W^(l)**: Learnable weight matrix
- **σ**: Activation function

**Purpose:**
- Normalizes message passing by node degree
- Prevents features from exploding/vanishing
- Ensures fair aggregation from neighbors

---

## 2. 📊 Aggregation Functions

### **A. Local Aggregation (GCN Layers)**

**GCNConv aggregation:**
```python
self.conv1 = GCNConv(hidden_dim, hidden_dim)
self.conv2 = GCNConv(hidden_dim, hidden_dim)
```

**Process:**
1. **Neighbor aggregation**: Each node aggregates features from its neighbors
2. **Weighted by edge**: Uses normalized adjacency matrix
3. **Linear transformation**: Applies learnable weights

**Mathematical form:**
```
h_i^(l+1) = σ(Σ_{j∈N(i)} (1/√(d_i * d_j)) * W^(l) * h_j^(l))
```

Where:
- `h_i`: Feature vector of node i
- `N(i)`: Neighbors of node i
- `d_i, d_j`: Degrees of nodes i and j
- `W^(l)`: Weight matrix at layer l

### **B. Global Aggregation (Pooling)**

**Global Mean Pooling:**
```python
x_global = global_mean_pool(x, batch)
```

**Purpose:**
- Converts variable-sized graphs → fixed-size vectors
- Aggregates all node features into document representation

**Formula:**
```
h_graph = (1/|V|) * Σ_{i∈V} h_i
```

Where:
- `|V|`: Number of nodes in graph
- `h_i`: Feature vector of node i
- `h_graph`: Graph-level representation

**Why Mean Pooling?**
- ✅ Permutation invariant (order doesn't matter)
- ✅ Works with variable graph sizes
- ✅ Simple and effective
- ✅ Captures average semantic meaning

### **C. Skip Connection Aggregation**

**Concatenation + Fusion:**
```python
x_combined = torch.cat([x_global, x_skip], dim=-1)
x = self.fusion(x_combined)
```

**Purpose:**
- Combines features from different network depths
- Preserves both low-level and high-level features
- Improves gradient flow

---

## 3. 📉 Loss Function

### **Cross-Entropy Loss**

```python
criterion = nn.CrossEntropyLoss()
loss = criterion(outputs, labels)
```

**Formula:**
```
L = -Σ_{i=1}^{N} Σ_{c=1}^{C} y_{i,c} * log(ŷ_{i,c})
```

Where:
- `N`: Number of samples
- `C`: Number of classes
- `y_{i,c}`: True label (1 if sample i is class c, else 0)
- `ŷ_{i,c}`: Predicted probability for class c

**Why Cross-Entropy?**
- ✅ Standard for multi-class classification
- ✅ Penalizes confident wrong predictions heavily
- ✅ Works well with softmax output
- ✅ Convex optimization landscape

**Implementation Detail:**
The model outputs `log_softmax`, so technically it's using **Negative Log-Likelihood Loss (NLLLoss)**:

```python
# In model forward()
return F.log_softmax(x, dim=1)

# Loss calculation
loss = criterion(log_probs, labels)
# Equivalent to: -log_probs[labels]
```

---

## 4. 🎯 Activation Functions

### **A. GELU (Gaussian Error Linear Unit)**

**Primary activation function:**
```python
conv1_out = F.gelu(conv1_out)
conv2_out = F.gelu(conv2_out)
```

**Formula:**
```
GELU(x) = x * Φ(x)
```
Where Φ(x) is the cumulative distribution function of standard normal distribution.

**Approximation:**
```
GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
```

**Why GELU?**
- ✅ Smoother than ReLU
- ✅ Non-monotonic (can output negative values)
- ✅ Better gradient flow
- ✅ State-of-the-art in transformers (BERT, GPT)
- ✅ Empirically better performance than ReLU

**Comparison:**
- **ReLU**: `max(0, x)` - hard cutoff at 0
- **GELU**: Smooth probabilistic gating
- **ELU**: `x if x > 0 else α(e^x - 1)` - smooth but different curve

### **B. Log Softmax (Output Layer)**

**Final activation:**
```python
return F.log_softmax(x, dim=1)
```

**Formula:**
```
log_softmax(x_i) = log(e^{x_i} / Σ_j e^{x_j})
                 = x_i - log(Σ_j e^{x_j})
```

**Why Log Softmax?**
- ✅ Numerically stable (avoids overflow)
- ✅ Works with NLLLoss
- ✅ Outputs log probabilities
- ✅ Faster computation than softmax + log

---

## 5. 🔧 Additional Components

### **A. Dropout Regularization**

```python
self.dropout = nn.Dropout(dropout=0.3)
```

**Purpose:**
- Randomly zeros 30% of activations during training
- Prevents overfitting
- Forces network to learn robust features

### **B. Skip Connections (Residual)**

```python
x = conv1_out + x  # Skip connection
```

**Purpose:**
- Helps gradient flow in deep networks
- Prevents vanishing gradients
- Allows training deeper models
- Inspired by ResNet

### **C. Xavier Initialization**

```python
nn.init.xavier_uniform_(m.weight)
```

**Formula:**
```
W ~ U(-√(6/(n_in + n_out)), √(6/(n_in + n_out)))
```

**Purpose:**
- Maintains variance of activations across layers
- Prevents exploding/vanishing gradients at initialization

---

## 6. 📐 Complete Forward Pass

**Step-by-step:**

```python
# 1. Input Projection + Normalization
x = self.input_proj(x)           # [num_nodes, vocab_size] → [num_nodes, 256]
x = self.input_norm(x)            # Batch normalize
input_features = x                # Save for skip connection

# 2. First GCN Layer
conv1_out = self.conv1(x, edge_index)  # Graph convolution
conv1_out = self.bn1(conv1_out)        # Batch normalize
conv1_out = F.gelu(conv1_out)          # GELU activation
conv1_out = self.dropout(conv1_out)    # Dropout
x = conv1_out + x                      # Skip connection

# 3. Second GCN Layer
conv2_out = self.conv2(x, edge_index)  # Graph convolution
conv2_out = self.bn2(conv2_out)        # Batch normalize
conv2_out = F.gelu(conv2_out)          # GELU activation
conv2_out = self.dropout(conv2_out)    # Dropout
x = conv2_out + x                      # Skip connection

# 4. Global Pooling
x_global = global_mean_pool(x, batch)           # [batch_size, 256]
x_skip = global_mean_pool(input_features, batch) # [batch_size, 256]

# 5. Feature Fusion
x_combined = torch.cat([x_global, x_skip], dim=-1)  # [batch_size, 512]
x = self.fusion(x_combined)                          # [batch_size, 256]

# 6. Classification
x = self.classifier(x)              # [batch_size, num_classes]
return F.log_softmax(x, dim=1)      # Log probabilities
```

---

## 7. 🎓 Key Takeaways

| Component | Type | Purpose |
|-----------|------|---------|
| **Normalization** | BatchNorm1d | Training stability, faster convergence |
| **Graph Norm** | Symmetric (D^-1/2 A D^-1/2) | Fair neighbor aggregation |
| **Aggregation** | Mean pooling | Graph → fixed vector |
| **Loss** | CrossEntropyLoss | Multi-class classification |
| **Activation** | GELU | Smooth non-linearity |
| **Output** | Log Softmax | Stable probability output |
| **Regularization** | Dropout (0.3) | Prevent overfitting |
| **Skip Connections** | Residual | Better gradient flow |

---

## 8. 📊 Comparison with Standard GCN

| Aspect | Standard GCN | This Implementation |
|--------|-------------|---------------------|
| Normalization | None | BatchNorm at each layer |
| Activation | ReLU | GELU |
| Skip Connections | No | Yes (residual) |
| Pooling | Mean | Mean + skip fusion |
| Initialization | Random | Xavier uniform |
| Regularization | Basic dropout | Dropout + BatchNorm |

**Result:** More stable training, better performance, deeper networks possible!

---

## 9. 💡 Why These Choices?

1. **GELU over ReLU**: Better empirical performance, smoother gradients
2. **BatchNorm**: Essential for deep networks, stabilizes training
3. **Skip Connections**: Allows deeper networks without vanishing gradients
4. **Mean Pooling**: Simple, effective, permutation invariant
5. **CrossEntropy Loss**: Standard for classification, well-understood
6. **Xavier Init**: Maintains gradient variance across layers

This implementation represents **modern best practices** for GNN-based text classification! 🚀

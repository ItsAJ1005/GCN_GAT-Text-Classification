# Model Optimization Guide

## Current Performance Baseline
- **20News GAT**: 70.41% accuracy, Loss: 1.1431

## 🚀 Recommended Optimizations

### 1. **Increase Model Capacity**

**Current Settings:**
```bash
--hidden_dim 128
--num_heads 4
--dropout 0.5
```

**Optimized Settings:**
```bash
# Larger model with more capacity
python main.py --dataset 20news --model gat \
  --hidden_dim 256 \
  --num_heads 8 \
  --dropout 0.3 \
  --num_epochs 50 \
  --learning_rate 0.0005 \
  --batch_size 64
```

**Expected Improvement:** +3-5% accuracy

---

### 2. **Adjust Learning Rate & Schedule**

**Current:** Fixed learning rate of 0.001

**Optimized:** Lower learning rate with more epochs
```bash
python main.py --dataset 20news --model gat \
  --learning_rate 0.0005 \
  --num_epochs 50
```

**Expected Improvement:** +2-3% accuracy, better convergence

---

### 3. **Increase Training Epochs**

**Current:** 20 epochs

**Optimized:** 50-100 epochs with early stopping
```bash
python main.py --dataset 20news --model gat \
  --num_epochs 100
```

**Note:** The model has early stopping (patience=10), so it will stop if validation loss doesn't improve.

**Expected Improvement:** +2-4% accuracy

---

### 4. **Optimize Graph Construction**

**Current Settings:**
```bash
--window_size 3
--max_nodes 100
```

**Optimized Settings:**
```bash
# Larger context window and more nodes
python main.py --dataset 20news --model gat \
  --window_size 5 \
  --max_nodes 150
```

**Expected Improvement:** +1-3% accuracy (better graph structure)

---

### 5. **Batch Size Optimization**

**Current:** 32

**Optimized:** 64 or 128 (if memory allows)
```bash
python main.py --dataset 20news --model gat \
  --batch_size 64
```

**Expected Improvement:** More stable gradients, faster training

---

## 🎯 Best Configuration (Recommended)

Combine all optimizations for maximum performance:

```bash
python main.py --dataset 20news --model gat \
  --hidden_dim 256 \
  --num_heads 8 \
  --dropout 0.3 \
  --num_epochs 100 \
  --learning_rate 0.0005 \
  --batch_size 64 \
  --window_size 5 \
  --max_nodes 150
```

**Expected Results:**
- Accuracy: 75-78% (up from 70.41%)
- Loss: 0.8-1.0 (down from 1.1431)
- Training time: ~2-3x longer

---

## 🔧 Advanced Optimizations

### 6. **Data Preprocessing**

Currently using all 20 categories. You can focus on specific categories for better performance:

Edit `data/loader.py` line 76:
```python
# Current (all 20 categories)
'categories': None,

# Optimized (4 categories - easier task)
'categories': ['sci.space', 'comp.graphics', 'rec.sport.baseball', 'talk.politics.mideast'],
```

**Expected Improvement:** +10-15% accuracy (but only 4 classes)

---

### 7. **Model Architecture Tweaks**

For GAT models, try different attention configurations:

```bash
# More attention heads
python main.py --dataset 20news --model gat \
  --num_heads 8 \
  --hidden_dim 256

# Fewer heads but larger hidden dim
python main.py --dataset 20news --model gat \
  --num_heads 4 \
  --hidden_dim 512
```

---

### 8. **Regularization Adjustments**

**Current Dropout:** 0.5 (aggressive)

**Try Different Values:**
```bash
# Less dropout for larger models
--dropout 0.3

# More dropout if overfitting
--dropout 0.6
```

---

## 📊 Quick Comparison Table

| Configuration | Hidden Dim | Heads | Dropout | LR | Epochs | Expected Acc | Training Time |
|--------------|-----------|-------|---------|-----|--------|--------------|---------------|
| **Current** | 128 | 4 | 0.5 | 0.001 | 20 | 70% | 1x |
| **Balanced** | 256 | 8 | 0.3 | 0.0005 | 50 | 75% | 2.5x |
| **Aggressive** | 512 | 8 | 0.2 | 0.0003 | 100 | 78% | 5x |
| **Fast** | 128 | 4 | 0.4 | 0.002 | 30 | 72% | 1.5x |

---

## 🎯 Step-by-Step Improvement Plan

### Phase 1: Quick Wins (30 min)
```bash
python main.py --dataset 20news --model gat \
  --hidden_dim 256 \
  --dropout 0.3 \
  --num_epochs 30 \
  --learning_rate 0.0005
```

### Phase 2: Better Architecture (1 hour)
```bash
python main.py --dataset 20news --model gat \
  --hidden_dim 256 \
  --num_heads 8 \
  --dropout 0.3 \
  --num_epochs 50 \
  --learning_rate 0.0005 \
  --batch_size 64
```

### Phase 3: Full Optimization (2-3 hours)
```bash
python main.py --dataset 20news --model gat \
  --hidden_dim 512 \
  --num_heads 8 \
  --dropout 0.2 \
  --num_epochs 100 \
  --learning_rate 0.0003 \
  --batch_size 64 \
  --window_size 5 \
  --max_nodes 150
```

---

## 🔍 Monitoring Improvements

After each training run, check:
1. **Test Accuracy**: Should increase
2. **Test Loss**: Should decrease
3. **Training vs Validation Gap**: Should be small (no overfitting)
4. **Per-class F1 scores**: Should improve for weak classes

---

## 💡 Tips

1. **Start with Phase 1** - Quick wins with minimal time investment
2. **Monitor overfitting** - If train acc >> val acc, increase dropout
3. **Use early stopping** - Already enabled with patience=10
4. **Compare models** - Train both GCN and GAT with same settings
5. **Save results** - All metrics are saved in `results/{dataset}/{model}/`

---

## 🎓 Understanding the Trade-offs

- **Larger models** = Better accuracy but slower training
- **More epochs** = Better convergence but longer time
- **Higher dropout** = Less overfitting but slower learning
- **Larger batch size** = More stable but needs more memory
- **Lower learning rate** = Better convergence but slower training

Choose based on your time and compute constraints!

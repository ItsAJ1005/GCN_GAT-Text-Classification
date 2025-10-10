# Quick Start Guide

## Training Models (Optimized Settings)

Train all four model-dataset combinations with optimized hyperparameters:

```bash
# MR Dataset
python main.py --dataset mr --model gcn --num_epochs 20
python main.py --dataset mr --model gat --num_epochs 20

# 20News Dataset
python main.py --dataset 20news --model gcn --num_epochs 20
python main.py --dataset 20news --model gat --num_epochs 20
```

**Note:** Default settings are now optimized:
- Hidden dim: 256 (was 128)
- Dropout: 0.3 (was 0.5)
- Learning rate: 2e-4 (was 1e-3)
- Batch size: 16 (was 32)
- Window size: 4 (was 3)
- Max nodes: 200 (was 100)
- Early stopping patience: 5 epochs

**Models will be saved to:**
- `models/mr/gcn/`
- `models/mr/gat/`
- `models/20news/gcn/`
- `models/20news/gat/`

## Making Predictions

### MR Dataset (Movie Reviews)

```bash
# Using GCN
python predict.py --dataset mr --model-type gcn --text "This movie was amazing!"

# Using GAT
python predict.py --dataset mr --model-type gat --text "This movie was amazing!"
```

### 20News Dataset

```bash
# Using GCN
python predict.py --dataset 20news --model-type gcn --text "The new graphics card has great performance"

# Using GAT
python predict.py --dataset 20news --model-type gat --text "The new graphics card has great performance"
```

## Directory Structure

```
models/
├── mr/
│   ├── gcn/          # MR + GCN model
│   └── gat/          # MR + GAT model
└── 20news/
    ├── gcn/          # 20News + GCN model
    └── gat/          # 20News + GAT model

results/
├── mr/
│   ├── gcn/          # MR + GCN results
│   └── gat/          # MR + GAT results
└── 20news/
    ├── gcn/          # 20News + GCN results
    └── gat/          # 20News + GAT results
```

## Common Options

### Training
- `--num_epochs`: Number of training epochs (default: 2)
- `--learning_rate`: Learning rate (default: 0.001)
- `--batch_size`: Batch size (default: 32)
- `--hidden_dim`: Hidden dimension size (default: 128)
- `--dropout`: Dropout rate (default: 0.5)
- `--model`: Model type (gcn or gat)

### Prediction
- `--text`: Text to classify (required)
- `--dataset`: Dataset name (mr or 20news)
- `--model-type`: Model type (gcn or gat)
- `--device`: Device to use (cpu or cuda)

## Examples

### Train with custom parameters
```bash
python main.py \
  --dataset mr \
  --model gat \
  --num_epochs 100 \
  --learning_rate 0.0005 \
  --batch_size 64 \
  --hidden_dim 256 \
  --dropout 0.3
```

### Batch predictions
```bash
# Create a script for multiple predictions
for text in "Great movie!" "Terrible film" "Amazing acting"; do
  python predict.py --dataset mr --model-type gat --text "$text"
done
```

## Troubleshooting

### Model not found
If you get "Model not found" error:
1. Check if the model exists: `ls models/{dataset}/{model_type}/`
2. Train the model first: `python main.py --dataset {dataset} --model {model_type}`

### Vocabulary not found
If vocabulary is missing:
1. Retrain the model (vocabulary is generated during training)
2. The new training script automatically saves vocabulary

### CUDA out of memory
If you run out of GPU memory:
1. Reduce batch size: `--batch_size 16`
2. Reduce hidden dimension: `--hidden_dim 64`
3. Use CPU: The script will automatically use CPU if CUDA is unavailable

## Next Steps

1. Train all four model combinations
2. Compare their performance
3. Use the best model for your task
4. Check `PROJECT_STRUCTURE.md` for detailed documentation

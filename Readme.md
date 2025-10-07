# Text Classification with GCN and GAT

## Team Members
- [Your Name] - Project Lead, Model Development
- [Team Member 2] - Data Processing & Visualization
- [Team Member 3] - Evaluation & Documentation

## Introduction

### What are Graph-of-Words?
Graph-of-Words is a document representation where words become nodes and their co-occurrence relationships become edges. This captures both local word order and global document structure, making it powerful for text analysis.

### Why GNN for Text Classification?
- **Captures non-sequential relationships**: Unlike RNNs/CNNs, GNNs can model arbitrary relationships between words
- **Handles variable-length documents**: Graph structures naturally handle documents of different lengths
- **Interpretability**: The graph structure provides insights into how words influence classification

This implementation is based on the paper: [Graph Convolutional Networks for Text Classification](https://arxiv.org/abs/1809.05679) (Yao et al., 2019)

## Features

- **Graph-based Text Representation**: Converts text documents into graph structures where words are nodes and relationships (co-occurrence) are edges.
- **Two GNN Architectures**:
  - **Graph Convolutional Network (GCN)**: Applies graph convolutions to aggregate information from neighboring nodes.
  - **Graph Attention Network (GAT)**: Uses attention mechanisms to weigh the importance of neighboring nodes.
- **Easy-to-Use Pipeline**: Simple command-line interface for training and evaluation.
- **Comprehensive Evaluation**: Includes accuracy, precision, recall, F1-score, and confusion matrix visualization.

## Project Structure

```
text-classification-gnn/
├── data/
│   ├── __init__.py
│   ├── loader.py        # Data loading and preprocessing
│   └── graph_builder.py # Text to graph conversion
├── models/
│   ├── __init__.py
│   ├── gcn.py           # GCN model implementation
│   └── gat.py           # GAT model implementation
├── train.py             # Training utilities
├── evaluate.py          # Evaluation and visualization
├── main.py              # Main script for training and evaluation
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/text-classification-gnn.git
   cd text-classification-gnn
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   python -m nltk.downloader stopwords
   ```

## Usage

### Training a Model

To train a model with default settings:

```bash
python main.py --data_path path/to/your/data.csv --model_type gcn --output_dir results/gcn
```

### Command Line Arguments

```
optional arguments:
  --data_path DATA_PATH  Path to the dataset file (CSV with text and label columns)
  --test_size TEST_SIZE  Proportion of data to use for testing (default: 0.2)
  --val_size VAL_SIZE    Proportion of training data to use for validation (default: 0.1)
  --random_state RANDOM_STATE
                        Random seed for reproducibility (default: 42)
  --model_type {gcn,gat}
                        Type of GNN model to use (default: gcn)
  --hidden_dim HIDDEN_DIM
                        Dimensionality of hidden layers (default: 128)
  --dropout DROPOUT     Dropout rate (default: 0.5)
  --num_heads NUM_HEADS
                        Number of attention heads (for GAT only) (default: 4)
  --batch_size BATCH_SIZE
                        Batch size for training (default: 32)
  --num_epochs NUM_EPOCHS
                        Number of training epochs (default: 50)
  --learning_rate LEARNING_RATE
                        Learning rate (default: 0.001)
  --max_nodes MAX_NODES
                        Maximum number of nodes per graph (default: 100)
  --window_size WINDOW_SIZE
                        Window size for graph construction (default: 3)
  --output_dir OUTPUT_DIR
                        Directory to save results and model (default: results)
  --model_save_path MODEL_SAVE_PATH
                        Path to save the best model (default: best_model.pt)
```

### Input Data Format

The input data should be a CSV file with at least two columns:
- `text`: The input text document
- `label`: The class label (can be string or integer)

Example:

```csv
"text","label"
"I loved this movie! It was amazing!","positive"
"Terrible experience, would not recommend.","negative"
```

### Output

The script will create the following files in the output directory:
- `best_model.pt`: The trained model with the best validation performance
- `training_history.json`: Training and validation metrics over epochs
- `training_history.png`: Plots of training/validation loss and accuracy
- `metrics.json`: Evaluation metrics on the test set
- `classification_report.json`: Detailed classification report
- `confusion_matrix.png`: Confusion matrix visualization
- `predictions.json`: Model predictions on the test set
- `misclassifications.txt`: Analysis of misclassified examples

## Quick Start

### Train and evaluate GCN on R8 dataset:
```bash
python main.py --model gcn --dataset R8
```

### Train and evaluate GAT on MR dataset:
```bash
python main.py --model gat --dataset MR
```

## Dataset Information

| Dataset | Description | #Classes | #Samples |
|---------|-------------|----------|-----------|
| R8 | 8 news categories | 8 | ~7,674 |
| MR | Movie reviews sentiment | 2 | ~10,662 |

*Table 1: Dataset statistics from Yao et al. (2019)*

## Architecture Overview

### Text to Graph Conversion
1. **Node Creation**: Each unique word becomes a node
2. **Edge Creation**: Words are connected if they co-occur within a sliding window
3. **Node Features**: TF-IDF weighted word vectors
4. **Edge Weights**: Pointwise Mutual Information (PMI) scores

### GCN Architecture
```
Input Layer (TF-IDF) → GCN Layer 1 (ReLU) → Dropout → 
GCN Layer 2 → Global Mean Pooling → Output Layer (Softmax)
```

### GAT Architecture
```
Input Layer (TF-IDF) → GAT Layer 1 (4 heads, ELU) → Dropout → 
GAT Layer 2 (1 head) → Global Mean Pooling → Output Layer (Softmax)
```

## Results
*To be updated after experiments*

## References
1. Yao, L., Mao, C., & Luo, Y. (2019). Graph Convolutional Networks for Text Classification. In *AAAI*.
2. Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. In *ICLR*.
3. Veličković, P., et al. (2018). Graph Attention Networks. In *ICLR*.

## Dependencies

- Python 3.7+
- PyTorch 2.0.0+
- PyTorch Geometric 2.3.0+
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- NLTK
- tqdm

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

1. [Semi-Supervised Classification with Graph Convolutional Networks (Kipf & Welling, 2017)](https://arxiv.org/abs/1609.02907)
2. [Graph Attention Networks (Veličković et al., 2018)](https://arxiv.org/abs/1710.10903)
3. [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

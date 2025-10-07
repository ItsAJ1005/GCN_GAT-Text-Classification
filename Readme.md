# Text Classification with Graph Neural Networks (GNNs)

This project implements text classification using Graph Neural Networks (GNNs), specifically Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT). The project is designed to be simple, well-documented, and easy to understand for beginners in GNNs and text processing.

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

## Example

Here's an example of how to train and evaluate a GAT model on a sample dataset:

```bash
# Train a GAT model
python main.py \
  --data_path data/sample_data.csv \
  --model_type gat \
  --hidden_dim 128 \
  --num_heads 4 \
  --batch_size 32 \
  --num_epochs 20 \
  --output_dir results/gat_experiment
```

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

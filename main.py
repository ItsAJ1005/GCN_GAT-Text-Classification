import os
import argparse
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data.loader import TextDataset
from data.graph_builder import GraphBuilder
from models.gcn import DocumentGCN as GCN
from models.gat import DocumentGAT as GAT
from train import train_model, GraphDataset, collate_fn
from evaluate import evaluate_model, compare_models
from utils import plot_training_history
from torch.utils.data import DataLoader

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Text Classification with GNNs')
    
    # Data arguments
    parser.add_argument('--dataset', type=str, default='20news',
                        choices=['20news', 'mr'],
                        help='Dataset to use (20news or mr)')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to the dataset file (CSV with text and label columns). If not provided, the built-in dataset will be used.')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data to use for testing')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='Proportion of training data to use for validation')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Model arguments
    parser.add_argument('--model', '--model_type', type=str, dest='model_type', 
                        choices=['gcn', 'gat'], default='gcn',
                        help='Type of GNN model to use (gcn or gat)')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Dimensionality of hidden layers')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads (for GAT only)')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=2,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--max_nodes', type=int, default=100,
                        help='Maximum number of nodes per graph')
    parser.add_argument('--window_size', type=int, default=3,
                        help='Window size for graph construction')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results and model')
    parser.add_argument('--model_save_path', type=str, default='best_model.pt',
                        help='Path to save the best model')
    
    return parser.parse_args()

def setup_environment():
    """Setup environment including device and random seeds."""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    return device

def prepare_data(args):
    """Load and prepare the dataset."""
    print("\nLoading and preparing data...")
    
    # Load the dataset
    dataset = TextDataset(
        dataset_name=args.dataset,
        data_dir='data',
        test_size=args.test_size,
        random_state=args.random_state,
        preprocess=False  # Skip preprocessing for now to avoid NLTK issues
    )
    
    # Get dataset splits
    train_df, test_df = dataset.get_splits(return_dataframe=True)
    
    # Further split train into train and validation
    train_df, val_df = train_test_split(
        train_df, 
        test_size=args.val_size,
        random_state=args.random_state,
        stratify=train_df['label_id']
    )
    
    # Extract texts and labels
    X_train = train_df['text'].values
    X_val = val_df['text'].values
    X_test = test_df['text'].values
    
    y_train = train_df['label_id'].values
    y_val = val_df['label_id'].values
    y_test = test_df['label_id'].values
    
    # Initialize graph builder
    graph_builder = GraphBuilder(
        min_word_freq=5,  # Minimum word frequency
        window_size=args.window_size,  # Context window size
        max_vocab_size=10000  # Maximum vocabulary size
    )
    
    # Simple text cleaning before building vocabulary
    def simple_clean(text):
        if not isinstance(text, str):
            text = str(text)
        return ' '.join(text.lower().split())
    
    # Apply simple cleaning
    X_train = [simple_clean(text) for text in X_train]
    X_val = [simple_clean(text) for text in X_val]
    X_test = [simple_clean(text) for text in X_test]
    
    # Build vocabulary on training data
    graph_builder.build_vocab(X_train)
    
    # Create datasets
    train_dataset = GraphDataset(X_train, y_train, graph_builder)
    val_dataset = GraphDataset(X_val, y_val, graph_builder)
    test_dataset = GraphDataset(X_test, y_test, graph_builder)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    return dataset, graph_builder, train_loader, val_loader, test_loader

def create_model(args, vocab_size, num_classes, device):
    """Create the GNN model."""
    if args.model_type == 'gcn':
        print("\nCreating GCN model...")
        model = GCN(
            vocab_size=vocab_size,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            dropout=args.dropout
        )
    elif args.model_type == 'gat':
        print("\nCreating GAT model...")
        model = GAT(
            input_dim=vocab_size,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            num_heads=args.num_heads,
            dropout=args.dropout
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # Move model to the specified device
    return model.to(device)

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Setup environment
    device = setup_environment()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    args.model_save_path = os.path.join(args.output_dir, 'best_model.pt')
    
    # Prepare data
    dataset, graph_builder, train_loader, val_loader, test_loader = prepare_data(args)
    
    # Get input dimension and number of classes
    input_dim = len(graph_builder.vocab)  # Using vocabulary size as input dimension
    num_classes = dataset.get_num_classes()
    # Get the reverse mapping from label IDs to label names
    label_map = dataset.get_label_map()
    id_to_label = {v: k for k, v in label_map.items()}
    class_names = [id_to_label[i] for i in range(num_classes)]
    
    print(f"\nDataset Info:")
    print(f"  Number of training examples: {len(train_loader.dataset)}")
    print(f"  Number of validation examples: {len(val_loader.dataset)}")
    print(f"  Number of test examples: {len(test_loader.dataset)}")
    print(f"  Vocabulary size: {input_dim}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Classes: {', '.join(class_names)}")
    
    # Create model
    model = create_model(args, input_dim, num_classes, device)
    print(f"\nModel architecture:")
    print(model)
    
    # Train the model
    print("\nStarting training...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=args.num_epochs,
        lr=args.learning_rate,
        device=device,
        model_save_path=args.model_save_path
    )
    
    # Save training history
    history_path = os.path.join(args.output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        import json
        # Convert numpy arrays and other non-serializable types to Python native types
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(x) for x in obj]
            else:
                return obj
                
        serializable_history = convert_to_serializable(history)
        json.dump(serializable_history, f, indent=4)
    
    # Plot training history
    plot_training_history(history, save_dir=args.output_dir)
    
    # Evaluate the model on test set
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(args.model_save_path))
    
    # Create test dataset and loader with batch_size=1 for individual predictions
    test_dataset = GraphDataset(
        test_loader.dataset.texts,
        test_loader.dataset.labels,
        graph_builder
    )
    test_loader_indiv = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Evaluate and save results
    evaluate_model(
        model=model,
        test_loader=test_loader_indiv,
        criterion=torch.nn.CrossEntropyLoss(),
        device=device,
        class_names=class_names,
        output_dir=args.output_dir
    )
    
    # Analyze misclassifications
    predictions_path = os.path.join(args.output_dir, 'predictions.json')
    if os.path.exists(predictions_path):
        analyze_misclassifications(predictions_path, output_dir=args.output_dir)
    
    print("\nDone!")

if __name__ == "__main__":
    main()

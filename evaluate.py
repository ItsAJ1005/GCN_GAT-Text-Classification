import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, f1_score, precision_score, recall_score
)
from torch_geometric.data import DataLoader
from torch.utils.data import random_split
from models.gcn import DocumentGCN
from models.gat import DocumentGAT
from data.loader import TextDataset
from data.graph_builder import GraphBuilder
from tqdm import tqdm
import argparse

# Set style for plots
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

def load_model(model_path, model_type, vocab_size, num_classes, device='cuda'):
    """Load a trained GNN model."""
    if model_type.lower() == 'gcn':
        model = DocumentGCN(vocab_size=vocab_size, hidden_dim=64, num_classes=num_classes)
    elif model_type.lower() == 'gat':
        model = DocumentGAT(vocab_size=vocab_size, hidden_dim=64, num_classes=num_classes, num_heads=4)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    return model


def plot_confusion_matrix(y_true, y_pred, class_names, title='Confusion Matrix', save_path=None):
    """
    Plot and save a normalized confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        title: Title for the plot
        save_path: Path to save the plot
    """
    # Convert class names to indices if they're strings
    if isinstance(y_true[0], str):
        label_encoder = LabelEncoder()
        y_true = label_encoder.fit_transform(y_true)
        y_pred = label_encoder.transform(y_pred)
        class_names = label_encode.classes_
    
    # Filter out any classes that don't appear in y_true
    present_classes = np.unique(y_true)
    class_names = [class_names[i] for i in range(len(class_names)) if i in present_classes]
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    
    # Normalize the confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_metrics_comparison(metrics_dict, save_path):
    """Plot comparison of metrics between models."""
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    models = list(metrics_dict.keys())
    
    # Prepare data
    data = {m: [metrics_dict[model][m] for model in models] for m in metrics}
    df = pd.DataFrame(data, index=models)
    
    # Plot
    plt.figure(figsize=(12, 6))
    ax = df.plot(kind='bar', rot=0, width=0.8, figsize=(12, 6))
    plt.title('Model Comparison', fontsize=14)
    plt.ylabel('Score', fontsize=12)
    plt.xticks(rotation=45)
    plt.ylim(0, 1.1)
    
    # Add value labels
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}", 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center', 
                   xytext=(0, 10), 
                   textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_sample_graph(graph, pred_label, true_label, class_names, save_path):
    """Visualize a sample graph with prediction."""
    import networkx as nx
    
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add nodes with their features
    for i, feat in enumerate(graph.x):
        G.add_node(i, weight=float(torch.sum(feat)))
    
    # Add edges
    edge_index = graph.edge_index.cpu().numpy()
    for i in range(edge_index.shape[1]):
        G.add_edge(edge_index[0, i], edge_index[1, i])
    
    # Draw graph
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    
    # Get node sizes based on feature weights
    node_weights = [G.nodes[n]['weight'] * 300 + 100 for n in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_size=node_weights, 
                          node_color='lightblue', alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    
    # Add title with prediction
    title = f"Predicted: {class_names[pred_label]}\nTrue: {class_names[true_label]}"
    if pred_label == true_label:
        plt.title(title, color='green')
    else:
        plt.title(title, color='red')
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_classification_report(y_true, y_pred, class_names, save_path):
    """Save classification report to a file with both micro and macro averages."""
    # Generate reports
    report = classification_report(
        y_true, y_pred, 
        target_names=class_names,
        output_dict=True
    )
    
    # Add micro and macro averages
    micro_avg = {
        'precision': precision_score(y_true, y_pred, average='micro'),
        'recall': recall_score(y_true, y_pred, average='micro'),
        'f1-score': f1_score(y_true, y_pred, average='micro'),
        'support': len(y_true)
    }
    
    macro_avg = {
        'precision': precision_score(y_true, y_pred, average='macro'),
        'recall': recall_score(y_true, y_pred, average='macro'),
        'f1-score': f1_score(y_true, y_pred, average='macro'),
        'support': len(y_true)
    }
    
    report['micro avg'] = micro_avg
    report['macro avg'] = macro_avg
    
    # Save as JSON
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    # Print summary
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    return report


def evaluate_model(model, test_loader, criterion, device, class_names, output_dir='results'):
    """
    Evaluate the model and save comprehensive results.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to run evaluation on
        class_names: List of class names
        output_dir: Directory to save results
    
    Returns:
        dict: Dictionary containing all metrics and predictions
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_texts = []
    all_graphs = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            if batch is None or len(batch) != 3:
                continue
                
            data, labels, texts = batch
            
            # Skip this batch if data or labels are None (can happen if all graphs in batch are empty)
            if data is None or labels is None:
                continue
                
            data = data.to(device)
            labels = labels.to(device)
            
            # Forward pass - pass the entire data object
            outputs = model(data)
            loss = criterion(outputs, labels)
            
            # Get predictions
            _, preds = torch.max(outputs, 1)
            
            # Store results
            batch_size = len(labels)
            total_loss += loss.item() * batch_size
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_texts.extend(texts)
            
            # Store graphs for visualization
            if hasattr(data, 'num_graphs') and hasattr(data, 'get_example'):
                for i in range(data.num_graphs):
                    try:
                        graph = data.get_example(i)
                        all_graphs.append({
                            'x': graph.x,
                            'edge_index': graph.edge_index,
                            'pred': preds[i].item() if i < len(preds) else -1,
                            'true': labels[i].item() if i < len(labels) else -1
                        })
                    except Exception as e:
                        print(f"Error processing graph {i}: {e}")
    
    # Calculate metrics
    avg_loss = total_loss / len(test_loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # Print results
    print("\n" + "="*60)
    print(f"Evaluation Results")
    print("="*60)
    print(f"Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("="*60 + "\n")
    
    # Save classification report
    report_path = os.path.join(output_dir, 'classification_report.json')
    report = save_classification_report(all_labels, all_preds, class_names, report_path)
    
    # Save confusion matrix
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(all_labels, all_preds, class_names, 
                         f'Confusion Matrix - {os.path.basename(output_dir)}', cm_path)
    
    # Save sample graph visualizations (first correct and incorrect prediction)
    os.makedirs(os.path.join(output_dir, 'sample_graphs'), exist_ok=True)
    correct_found = False
    incorrect_found = False
    
    for i, graph_data in enumerate(all_graphs):
        if not correct_found and graph_data['pred'] == graph_data['true']:
            plot_sample_graph(
                graph_data, 
                graph_data['pred'], 
                graph_data['true'],
                class_names,
                os.path.join(output_dir, 'sample_graphs', f'correct_{i}.png')
            )
            correct_found = True
        elif not incorrect_found and graph_data['pred'] != graph_data['true']:
            plot_sample_graph(
                graph_data,
                graph_data['pred'],
                graph_data['true'],
                class_names,
                os.path.join(output_dir, 'sample_graphs', f'incorrect_{i}.png')
            )
            incorrect_found = True
        
        if correct_found and incorrect_found:
            break
    
    # Compile metrics
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'report': report,
        'predictions': [{
            'text': text,
            'predicted_class': int(pred),
            'true_class': int(label),
            'correct': int(pred) == int(label)
        } for text, pred, label in zip(all_texts, all_preds, all_labels)]
    }
    
    # Save metrics
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return metrics


def compare_models(results_dir='results'):
    """Compare performance of GCN and GAT models."""
    model_dirs = [d for d in os.listdir(results_dir) 
                 if os.path.isdir(os.path.join(results_dir, d)) 
                 and any(m in d.lower() for m in ['gcn', 'gat'])]
    
    if not model_dirs:
        print("No model results found for comparison.")
        return
    
    comparison = {}
    
    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir)
        metrics_path = os.path.join(results_dir, model_dir, 'metrics.json')
        
        if not os.path.exists(metrics_path):
            print(f"Metrics not found for {model_name}")
            continue
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        comparison[model_name] = {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1']
        }
    
    # Create comparison table
    if not comparison:
        print("No valid model results to compare.")
        return
    
    # Save comparison to file
    comparison_path = os.path.join(results_dir, 'model_comparison.txt')
    with open(comparison_path, 'w') as f:
        # Write markdown table
        f.write("# Model Comparison\n\n")
        f.write("| Metric    | " + " | ".join(comparison.keys()) + " |\n")
        f.write("|-----------|" + "|-" * len(comparison) + "|\n")
        
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            f.write(f"| {metric.capitalize():<9} | ")
            for model_metrics in comparison.values():
                f.write(f"{model_metrics[metric]:.4f} | ")
            f.write("\n")
    
    # Plot comparison
    plot_metrics_comparison(
        {k: v for k, v in comparison.items()},
        os.path.join(results_dir, 'model_comparison.png')
    )
    
    print(f"\nModel comparison saved to: {os.path.abspath(comparison_path)}")
    
    # Print to console
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print(pd.DataFrame(comparison).transpose().round(4))
    print("="*60)
    
    return comparison

def prepare_data_loaders(dataset_name='r8', batch_size=32, test_size=0.2, val_size=0.1):
    """Prepare data loaders for training, validation, and testing."""
    # Load dataset
    dataset = TextDataset(dataset_name=dataset_name)
    
    # Split into train/val/test
    train_val_size = 1 - test_size
    train_size = 1 - (val_size / train_val_size)
    
    train_val, test = random_split(
        dataset, 
        [int(len(dataset) * train_val_size), 
         len(dataset) - int(len(dataset) * train_val_size)]
    )
    
    train, val = random_split(
        train_val,
        [int(len(train_val) * train_size), 
         len(train_val) - int(len(train_val) * train_size)]
    )
    
    # Initialize graph builder
    graph_builder = GraphBuilder()
    
    # Create datasets
    train_dataset = GraphDataset(
        [dataset.texts[i] for i in train.indices],
        [dataset.labels[i] for i in train.indices],
        graph_builder
    )
    
    val_dataset = GraphDataset(
        [dataset.texts[i] for i in val.indices],
        [dataset.labels[i] for i in val.indices],
        graph_builder
    )
    
    test_dataset = GraphDataset(
        [dataset.texts[i] for i in test.indices],
        [dataset.labels[i] for i in test.indices],
        graph_builder
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    return train_loader, val_loader, test_loader, len(dataset.vocab), len(dataset.label_encoder.classes_)

def main():
    parser = argparse.ArgumentParser(description='Evaluate GNN models for text classification')
    parser.add_argument('--dataset', type=str, default='20news', choices=['r8', 'mr', '20news'],
                      help='Dataset to evaluate on (r8, mr, or 20news)')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to run evaluation on (cuda or cpu)')
    parser.add_argument('--model_dir', type=str, default='saved_models',
                      help='Directory containing saved models')
    parser.add_argument('--results_dir', type=str, default='results',
                      help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Load dataset and prepare data loaders
    print(f"\nLoading {args.dataset.upper()} dataset...")
    dataset = TextDataset(dataset_name=args.dataset)
    _, _, test_loader, vocab_size, num_classes = prepare_data_loaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size
    )
    
    class_names = dataset.label_encoder.classes_
    
    # Evaluate each model
    os.makedirs(args.results_dir, exist_ok=True)
    
    for model_type in ['gcn', 'gat']:
        model_path = os.path.join(args.model_dir, f'{model_type}_best_model.pt')
        if not os.path.exists(model_path):
            print(f"\nModel not found: {model_path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Evaluating {model_type.upper()} model")
        print(f"{'='*60}")
        
        # Set up output directory
        model_output_dir = os.path.join(args.results_dir, f'{model_type}_results')
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Load model
        model = load_model(
            model_path=model_path,
            model_type=model_type,
            vocab_size=vocab_size,
            num_classes=num_classes,
            device=args.device
        )
        
        # Evaluate
        criterion = torch.nn.CrossEntropyLoss()
        evaluate_model(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=args.device,
            class_names=class_names,
            output_dir=model_output_dir
        )
    
    # Compare models
    compare_models(args.results_dir)
    
    print("\nEvaluation complete!")
    print(f"Results saved to: {os.path.abspath(args.results_dir)}")

if __name__ == "__main__":
    main()

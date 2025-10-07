import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

class GraphDataset(Dataset):
    """A simple dataset class for graph data."""
    def __init__(self, texts, labels, graph_builder):
        self.texts = texts
        self.labels = labels
        self.graph_builder = graph_builder
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        node_features, edge_index = self.graph_builder.text_to_graph(text)
        return node_features, edge_index, label, text


def collate_fn(batch):
    """Collate function for creating batches of graphs."""
    node_features_list, edge_indices, labels, texts = zip(*batch)
    
    # Create a batch of graphs
    data_list = []
    for i, (feat, edge_idx, label, text) in enumerate(zip(node_features_list, 
                                                         edge_indices, 
                                                         labels, 
                                                         texts)):
        if edge_idx.size(1) == 0:  # Skip empty graphs
            continue
        data = {
            'x': feat,
            'edge_index': edge_idx,
            'y': torch.tensor([label], dtype=torch.long),
            'text': text
        }
        data_list.append(data)
    
    # Create a batch from the list of graphs
    batch_data = Batch.from_data_list([
        {'x': d['x'], 'edge_index': d['edge_index'], 'y': d['y']} 
        for d in data_list
    ])
    
    # Extract texts and labels for the non-empty graphs
    texts = [d['text'] for d in data_list]
    labels = [d['y'].item() for d in data_list]
    
    return batch_data, torch.tensor(labels, dtype=torch.long), texts


def train_epoch(model, loader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(loader, desc="Training"):
        data, labels, _ = batch
        data = data.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(data.x, data.edge_index, data.batch)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, accuracy, f1


def evaluate(model, loader, criterion, device):
    """Evaluate the model on the given dataset."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_texts = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            data, labels, texts = batch
            data = data.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(data.x, data.edge_index, data.batch)
            loss = criterion(outputs, labels)
            
            # Track metrics
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_texts.extend(texts)
    
    # Calculate metrics
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, accuracy, f1, all_texts, all_preds, all_labels


def train_model(model, train_loader, val_loader, test_loader, 
               num_epochs=10, lr=0.001, device='cuda', 
               model_save_path='best_model.pt'):
    """
    Train and evaluate the model.
    
    Args:
        model: The GNN model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        test_loader: DataLoader for test data.
        num_epochs: Number of training epochs.
        lr: Learning rate.
        device: Device to train on ('cuda' or 'cpu').
        model_save_path: Path to save the best model.
    
    Returns:
        dict: Training history with loss and metrics.
    """
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    # Initialize training history
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [],
        'test_loss': None, 'test_acc': None, 'test_f1': None
    }
    
    best_val_f1 = 0.0
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 60)
        
        # Train for one epoch
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Evaluate on validation set
        val_loss, val_acc, val_f1, _, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        
        # Update learning rate
        scheduler.step(val_f1)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Save the best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved new best model with Val F1: {val_f1:.4f}")
    
    # Load the best model and evaluate on test set
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(model_save_path))
    test_loss, test_acc, test_f1, test_texts, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )
    
    print(f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}")
    
    # Save test results
    history['test_loss'] = test_loss
    history['test_acc'] = test_acc
    history['test_f1'] = test_f1
    history['test_results'] = {
        'texts': test_texts,
        'preds': test_preds,
        'labels': test_labels
    }
    
    return history

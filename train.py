import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split
from torch_geometric.data import Batch, Data
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Import models
from models.gcn import DocumentGCN
from models.gat import DocumentGAT
from data.loader import TextDataset, TextPreprocessor
from data.graph_builder import GraphBuilder

class GraphDataset(Dataset):
    def __init__(self, texts, labels, graph_builder):
        self.texts = texts
        self.labels = labels
        self.graph_builder = graph_builder
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        node_features, edge_index, edge_weights = self.graph_builder.text_to_graph(text)
        return node_features, edge_index, edge_weights, label, text


def collate_fn(batch):
    """Collate function for creating batches of graphs with edge weights."""
    node_features_list, edge_indices, edge_weights_list, labels, texts = zip(*batch)
    
    # Create a list to store PyG Data objects
    data_list = []
    labels_list = []
    texts_list = []
    
    for i, (feat, edge_idx, edge_weights, label, text) in enumerate(zip(
        node_features_list, 
        edge_indices, 
        edge_weights_list,
        labels, 
        texts
    )):
        if edge_idx.size(1) == 0:  # Skip empty graphs
            continue
            
        # Ensure edge indices are within bounds
        num_nodes = feat.size(0)
        edge_idx = edge_idx.clone()
        edge_idx[edge_idx >= num_nodes] = num_nodes - 1  # Clamp to last node index
        
        # Ensure we have the same number of edges and edge weights
        if edge_weights.size(0) != edge_idx.size(1):
            edge_weights = torch.ones(edge_idx.size(1), device=edge_weights.device)
        
        # Create a PyG Data object
        data = Data(
            x=feat,
            edge_index=edge_idx,
            edge_attr=edge_weights.unsqueeze(1),  # Add edge weights as edge features
            y=torch.tensor([label], dtype=torch.long)
        )
        data_list.append(data)
        labels_list.append(label)
        texts_list.append(text)
    
    if not data_list:  # If all graphs were empty
        return None, None, None
    
    try:
        # Create a batch from the list of Data objects
        batch_data = Batch.from_data_list(data_list)
    except Exception as e:
        print(f"Error creating batch: {e}")
        return None, None, None
    
    return batch_data, torch.tensor(labels_list, dtype=torch.long), texts_list


def train_epoch(model, loader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    num_batches = 0
    
    for batch in tqdm(loader, desc="Training"):
        data, labels, _ = batch
        
        # Skip empty batches
        if data is None or labels is None:
            continue
            
        data = data.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass - pass the entire data object to the model
        outputs = model(data)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        num_batches += 1
    
    # Calculate metrics
    if num_batches == 0:
        return 0.0, 0.0, 0.0
        
    avg_loss = total_loss / num_batches
    accuracy = accuracy_score(all_labels, all_preds) if all_preds else 0.0
    f1 = f1_score(all_labels, all_preds, average='weighted') if all_preds else 0.0
    
    return avg_loss, accuracy, f1


def evaluate(model, loader, criterion, device):
    """Evaluate the model on the given dataset."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_texts = []
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            data, labels, texts = batch
            
            # Skip empty batches
            if data is None or labels is None:
                continue
                
            data = data.to(device)
            labels = labels.to(device)
            
            # Forward pass - pass the entire data object to the model
            outputs = model(data)
            loss = criterion(outputs, labels)
            
            # Track metrics
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_texts.extend(texts)
            num_batches += 1
    
    # Calculate metrics
    if num_batches == 0:
        return 0.0, 0.0, 0.0, [], [], []
        
    avg_loss = total_loss / num_batches
    accuracy = accuracy_score(all_labels, all_preds) if all_preds else 0.0
    f1 = f1_score(all_labels, all_preds, average='weighted') if all_preds else 0.0
    
    return avg_loss, accuracy, f1, all_texts, all_preds, all_labels


def plot_training_curves(history, model_name, output_dir='results'):
    """Plot training and validation metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot loss
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title(f'{model_name} - Training & Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name.lower()}_training_curves.png'))
    plt.close()

def train_model(model, train_loader, val_loader, test_loader, 
               num_epochs=50, lr=0.0002, device='cuda', 
               model_save_path='best_model.pt', patience=5,
               max_grad_norm=1.0):
    """
    Enhanced training with learning rate scheduling, gradient clipping, and improved early stopping.
    
    Args:
        model: The GNN model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        test_loader: DataLoader for test data.
        num_epochs: Maximum number of training epochs.
        lr: Initial learning rate.
        device: Device to train on ('cuda' or 'cpu').
        model_save_path: Path to save the best model.
        patience: Number of epochs to wait before early stopping.
        max_grad_norm: Maximum gradient norm for clipping.
    
    Returns:
        dict: Training history and metrics
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Learning rate scheduler with warmup
    warmup_epochs = 5
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=warmup_epochs/num_epochs,  # Warmup phase
        anneal_strategy='cos',  # Cosine annealing
        div_factor=25.0,  # Initial lr = max_lr/25
        final_div_factor=1000.0  # Final lr = initial_lr/1000
    )
    
    best_loss = float('inf')
    best_val_f1 = 0.0
    best_val_acc = 0.0
    patience_counter = 0
    best_model = None
    best_epoch = 0
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_f1': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'learning_rates': []
    }
    
    # Initialize EMA for model averaging
    ema_alpha = 0.999
    ema_model = type(model)(*model.__init_args__).to(device)
    ema_model.load_state_dict(model.state_dict())
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        total_samples = 0
        all_preds = []
        all_labels = []
        
        # Training loop with gradient clipping and learning rate scheduling
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch in progress_bar:
            if batch is None or len(batch) != 3:
                continue
                
            data, labels, _ = batch
            if data is None or labels is None:
                continue
                
            data = data.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            scheduler.step()  # Update learning rate
            
            # Update EMA model
            with torch.no_grad():
                for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                    ema_param.data.mul_(ema_alpha).add_(param.data, alpha=1 - ema_alpha)
            
            # Track metrics
            batch_size = labels.size(0)
            total_train_loss += loss.item() * batch_size
            total_samples += batch_size
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{current_lr:.6f}'
            })
        
        # Calculate epoch metrics
        train_loss = total_train_loss / total_samples
        train_acc = accuracy_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        # Validation phase using EMA model
        ema_model.eval()
        val_loss, val_acc, val_f1, _, _, _ = evaluate(
            ema_model, val_loader, criterion, device
        )
        
        # Track history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['learning_rates'].append(current_lr)
        
        # Print epoch results
        print(f"\nEpoch {epoch+1}/{num_epochs} Results:")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Model saving with comprehensive criteria
        improved = False
        save_message = []
        
        if val_loss < best_loss:
            best_loss = val_loss
            save_message.append(f"New best loss: {val_loss:.4f}")
            improved = True
            
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_message.append(f"New best accuracy: {val_acc:.4f}")
            improved = True
            
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_message.append(f"New best F1: {val_f1:.4f}")
            improved = True
        
        if improved:
            best_model = ema_model.state_dict().copy()  # Save EMA model
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_f1': val_f1
            }, model_save_path)
            print(f"Model saved to {model_save_path} (" + ", ".join(save_message) + ")")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
        
        # Early stopping with more informative message
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            print(f"Best model from epoch {best_epoch+1}:")
            print(f"  Loss: {best_loss:.4f}")
            print(f"  Accuracy: {best_val_acc:.4f}")
            print(f"  F1 Score: {best_val_f1:.4f}")
            break
    
    model.load_state_dict(torch.load(model_save_path))
    
    test_loss, test_acc, test_f1, test_texts, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )
    
    test_metrics = {
        'loss': test_loss,
        'accuracy': test_acc,
        'f1': test_f1,
        'classification_report': classification_report(test_labels, test_preds, output_dict=True)
    }
    
    history['test_loss'] = test_loss
    history['test_acc'] = test_acc
    history['test_f1'] = test_f1
    
    print(f"\nTest Results - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}")
    
    # Add test results to history
    history['test_results'] = {
        'texts': test_texts,
        'preds': test_preds,
        'labels': test_labels
    }
    
    # Add test metrics to history for easier access
    history['test_metrics'] = test_metrics
    return history

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Union
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, global_add_pool

"""models/gat.py

Improved Graph Attention Network (GAT) for document classification.

This file keeps the original 2-layer GAT design but adds several small
quality-of-life improvements:
- Type hints and clearer docstrings
- Input validation for dimensions and heads
- Configurable activation and readout (mean/max/sum)
- Optional embedding output (for retrieval / analysis)
- BatchNorm after GAT layers for training stability
- Residual connection when dimensions match
- Safety handling when `data.batch` is missing (single graph)

The goal is to keep backward compatibility: the default behavior returns
log-probabilities exactly like the original implementation.
"""


def _get_activation(name: Union[str, Callable]) -> Callable:
    """Return an activation function given a name or callable.

    Accepts either a string ('elu', 'relu', 'tanh', 'identity') or a
    callable. Defaults to ELU for compatibility with the previous model.
    """
    if callable(name):
        return name
    name = (name or 'elu').lower()
    if name == 'elu':
        return F.elu
    if name == 'relu':
        return F.relu
    if name == 'tanh':
        return F.tanh
    if name in ('none', 'identity'):
        return lambda x: x
    raise ValueError(f"Unsupported activation: {name}")


class DocumentGAT(nn.Module):
    """Two-layer GAT for document classification with small API improvements.

    Args:
        vocab_size: input node feature dimension (vocabulary or embedding dim)
        num_classes: number of target classes
        hidden_dim: hidden dimension for node representations (default: 64)
        num_heads: number of attention heads in first layer (default: 4)
        dropout: dropout probability applied after attention (default: 0.5)
        activation: activation to use (str or callable). Default 'elu'.
        readout: pooling method for node->graph: 'mean' (default), 'max', or 'sum'
        return_embedding: if True, forward returns (logits, embedding) where
            embedding is the pooled document vector before classifier.
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.5,
        activation: Union[str, Callable] = 'elu',
        readout: str = 'mean',
        return_embedding: bool = False,
    ) -> None:
        super().__init__()

        # Basic validation
        if vocab_size <= 0:
            raise ValueError('vocab_size must be > 0')
        if num_classes <= 0:
            raise ValueError('num_classes must be > 0')
        if hidden_dim <= 0:
            raise ValueError('hidden_dim must be > 0')
        if num_heads <= 0:
            raise ValueError('num_heads must be > 0')
        if hidden_dim % num_heads != 0:
            # make an explicit requirement to avoid subtle shape bugs
            raise ValueError('hidden_dim must be divisible by num_heads')

        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = float(dropout)
        self.activation = _get_activation(activation)
        self.readout = readout.lower()
        self.return_embedding = bool(return_embedding)

        # First GAT layer: multi-head. We set out_channels per head so that
        # heads * out_channels == hidden_dim when concat=True.
        self.conv1 = GATConv(
            in_channels=vocab_size,
            out_channels=hidden_dim // num_heads,
            heads=num_heads,
            dropout=self.dropout,
            concat=True,
        )

        # Follow conv1 with BatchNorm for stability (applied to concatenated output)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        # Second GAT layer: single head that maps hidden_dim -> hidden_dim
        self.conv2 = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=1,
            dropout=self.dropout,
            concat=False,
        )

        # Optional BatchNorm after second layer
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # Classifier head
        self.classifier = nn.Linear(hidden_dim, num_classes)

        # Dropout module
        self._drop = nn.Dropout(self.dropout)

        # Initialize parameters
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize linear layers with Xavier uniform and zero biases."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Apply the configured readout (pooling) to node features."""
        if self.readout == 'mean':
            return global_mean_pool(x, batch)
        if self.readout == 'max':
            return global_max_pool(x, batch)
        if self.readout in ('sum', 'add'):
            return global_add_pool(x, batch)
        raise ValueError(f'Unsupported readout: {self.readout}')

    def forward(self, data) -> Union[torch.Tensor, tuple]:
        """Forward pass.

        Accepts a PyG Data object with attributes `x`, `edge_index`, and
        optionally `batch`. If `batch` is missing, we assume a single graph
        (all nodes belong to batch 0).

        Returns:
            - If return_embedding is False (default): log-probabilities [B, C]
            - If return_embedding is True: (log-probs [B, C], embedding [B, H])
        """
        x = data.x
        edge_index = data.edge_index
        batch = getattr(data, 'batch', None)

        if batch is None:
            # single graph -> all nodes belong to batch 0
            device = x.device if isinstance(x, torch.Tensor) else torch.device('cpu')
            batch = torch.zeros(x.size(0), dtype=torch.long, device=device)

        # Layer 1
        x1 = self.conv1(x, edge_index)
        x1 = self.activation(x1)
        # BatchNorm expects shape [N, C]
        x1 = self.bn1(x1)
        x1 = self._drop(x1)

        # Layer 2
        x2 = self.conv2(x1, edge_index)
        x2 = self.activation(x2)
        x2 = self.bn2(x2)

        # Residual connection when dims match
        if x2.size(-1) == x1.size(-1):
            x_out = x2 + x1
        else:
            x_out = x2

        # Pool to get graph/document embedding
        emb = self._pool(x_out, batch)

        # Classify
        logits = self.classifier(emb)
        log_probs = F.log_softmax(logits, dim=1)

        if self.return_embedding:
            return log_probs, emb
        return log_probs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

"""
Graph Attention Network (GAT) for Document Classification

Key Concepts:
1. Attention Mechanism: Learns to assign different importance to different words
   based on their semantic relationships, regardless of distance.
2. Multi-head Attention: Uses multiple attention heads to capture different types
   of relationships between words (e.g., syntactic, semantic).
3. The model excels at capturing long-range dependencies and non-consecutive word
   relationships that are common in natural language.
"""

class DocumentGAT(nn.Module):
    """
    2-layer Graph Attention Network for document classification.
    
    Input: Batch of document graphs
    - Node features: TF-IDF vectors [num_nodes, vocab_size]
    - Edge indices: Word co-occurrence connections [2, num_edges]
    - Batch vector: Maps nodes to their respective documents
    
    Output: Class probabilities [batch_size, num_classes]
    
    Key Advantages:
    1. Captures long-range dependencies through attention mechanism
    2. Handles non-consecutive word relationships effectively
    3. Learns to focus on important words regardless of their position
    """
    
    def __init__(self, vocab_size, num_classes, hidden_dim=64, num_heads=4, dropout=0.5):
        """
        Initialize the DocumentGAT model.
        
        Args:
            vocab_size (int): Size of the vocabulary (input dimension)
            num_classes (int): Number of output classes
            hidden_dim (int): Hidden dimension size (default: 64)
            num_heads (int): Number of attention heads in first layer (default: 4)
            dropout (float): Dropout rate (default: 0.5)
        """
        super(DocumentGAT, self).__init__()
        
        # First GAT layer with multi-head attention
        # Each head learns different types of word relationships
        self.conv1 = GATConv(
            in_channels=vocab_size,
            out_channels=hidden_dim // num_heads,  # Split hidden_dim across heads
            heads=num_heads,
            dropout=dropout,
            concat=True  # Concatenate head outputs
        )
        
        # Second GAT layer with single attention head
        # Combines information from all previous heads
        self.conv2 = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=1,  # Single head for final representation
            dropout=dropout,
            concat=False
        )
        
        # Classifier head
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # Regularization
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with xavier uniform and zeros for biases."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            # GATConv layers handle their own initialization
    
    def forward(self, data):
        """
        Forward pass of the DocumentGAT model.
        
        Args:
            data: PyG Data object containing:
                - x: Node features [num_nodes, vocab_size]
                - edge_index: Graph connectivity [2, num_edges]
                - batch: Batch vector [num_nodes]
                
        Returns:
            torch.Tensor: Class logits [batch_size, num_classes]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # First GAT layer: Multi-head attention
        # Each head learns different types of word relationships
        x = F.elu(self.conv1(x, edge_index))  # [num_nodes, hidden_dim]
        x = self.dropout(x)
        
        # Second GAT layer: Combine information from all heads
        # Uses a single attention head to create a unified representation
        x = F.elu(self.conv2(x, edge_index))  # [num_nodes, hidden_dim]
        
        # Global mean pooling: Aggregate node features into document representation
        # The attention mechanism ensures important words contribute more to the final representation
        x = global_mean_pool(x, batch)  # [batch_size, hidden_dim]
        
        # Final classifier
        x = self.classifier(x)
        
        return F.log_softmax(x, dim=1)  # Return log probabilities for NLLLoss

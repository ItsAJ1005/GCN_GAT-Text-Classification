import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

"""
Enhanced 3-layer GCN for Document Classification with Residual Connections

Key Improvements:
1. Added third GCN layer for deeper feature extraction
2. Improved residual connections between layers
3. Better weight initialization with Kaiming Normal
4. Optimized hidden dimension and dropout for MR dataset
5. Enhanced feature fusion and classification head
"""

class DocumentGCN(nn.Module):
    """
    Enhanced 3-layer GCN with residual connections for document classification.
    
    Input: Batch of document graphs
    - Node features: TF-IDF vectors [num_nodes, vocab_size]
    - Edge indices: Word co-occurrence connections [2, num_edges]
    - Batch vector: Maps nodes to their respective documents
    
    Architecture:
    1. Input projection with GELU activation
    2. Three GCN layers with residual connections
    3. Global mean pooling
    4. Enhanced classifier head with dropout
    
    Output: Class logits [batch_size, num_classes]
    """
    def __init__(self, vocab_size, num_classes, hidden_dim=128, dropout=0.4):
        """
        Initialize the enhanced DocumentGCN model.
        
        Args:
            vocab_size (int): Size of the vocabulary (input dimension)
            num_classes (int): Number of output classes
            hidden_dim (int): Hidden dimension size (default: 128)
            dropout (float): Dropout rate (default: 0.4)
        """
        super().__init__()
        
        # Input projection with activation
        self.input_proj = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        
        # Three GCN layers with residual connections
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        # Enhanced classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Regularization
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Kaiming Normal and zeros for biases."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Using 'leaky_relu' as it's widely supported and works well with GELU
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, data):
        """
        Forward pass of the enhanced DocumentGCN model.
        
        Args:
            data: PyG Data object containing:
                - x: Node features [num_nodes, vocab_size]
                - edge_index: Graph connectivity [2, num_edges]
                - batch: Batch vector [num_nodes]
                
        Returns:
            torch.Tensor: Class logits [batch_size, num_classes]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Input projection
        h = self.input_proj(x)
        
        # First GCN layer with residual connection
        h1 = self.conv1(h, edge_index)
        h1 = self.bn1(h1)
        h1 = F.gelu(h1)
        h1 = self.dropout(h1)
        
        # Second GCN layer with residual connection
        h2 = self.conv2(h + h1, edge_index)  # Residual connection
        h2 = self.bn2(h2)
        h2 = F.gelu(h2)
        h2 = self.dropout(h2)
        
        # Third GCN layer with residual connection
        h3 = self.conv3(h1 + h2, edge_index)  # Residual connection
        h3 = self.bn3(h3)
        h3 = F.gelu(h3)
        
        # Global pooling
        h_pool = global_mean_pool(h3, batch)
        
        # Classifier
        logits = self.classifier(h_pool)
        
        return F.log_softmax(logits, dim=1)  # Return log probabilities for NLLLoss

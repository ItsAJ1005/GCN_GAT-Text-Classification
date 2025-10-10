import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

"""
Simple 2-layer GCN for Document Classification

Key Concepts:
1. Graph Convolution: Aggregates information from neighboring nodes to learn
   meaningful node representations that capture local graph structure.
2. Global Pooling: Converts variable-sized graphs into fixed-size document
   representations by aggregating node features.
3. The model learns both local word relationships (via GCN) and global document
   semantics (via pooling) for classification.
"""

class DocumentGCN(nn.Module):
    """
    Enhanced 2-layer GCN for document classification.
    
    Input: Batch of document graphs
    - Node features: TF-IDF vectors [num_nodes, vocab_size]
    - Edge indices: Word co-occurrence connections [2, num_edges]
    - Batch vector: Maps nodes to their respective documents
    
    Improvements:
    1. Batch normalization for training stability
    2. Skip connections for better gradient flow
    3. Improved activation (GELU) for better non-linearity
    4. Enhanced feature processing
    
    Output: Class probabilities [batch_size, num_classes]
    """
    def __init__(self, vocab_size, num_classes, hidden_dim=256, dropout=0.3):
        """
        Initialize the enhanced DocumentGCN model.
        
        Args:
            vocab_size (int): Size of the vocabulary (input dimension)
            num_classes (int): Number of output classes
            hidden_dim (int): Hidden dimension size (default: 256)
            dropout (float): Dropout rate (default: 0.3)
        """
        super(DocumentGCN, self).__init__()
        
        # Input projection and normalization
        self.input_proj = nn.Linear(vocab_size, hidden_dim)
        self.input_norm = nn.BatchNorm1d(hidden_dim)
        
        # First GCN layer with improved architecture
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        # Second GCN layer with increased capacity
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Feature fusion for skip connections
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
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
        """Initialize weights with xavier uniform and zeros for biases."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, data):
        """
        Enhanced forward pass of the DocumentGCN model.
        
        Args:
            data: PyG Data object containing:
                - x: Node features [num_nodes, vocab_size]
                - edge_index: Graph connectivity [2, num_edges]
                - batch: Batch vector [num_nodes]
                
        Returns:
            torch.Tensor: Class logits [batch_size, num_classes]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Initial feature projection and normalization
        x = self.input_proj(x)
        x = self.input_norm(x)
        input_features = x  # Save for skip connection
        
        # First GCN layer with batch normalization and skip connection
        conv1_out = self.conv1(x, edge_index)
        conv1_out = self.bn1(conv1_out)
        conv1_out = F.gelu(conv1_out)  # GELU activation
        conv1_out = self.dropout(conv1_out)
        x = conv1_out + x  # Skip connection
        
        # Second GCN layer with batch normalization and skip connection
        conv2_out = self.conv2(x, edge_index)
        conv2_out = self.bn2(conv2_out)
        conv2_out = F.gelu(conv2_out)  # GELU activation
        conv2_out = self.dropout(conv2_out)
        x = conv2_out + x  # Skip connection
        
        # Global pooling with skip connection from input
        x_global = global_mean_pool(x, batch)  # Current features
        x_skip = global_mean_pool(input_features, batch)  # Skip connection from input
        
        # Combine features from different levels
        x_combined = torch.cat([x_global, x_skip], dim=-1)
        x = self.fusion(x_combined)
        
        # Enhanced classification head
        x = self.classifier(x)
        
        return F.log_softmax(x, dim=1)  # Return log probabilities for NLLLoss

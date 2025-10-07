import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_max_pool

class GAT(nn.Module):
    """
    Graph Attention Network for text classification.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, 
                 num_heads=4, dropout=0.5):
        """
        Initialize the GAT model.
        
        Args:
            input_dim (int): Dimensionality of input features.
            hidden_dim (int): Dimensionality of hidden layers.
            output_dim (int): Number of output classes.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
        super(GAT, self).__init__()
        
        # First GAT layer with multiple attention heads
        self.conv1 = GATConv(input_dim, hidden_dim, 
                            heads=num_heads, 
                            dropout=dropout)
        
        # Second GAT layer with single attention head
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, 
                            heads=1, 
                            dropout=dropout)
        
        # Classifier
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with xavier uniform and zeros for biases."""
        for m in self.modules():
            if isinstance(m, (GATConv, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass of the GAT model.
        
        Args:
            x (torch.Tensor): Node feature matrix [num_nodes, input_dim].
            edge_index (torch.Tensor): Graph connectivity in COO format [2, num_edges].
            batch (torch.Tensor, optional): Batch vector [num_nodes] which maps each node to its graph.
                                          If None, assumes all nodes belong to the same graph.
        
        Returns:
            torch.Tensor: Output logits [batch_size, output_dim].
        """
        # First GAT layer with multiple attention heads
        x = F.elu(self.conv1(x, edge_index))
        x = self.dropout(x)
        
        # Second GAT layer with single attention head
        x = F.elu(self.conv2(x, edge_index))
        
        # Global max pooling to get graph-level representation
        if batch is not None:
            x = global_max_pool(x, batch)  # [batch_size, hidden_dim]
        else:
            x = x.max(dim=0, keepdim=True)[0]  # [1, hidden_dim] if single graph
        
        # Final classifier
        x = self.fc(x)
        
        return x

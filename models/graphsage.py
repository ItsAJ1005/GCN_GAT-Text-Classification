import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool

class GraphSAGE(nn.Module):
    """
    GraphSAGE model for document classification.
    Implements the Graph Sample and Aggregate architecture with mean aggregation.
    
    Input: 
    - Node features: [num_nodes, in_channels]
    - Edge indices: [2, num_edges]
    - Batch indices: [num_nodes]
    
    Output: Class probabilities [batch_size, num_classes]
    
    Args:
        in_channels (int): Size of input node features (vocab_size for bag-of-words)
        hidden_channels (int): Size of hidden layers
        out_channels (int): Number of output classes
        num_layers (int): Number of GraphSAGE layers
        dropout (float): Dropout rate
        aggr (str): Aggregation method ('mean', 'max', 'lstm', 'pool')
    """
    
    def __init__(self, in_channels, hidden_channels=256, out_channels=2, 
                 num_layers=2, dropout=0.3, aggr='mean'):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        
        # Input layer
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
            
        # Output layer
        self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )
        
        # Batch normalization
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)
        ])
        
    def forward(self, x, edge_index, batch=None):
        # Node feature transformation
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        # Final layer
        x = self.convs[-1](x, edge_index)
        
        # Graph-level prediction
        if batch is not None:
            x = global_mean_pool(x, batch)
            
        # Classifier
        x = self.classifier(x)
        return F.log_softmax(x, dim=-1)
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.batch_norms:
            bn.reset_parameters()
        for layer in self.classifier:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

class GraphSAGEWithFeatures(GraphSAGE):
    """
    Extended GraphSAGE that can handle additional edge features.
    """
    def __init__(self, in_channels, edge_dim=None, **kwargs):
        super().__init__(in_channels, **kwargs)
        if edge_dim is not None:
            self.edge_encoder = nn.Linear(edge_dim, in_channels)
        else:
            self.edge_encoder = None
            
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if edge_attr is not None and self.edge_encoder is not None:
            edge_attr = self.edge_encoder(edge_attr)
            
        return super().forward(x, edge_index, batch)
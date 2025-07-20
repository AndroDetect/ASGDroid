import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import HeteroConv, GCNConv, global_mean_pool, global_max_pool
from torch_geometric.data import HeteroData
from typing import Tuple

class HGNN(nn.Module):
    """
    Heterogeneous Graph Neural Network (HGNN)
    For malware detection with attention mechanism
    """
    def __init__(self, 
                 input_dim: int,
                 permission_dim: int = 0,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 num_classes: int = 2,
                 pooling: str = 'mean',
                 use_permission: bool = True,
                 use_attention: bool = True,
                 use_residual: bool = True):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of GCN layers
            num_classes: Number of classification classes (2: malware/benign)
            dropout: Dropout ratio
            pooling: Graph-level pooling method ('mean', 'max', 'concat')
            use_attention: Whether to use attention mechanism
            use_residual: Whether to use residual connections
        """
        super(HGNN, self).__init__()
        
        self.pooling = pooling
        self.use_permission = use_permission
        self.permission_dim = permission_dim
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.hidden_dim = hidden_dim
        
        # Input projection layer
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Heterogeneous graph convolution layers
        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for i in range(num_layers):
            conv_dict = {}
            # Create GCN layers for each edge type
            conv_dict[('api', 'call_graph', 'api')] = GCNConv(hidden_dim, hidden_dim)
            conv_dict[('api', 'taint', 'api')] = GCNConv(hidden_dim, hidden_dim)
            
            # Use sum aggregation to avoid dimension doubling, or adjust LayerNorm dimension
            hetero_conv = HeteroConv(conv_dict, aggr='sum')  # Changed to sum aggregation
            self.convs.append(hetero_conv)
            
            # LayerNorm dimension matches the aggregated dimension
            # sum aggregation: dimension remains hidden_dim
            # cat aggregation: dimension becomes hidden_dim * num_edge_types
            self.layer_norms.append(nn.LayerNorm(hidden_dim))  # Use hidden_dim for sum aggregation

        # Attention mechanism
        if self.use_attention:
            self.attention_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                #nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )
        
        # Classification head
        graph_pooling_dim = hidden_dim
        if pooling == 'concat':
            graph_pooling_dim = hidden_dim * 2  # mean + max pooling
        
        # Permission input features
        if self.use_permission:
            final_feature_dim = graph_pooling_dim + self.permission_dim
        else:
            final_feature_dim = graph_pooling_dim

        self.classifier = nn.Sequential(
            nn.Linear(final_feature_dim, final_feature_dim // 2),
            nn.ReLU(),
            #nn.Dropout(dropout),
            nn.Linear(final_feature_dim // 2, final_feature_dim // 4),
            nn.ReLU(),
            #nn.Dropout(dropout),
            nn.Linear(final_feature_dim // 4, num_classes),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        '''
        self.modules(): Get all sub-modules in the current model, including: nn.Linear, nn.ReLU, nn.Dropout, nn.Sequential, etc.
        for m in: Iterate through each sub-module
        m: Represents the current module object being processed
        '''
        for m in self.modules():
            if isinstance(m, nn.Linear): # Skip activation functions, Dropout, etc. that don't need initialization
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def get_graph_embedding_with_attention(self, data: HeteroData) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get graph embedding representation and attention weights
        
        Args:
            data: Heterogeneous graph data batch
            
        Returns:
            graph_features: Graph embedding features
            attention_weights: Attention weights [batch_size, num_nodes]
        """
        # Get node features
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        
        # Get batch information - Fix KeyError issue
        api_batch = None
        try:
            if hasattr(data['api'], 'batch'):
                api_batch = data['api'].batch
            elif 'batch_dict' in data._store and 'api' in data._store['batch_dict']:
                api_batch = data._store['batch_dict']['api']
        except (KeyError, AttributeError):
            pass
        
        if api_batch is None:
            # If no batch information, assume it's a single graph
            api_batch = torch.zeros(data['api'].x.size(0), dtype=torch.long, device=data['api'].x.device)
        
        # Input projection
        x_dict = {node_type: self.input_proj(x) for node_type, x in x_dict.items()}
        
        # Heterogeneous graph convolution - Add residual connections
        for i, conv in enumerate(self.convs):
            # Save input for residual connections
            if self.use_residual:
                residual_dict = {node_type: x.clone() for node_type, x in x_dict.items()}
            
            # Graph convolution
            x_dict = conv(x_dict, edge_index_dict)
            
            # Pre-activation: ReLU first, then LayerNorm (recommended order)
            for node_type in x_dict:
                x_dict[node_type] = F.relu(x_dict[node_type])                 # ReLU first
                x_dict[node_type] = self.layer_norms[i](x_dict[node_type])    # LayerNorm second
                
                # Add residual connections
                if self.use_residual:
                    x_dict[node_type] = x_dict[node_type] + residual_dict[node_type]
                
                #x_dict[node_type] = F.dropout(x_dict[node_type], p=self.dropout, training=self.training)
        
        # Get API node features
        api_features = x_dict['api']  # [num_nodes, hidden_dim]
        
        # Calculate attention weights
        attention_weights = None
        if self.use_attention:
            # Calculate attention scores
            attention_scores = self.attention_layer(api_features)  # [num_nodes, 1]
            attention_scores = attention_scores.squeeze(-1)  # [num_nodes]
            
            # Apply softmax to nodes within each graph
            attention_weights = torch.zeros_like(attention_scores)
            for batch_id in torch.unique(api_batch):
                mask = (api_batch == batch_id)
                if mask.any():
                    attention_weights[mask] = F.softmax(attention_scores[mask], dim=0)
            
            # Use attention weights for weighted pooling
            weighted_features = api_features * attention_weights.unsqueeze(-1)  # [num_nodes, hidden_dim]
            
            # Pooling based on pooling type
            if self.pooling == 'mean':
                graph_features = global_mean_pool(weighted_features, api_batch)
            elif self.pooling == 'max':
                graph_features = global_max_pool(weighted_features, api_batch)
            elif self.pooling == 'concat':
                mean_pool = global_mean_pool(weighted_features, api_batch)
                max_pool = global_max_pool(weighted_features, api_batch)
                graph_features = torch.cat([mean_pool, max_pool], dim=1)
            else:
                raise ValueError(f"Unsupported pooling method: {self.pooling}")
        else:
            # Regular pooling (without attention)
            if self.pooling == 'mean':
                graph_features = global_mean_pool(api_features, api_batch)
            elif self.pooling == 'max':
                graph_features = global_max_pool(api_features, api_batch)
            elif self.pooling == 'concat':
                mean_pool = global_mean_pool(api_features, api_batch)
                max_pool = global_max_pool(api_features, api_batch)
                graph_features = torch.cat([mean_pool, max_pool], dim=1)
            else:
                raise ValueError(f"Unsupported pooling method: {self.pooling}")
        
        return graph_features, attention_weights

    def get_graph_embedding(self, data: HeteroData) -> torch.Tensor:
        """
        Get graph embedding representation (doesn't return attention weights, maintains backward compatibility)
        
        Args:
            data: Heterogeneous graph data batch
            
        Returns:
            graph_features: Graph embedding features
        """
        graph_features, _ = self.get_graph_embedding_with_attention(data)
        return graph_features

    def forward(self, data: HeteroData, return_attention: bool = False) -> torch.Tensor:
        """
        Forward propagation
        
        Args:
            data: Heterogeneous graph data batch
            return_attention: Whether to return attention weights
            
        Returns:
            logits: Classification logits [batch_size, num_classes]
            If return_attention=True, also returns attention_weights
        """
        # Get graph embedding and attention weights
        graph_embedding, attention_weights = self.get_graph_embedding_with_attention(data)

        # Construct final classification features
        final_features = graph_embedding

        # If using permission, concatenate permission features with graph embedding
        if self.use_permission and hasattr(data, 'permission'):
            permission_features = data.permission
            final_features = torch.cat([graph_embedding, permission_features], dim=1)
        
        # Use classifier for classification
        logits = self.classifier(final_features)    
        
        if return_attention:
            return logits, attention_weights
        return logits
    
    def get_node_contributions(self, data: HeteroData, top_k: int = 5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get contribution of each node to classification
        
        Args:
            data: Single graph data
            top_k: Return top k nodes with highest contributions
            
        Returns:
            logits: Classification results
            top_indices: Indices of top k nodes
            top_scores: Contribution scores of top k nodes
        """
        self.eval()
        with torch.no_grad():
            logits, attention_weights = self.forward(data, return_attention=True)
            
            if attention_weights is not None:
                # Get top k nodes
                top_scores, top_indices = torch.topk(attention_weights, k=min(top_k, len(attention_weights)))
                return logits, top_indices, top_scores
            else:
                # If attention mechanism is not used, return empty values
                return logits, torch.tensor([]), torch.tensor([])

def create_model(input_dim: int, 
                 permission_dim: int = 0,
                model_type: str = 'hgnn',
                hidden_dim: int = 128,
                num_layers: int = 3,
                num_classes: int = 2,
                pooling: str = 'mean',
                use_permission: bool = True,
                use_attention: bool = True,
                use_residual: bool = True) -> nn.Module:
    """
    Create model
    
    Args:
        input_dim: API node feature dimension
        permission_dim: Permission feature dimension
        model_type: Model type ('hgcn')
        hidden_dim: Hidden layer dimension
        num_layers: Number of GCN layers
        num_classes: Number of classification classes
        pooling: Graph-level pooling method
        use_permission: Whether to use permission features
        use_attention: Whether to use attention mechanism
        use_residual: Whether to use residual connections
    Returns:
        Model instance
    """
    if model_type == 'hgnn':
        return HGNN(input_dim, permission_dim, hidden_dim, num_layers, num_classes, pooling, use_permission, use_attention, use_residual)

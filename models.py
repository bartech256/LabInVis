import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout, BatchNorm1d, LayerNorm
from torch_geometric.nn import GATConv, GCNConv

class SimpleGAT(torch.nn.Module):
    """
    Upgraded GAT - Much deeper with advanced attention and regularization
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, heads: int = 8, dropout: float = 0.4):
        super(SimpleGAT, self).__init__()
        
        # Input projection to handle high-dimensional input better
        self.input_proj = Linear(input_dim, hidden_dim)
        self.input_bn = BatchNorm1d(hidden_dim)
        
        # Multiple GAT layers with increasing complexity
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=heads, dropout=0.2, concat=True)
        self.bn1 = BatchNorm1d(hidden_dim * heads)
        
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=0.2, concat=True)
        self.bn2 = BatchNorm1d(hidden_dim * heads)
        
        self.gat3 = GATConv(hidden_dim * heads, hidden_dim//2, heads=heads//2, dropout=0.2, concat=True)
        self.bn3 = BatchNorm1d((hidden_dim//2) * (heads//2))
        
        # Final attention layer (single head for consolidation)
        self.gat_final = GATConv((hidden_dim//2) * (heads//2), hidden_dim//2, heads=1, dropout=0.1)
        self.bn_final = BatchNorm1d(hidden_dim//2)
        
        # Advanced prediction head with multiple layers
        self.pred_layers = torch.nn.Sequential(
            Linear(hidden_dim//2, hidden_dim),
            LayerNorm(hidden_dim),
            torch.nn.ELU(),  # Fixed: Use module instead of function
            Dropout(dropout),
            Linear(hidden_dim, hidden_dim//2),
            LayerNorm(hidden_dim//2),
            torch.nn.ELU(),  # Fixed: Use module instead of function
            Dropout(dropout),
            Linear(hidden_dim//2, hidden_dim//4),
            torch.nn.ELU(),  # Fixed: Use module instead of function
            Linear(hidden_dim//4, 1)
        )
        
        self.dropout = Dropout(dropout)
        
        # Calculate total parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"SimpleGAT upgraded: {total_params:,} parameters, {heads} heads, hidden_dim={hidden_dim}")

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Input projection
        x = self.input_proj(x)
        x = self.input_bn(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        # First GAT layer
        x = self.gat1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        # Second GAT layer
        x = self.gat2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        # Third GAT layer
        x = self.gat3(x, edge_index)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        # Final GAT layer
        x = self.gat_final(x, edge_index)
        x = self.bn_final(x)
        x = F.elu(x)
        
        # Advanced prediction through the sequential layers (Fixed)
        out = self.pred_layers(x)
        return out.squeeze(-1)

class SimpleGCN(torch.nn.Module):
    """
    Upgraded GCN - Much deeper with residual connections and advanced features
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.4):
        super(SimpleGCN, self).__init__()
        
        # Input projection
        self.input_proj = Linear(input_dim, hidden_dim)
        self.input_bn = BatchNorm1d(hidden_dim)
        
        # Multiple GCN layers with decreasing dimensions
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.bn1 = BatchNorm1d(hidden_dim)
        
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = BatchNorm1d(hidden_dim)
        
        self.conv3 = GCNConv(hidden_dim, hidden_dim//2)
        self.bn3 = BatchNorm1d(hidden_dim//2)
        
        self.conv4 = GCNConv(hidden_dim//2, hidden_dim//2)
        self.bn4 = BatchNorm1d(hidden_dim//2)
        
        self.conv5 = GCNConv(hidden_dim//2, hidden_dim//4)
        self.bn5 = BatchNorm1d(hidden_dim//4)
        
        # Advanced prediction head
        self.prediction_head = torch.nn.Sequential(
            Linear(hidden_dim//4, hidden_dim//2),
            LayerNorm(hidden_dim//2),
            torch.nn.ReLU(),
            Dropout(dropout),
            Linear(hidden_dim//2, hidden_dim//4),
            LayerNorm(hidden_dim//4),
            torch.nn.ReLU(),
            Dropout(dropout/2),
            Linear(hidden_dim//4, hidden_dim//8),
            torch.nn.ReLU(),
            Linear(hidden_dim//8, 1)
        )
        
        self.dropout = Dropout(dropout)
        
        # Calculate total parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"SimpleGCN upgraded: {total_params:,} parameters, 5 conv layers, hidden_dim={hidden_dim}")

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Input projection
        x = self.input_proj(x)
        x = self.input_bn(x)
        x0 = F.relu(x)  # Store for residual
        
        # First GCN layer
        x = self.conv1(x0, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second GCN layer with residual connection
        x1 = self.conv2(x, edge_index)
        x1 = self.bn2(x1)
        x1 = F.relu(x1)
        x1 = x1 + x  # Residual connection
        x = self.dropout(x1)
        
        # Third GCN layer
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Fourth GCN layer with residual
        x2 = self.conv4(x, edge_index)
        x2 = self.bn4(x2)
        x2 = F.relu(x2)
        x2 = x2 + x  # Residual connection
        x = self.dropout(x2)
        
        # Fifth GCN layer
        x = self.conv5(x, edge_index)
        x = self.bn5(x)
        x = F.relu(x)
        
        # Advanced prediction
        out = self.prediction_head(x)
        return out.squeeze(-1)

class MultiLayerGCN(torch.nn.Module):
    """
    Highly configurable deep GCN with dynamic architecture
    """
    def __init__(self, input_dim: int, hidden_dims: list = [256, 128, 64, 32, 16], dropout: float = 0.4):
        super(MultiLayerGCN, self).__init__()
        
        self.num_layers = len(hidden_dims)
        
        # Input projection for better feature transformation
        self.input_proj = Linear(input_dim, hidden_dims[0])
        self.input_bn = BatchNorm1d(hidden_dims[0])
        
        # Dynamic layer construction
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.layer_norms = torch.nn.ModuleList()
        
        for i in range(self.num_layers):
            if i == 0:
                in_dim = hidden_dims[0]
            else:
                in_dim = hidden_dims[i-1]
            out_dim = hidden_dims[i]
            
            self.convs.append(GCNConv(in_dim, out_dim))
            self.bns.append(BatchNorm1d(out_dim))
            self.layer_norms.append(LayerNorm(out_dim))
        
        # Advanced multi-stage prediction head
        pred_input_dim = hidden_dims[-1]
        self.prediction_stages = torch.nn.ModuleList([
            # Stage 1: Expansion
            torch.nn.Sequential(
                Linear(pred_input_dim, pred_input_dim * 2),
                LayerNorm(pred_input_dim * 2),
                torch.nn.GELU(),
                Dropout(dropout)
            ),
            # Stage 2: Processing
            torch.nn.Sequential(
                Linear(pred_input_dim * 2, pred_input_dim),
                LayerNorm(pred_input_dim),
                torch.nn.GELU(),
                Dropout(dropout/2)
            ),
            # Stage 3: Compression
            torch.nn.Sequential(
                Linear(pred_input_dim, pred_input_dim // 2),
                torch.nn.GELU(),
                Linear(pred_input_dim // 2, 1)
            )
        ])
        
        self.dropout = Dropout(dropout)
        
        # Calculate and report parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"MultiLayerGCN upgraded: {total_params:,} parameters, {self.num_layers} layers, dims={hidden_dims}")

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Input projection
        x = self.input_proj(x)
        x = self.input_bn(x)
        x = F.gelu(x)
        
        # Store intermediate outputs for potential skip connections
        layer_outputs = []
        
        # Pass through all GCN layers
        for i, (conv, bn, ln) in enumerate(zip(self.convs, self.bns, self.layer_norms)):
            x_new = conv(x, edge_index)
            x_new = bn(x_new)
            x_new = F.gelu(x_new)
            
            # Add residual connection for deeper layers (if dimensions match)
            if i > 1 and x.shape[1] == x_new.shape[1]:
                x_new = x_new + x
            
            x_new = ln(x_new)  # Layer normalization for stability
            x_new = self.dropout(x_new)
            
            layer_outputs.append(x_new)
            x = x_new
        
        # Multi-stage prediction
        for stage in self.prediction_stages:
            x = stage(x)
        
        return x.squeeze(-1)

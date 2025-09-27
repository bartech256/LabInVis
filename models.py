"""
Responsibility:
- Define all machine learning models (GNNs and baselines).
- Includes:
  - SimpleGCN
  - SimpleGAT
  - MultiLayerGCN
  - GraphSAGE
  - CombinedModel (to combine GNN + regression)
  - MLPRegressor (baseline)
  - LinearRegressionTorch (baseline)
"""

import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout, BatchNorm1d, LayerNorm
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, GINConv, TransformerConv, SuperGATConv, DNAConv, Sequential

# ======================
#   GNN MODELS
# ======================

class SimpleGAT(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, heads: int = 8, dropout: float = 0.4):
        super(SimpleGAT, self).__init__()
        self.input_proj = Linear(input_dim, hidden_dim)
        self.input_bn = BatchNorm1d(hidden_dim)
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=heads, dropout=0.2, concat=True)
        self.bn1 = BatchNorm1d(hidden_dim * heads)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=0.2, concat=True)
        self.bn2 = BatchNorm1d(hidden_dim * heads)
        self.gat3 = GATConv(hidden_dim * heads, hidden_dim//2, heads=heads//2, dropout=0.2, concat=True)
        self.bn3 = BatchNorm1d((hidden_dim//2)*(heads//2))
        self.gat_final = GATConv((hidden_dim//2)*(heads//2), hidden_dim//2, heads=1, dropout=0.1)
        self.bn_final = BatchNorm1d(hidden_dim//2)

        self.pred_layers = torch.nn.Sequential(
            Linear(hidden_dim//2, hidden_dim),
            LayerNorm(hidden_dim),
            torch.nn.ELU(),
            Dropout(dropout),
            Linear(hidden_dim, hidden_dim//2),
            LayerNorm(hidden_dim//2),
            torch.nn.ELU(),
            Dropout(dropout),
            Linear(hidden_dim//2, hidden_dim//4),
            torch.nn.ELU(),
            Linear(hidden_dim//4, 1)
        )
        self.dropout = Dropout(dropout)
        print(f"SimpleGAT initialized: hidden_dim={hidden_dim}, heads={heads}")

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.input_proj(x)
        x = self.input_bn(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.gat1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.gat2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.gat3(x, edge_index)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.gat_final(x, edge_index)
        x = self.bn_final(x)
        x = F.elu(x)

        out = self.pred_layers(x)
        return out.squeeze(-1)


class SimpleGCN(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.4):
        super(SimpleGCN, self).__init__()
        self.input_proj = Linear(input_dim, hidden_dim)
        self.input_bn = BatchNorm1d(hidden_dim)

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
        print(f"SimpleGCN initialized: hidden_dim={hidden_dim}")

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.input_proj(x)
        x = self.input_bn(x)
        x0 = F.relu(x)

        x = self.conv1(x0, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x1 = self.conv2(x, edge_index)
        x1 = self.bn2(x1)
        x1 = F.relu(x1)
        x1 = x1 + x
        x = self.dropout(x1)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)

        x2 = self.conv4(x, edge_index)
        x2 = self.bn4(x2)
        x2 = F.relu(x2)
        x2 = x2 + x
        x = self.dropout(x2)

        x = self.conv5(x, edge_index)
        x = self.bn5(x)
        x = F.relu(x)

        out = self.prediction_head(x)
        return out.squeeze(-1)


class MultiLayerGCN(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list = [256, 128, 64, 32, 16], dropout: float = 0.4):
        super(MultiLayerGCN, self).__init__()
        self.num_layers = len(hidden_dims)
        self.input_proj = Linear(input_dim, hidden_dims[0])
        self.input_bn = BatchNorm1d(hidden_dims[0])

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.layer_norms = torch.nn.ModuleList()

        for i in range(self.num_layers):
            in_dim = hidden_dims[i-1] if i > 0 else hidden_dims[0]
            out_dim = hidden_dims[i]
            self.convs.append(GCNConv(in_dim, out_dim))
            self.bns.append(BatchNorm1d(out_dim))
            self.layer_norms.append(LayerNorm(out_dim))

        pred_input_dim = hidden_dims[-1]
        self.prediction_stages = torch.nn.ModuleList([
            torch.nn.Sequential(
                Linear(pred_input_dim, pred_input_dim * 2),
                LayerNorm(pred_input_dim * 2),
                torch.nn.GELU(),
                Dropout(dropout)
            ),
            torch.nn.Sequential(
                Linear(pred_input_dim * 2, pred_input_dim),
                LayerNorm(pred_input_dim),
                torch.nn.GELU(),
                Dropout(dropout/2)
            ),
            torch.nn.Sequential(
                Linear(pred_input_dim, pred_input_dim // 2),
                torch.nn.GELU(),
                Linear(pred_input_dim // 2, 1)
            )
        ])

        self.dropout = Dropout(dropout)
        print(f"MultiLayerGCN initialized with {self.num_layers} layers, dims={hidden_dims}")

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.input_proj(x)
        x = self.input_bn(x)
        x = F.gelu(x)

        for i, (conv, bn, ln) in enumerate(zip(self.convs, self.bns, self.layer_norms)):
            x_new = conv(x, edge_index)
            x_new = bn(x_new)
            x_new = F.gelu(x_new)

            if i > 1 and x.shape[1] == x_new.shape[1]:
                x_new = x_new + x

            x_new = ln(x_new)
            x_new = self.dropout(x_new)
            x = x_new

        for stage in self.prediction_stages:
            x = stage(x)

        return x.squeeze(-1)


class GraphSAGE(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.4, attn=False, num_lin=2, conv='sage'):
        super(GraphSAGE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_lin = num_lin
        self.dropout = dropout
        self.attn = attn
        self.loss = torch.nn.MSELoss()

        if conv == 'sage':
            conv_layer1 = SAGEConv(input_dim, hidden_dim)
            conv_layer2 = SAGEConv(hidden_dim, hidden_dim)
        elif conv == 'gin':
            lin1 = torch.nn.ModuleList([Linear(input_dim if i==0 else hidden_dim, hidden_dim) for i in range(num_lin)])
            lin2 = torch.nn.ModuleList([Linear(hidden_dim, hidden_dim) for _ in range(num_lin)])
            conv_layer1 = GINConv(torch.nn.Sequential(*lin1))
            conv_layer2 = GINConv(torch.nn.Sequential(*lin2))
        elif conv == 'transformer':
            conv_layer1 = TransformerConv(input_dim, hidden_dim)
            conv_layer2 = TransformerConv(hidden_dim, hidden_dim)
        elif conv == 'gat':
            conv_layer1 = SuperGATConv(input_dim, hidden_dim)
            conv_layer2 = SuperGATConv(hidden_dim, hidden_dim)
        elif conv == 'dna':
            conv_layer1 = DNAConv(input_dim, hidden_dim)
            conv_layer2 = DNAConv(hidden_dim, hidden_dim)
        else:
            raise ValueError(f'Invalid convolution layer {conv}')

        self.convs = Sequential('x, edge_index', [
            (conv_layer1, 'x, edge_index -> x'),
            torch.nn.ReLU(inplace=True),
            (conv_layer2, 'x, edge_index -> x'),
            torch.nn.ReLU(inplace=True),
            (torch.nn.Dropout(p=dropout), 'x -> x')
        ])
        self.fc = Linear(hidden_dim, 1)
        print(f"GraphSAGE initialized: hidden_dim={hidden_dim}, conv={conv}")

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.convs(x, edge_index)
        out = self.fc(x)
        return out.squeeze(-1)


# ======================
#   COMBINED + BASELINES
# ======================

class CombinedModel(torch.nn.Module):
    """Combines a GNN model with a regression head."""
    def __init__(self, GNN_model=None, regression_model=None):
        super(CombinedModel, self).__init__()
        self.GNN_model = GNN_model
        self.regression_model = regression_model

    def forward(self, data):
        x = self.GNN_model(data) if self.GNN_model else data.x
        return self.regression_model(x) if self.regression_model else x


class MLPRegressor(torch.nn.Module):
    """Simple Multi-Layer Perceptron for regression."""
    def __init__(self, input_dim, hidden_dim=128, dropout=0.3):
        super(MLPRegressor, self).__init__()
        self.net = torch.nn.Sequential(
            Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            Dropout(dropout),
            Linear(hidden_dim, hidden_dim//2),
            torch.nn.ReLU(),
            Dropout(dropout),
            Linear(hidden_dim//2, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class LinearRegressionTorch(torch.nn.Module):
    """Basic linear regression implemented in PyTorch."""
    def __init__(self, input_dim):
        super(LinearRegressionTorch, self).__init__()
        self.linear = Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)

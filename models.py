"""
this file contains the model definitions for the application.
"""

class BaseGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
    def forward(self, data):
        raise NotImplementedError

class SimpleGNN(BaseGNN):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__(...)
        # one or two graph‑conv layers
    def forward(self, data):
        # x, edge_index = data.x, data.edge_index
        # apply conv → activation → readout → linear

class MultiLayerGNN(BaseGNN):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super().__init__(...)
        # stack num_layers of graph‑conv + activations
    def forward(self, data):
        # iterate through layers, then global pooling, then MLP head
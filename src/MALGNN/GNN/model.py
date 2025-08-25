import torch
from torch_geometric.nn import GINConv
from torch.nn import Linear
from torch.nn import BatchNorm1d
from torch_geometric.nn import BatchNorm
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool



class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin1 = Linear(in_dim, out_dim)
        self.lin2 = Linear(out_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x
        
class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_layers=3, dropout=0.5):
        super().__init__()
        self.gin_layers = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.dropout = dropout

        # First GIN layer
        mlp = MLP(in_channels, hidden_channels)
        self.gin_layers.append(GINConv(mlp))
        self.bns.append(BatchNorm(hidden_channels))

        # K GIN layers
        for _ in range(num_layers - 1):
            mlp = MLP(hidden_channels, hidden_channels)
            self.gin_layers.append(GINConv(mlp))
            self.bns.append(BatchNorm(hidden_channels))

        # Classifier head with BatchNorm
        self.head_lin1 = Linear(hidden_channels, hidden_channels)
        self.head_bn1 = BatchNorm1d(hidden_channels)
        self.head_lin2 = Linear(hidden_channels, hidden_channels // 2)
        self.head_bn2 = BatchNorm1d(hidden_channels // 2)
        self.out = Linear(hidden_channels // 2, num_classes)

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.gin_layers, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)

        x = self.head_lin1(x)
        x = self.head_bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.head_lin2(x)
        x = self.head_bn2(x)
        x = F.relu(x)
        x = self.out(x)

        return x

import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import torch
import dgl
from dgl.nn.pytorch import SAGEConv
from utils import *

class SAGE(nn.Module):
    def __init__(self, x_size, hidden_size, gnn_layers, aggregator_type, dropout):
        super(SAGE, self).__init__()
        self.hidden_size = hidden_size
        self.gnn_layers = gnn_layers
        self.aggregator_type = aggregator_type

        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(x_size, hidden_size, aggregator_type=aggregator_type, activation=F.relu))
        for _ in range(gnn_layers - 1):
            self.layers.append(SAGEConv(hidden_size, hidden_size, aggregator_type=aggregator_type, activation=F.relu))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, x):
        h = x
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return self.dropout(h)
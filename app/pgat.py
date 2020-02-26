import numpy as np
import torch.nn.functional as F
import torch
from torch_geometric.nn import PAGATConv


class PAGAT(torch.nn.Module):
    def __init__(self, emb_dim, repr_dim, hidden_size, heads, dropout=0.5, path_dropout=0.5):
        super(PAGAT, self).__init__()

        self.path_dropout = path_dropout
        self.dropout = dropout
        self.conv1 = PAGATConv(
            emb_dim,
            hidden_size,
            heads=heads,
            dropout=dropout,
        )
        self.conv2 = PAGATConv(
            heads * hidden_size,
            repr_dim,
            heads=heads,
            concat=False,
            dropout=dropout,
        )

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, path_index):
        if self.training:
            path_num = path_index.shape[1]
            path_index = path_index[:, np.random.choice(range(path_num), int(path_num * (1 - self.path_dropout)))]
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, path_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, path_index)
        return x

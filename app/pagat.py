import torch
import torch.nn.functional as F
import torch
import numpy as np
from torch_geometric.nn import PAGATConv


class PAGATNet(torch.nn.Module):
    def __init__(self, num_nodes, emb_dim, repr_dim, heads, dropout=0.5, path_dropout=0.5):
        super(PAGATNet, self).__init__()

        self.node_emb = torch.nn.Embedding(num_nodes, emb_dim, max_norm=1, norm_type=2.0)

        self.path_dropout = path_dropout
        self.dropout = dropout
        self.conv = PAGATConv(
            emb_dim,
            repr_dim,
            heads=heads,
            dropout=dropout,
            concat=False
        )

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, path_index):
        if self.training:
            path_num = path_index.shape[1]
            path_index = path_index[:, np.random.choice(range(path_num), int(path_num * (1 - self.path_dropout)))]
        x = F.dropout(x, p=self.dropout, training=self.training)
        x, att = self.conv(x, path_index)
        return x, att

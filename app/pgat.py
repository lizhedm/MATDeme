import torch
import torch.nn.functional as F
import torch
from torch_geometric.nn import PAGATConv


class PAGATNet(torch.nn.Module):
    def __init__(self, emb_dim, repr_dim, heads=4, dropout=0.6):
        super(PAGATNet, self).__init__()
        self.emb_dim = emb_dim
        self.repr_dim = repr_dim

        self.conv = PAGATConv(emb_dim, repr_dim, heads=heads, dropout=dropout)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, path):
        return self.conv(x, path)


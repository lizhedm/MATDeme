import torch
import torch.nn.functional as F
import torch
from torch_geometric.nn import GATConv, PAConv


class PAGATNet(torch.nn.Module):
    def __init__(self, num_nodes, emb_dim, repr_dim, heads=4, dropout=0.6):
        super(PAGATNet, self).__init__()
        self.emb_dim = emb_dim
        self.repr_dim = repr_dim

        self.node_emb = torch.nn.Embedding(num_nodes, emb_dim, max_norm=1, norm_type=2.0)

        self.conv = PAConv(emb_dim, repr_dim, heads=heads, dropout=dropout)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward_(self, x, path):
        return self.conv(x, path)

    def forward(self, path):
        '''

        :param edge_index: np.array, [2, N]
        :param sec_order_edge_index: [3, M]
        :return:
        '''
        return self.forward_(self.node_emb.weight, path)


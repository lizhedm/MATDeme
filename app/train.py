import argparse
import torch
from torch.utils.data import DataLoader
from torch_geometric.datasets import MovieLens
from torch.optim import Adam
import time
import numpy as np

from utils import get_folder_path
from pagat import PAGAT
from eval_rec_sys import metrics
parser = argparse.ArgumentParser()

# Dataset params
parser.add_argument("--dataset", type=str, default='movielens', help="")
parser.add_argument("--dataset_name", type=str, default='1m', help="")
parser.add_argument("--num_core", type=int, default=10, help="")
parser.add_argument("--step_length", type=int, default=2, help="")
parser.add_argument("--train_ratio", type=float, default=0.8, help="")
parser.add_argument("--debug", default=0.01, help="")

# Model params
parser.add_argument("--heads", type=int, default=4, help="")
parser.add_argument("--emb_dim", type=int, default=16, help="")
parser.add_argument("--repr_dim", type=int, default=8, help="")
parser.add_argument("--hidden_size", type=int, default=64, help="")

# Train params
parser.add_argument("--device", type=str, default='cpu', help="")
parser.add_argument("--gpu_idx", type=str, default='0', help="")
parser.add_argument("--epochs", type=int, default=1000, help="")
parser.add_argument("--opt", type=str, default='adam', help="")
parser.add_argument("--loss", type=str, default='mse', help="")
parser.add_argument("--batch_size", type=int, default=16, help="")
parser.add_argument("--lr", type=float, default=1e-4, help="")
parser.add_argument("--weight_decay", type=float, default=0, help="")
parser.add_argument("--early_stopping", type=int, default=40, help="")

# Recommender params
parser.add_argument("--num_recs", type=int, default=10, help="")


args = parser.parse_args()

# Setup data and weights file path
data_folder, weights_folder, logger_folder = get_folder_path(args.dataset + args.dataset_name)

# Setup device
if not torch.cuda.is_available() or args.device == 'cpu':
    device = 'cpu'
else:
    device = 'cuda:{}'.format(args.gpu_idx)

# Setup args
dataset_args = {
    'root': data_folder, 'dataset': args.dataset, 'name': args.dataset_name,
    'num_core': args.num_core, 'step_length': args.step_length, 'train_ratio': args.train_ratio,
    'debug': args.debug
}
model_args = {
    'heads': args.heads, 'hidden_size': args.hidden_size, 'emb_dim': args.emb_dim,
    'repr_dim': args.repr_dim
}
train_args = {
    'debug': args.debug,
    'opt': args.opt, 'loss': args.loss,
    'epochs': args.epochs, 'batch_size': args.batch_size,
    'weight_decay': args.weight_decay, 'lr': args.lr, 'device': device,
    'weights_folder': weights_folder, 'logger_folder': logger_folder}
rec_args = {
    'num_recs': args.num_recs
}
print('dataset params: {}'.format(dataset_args))
print('task params: {}'.format(model_args))
print('train params: {}'.format(train_args))
print('rec params: {}'.format(rec_args))


if __name__ == '__main__':
    dataset = MovieLens(**dataset_args)
    dataset.data = dataset.data.to(train_args['device'])
    model = PAGAT(num_nodes=dataset.data.num_nodes[0], **model_args).to(train_args['device'])
    optimizer = Adam(model.parameters(), lr=train_args['lr'], weight_decay=train_args['weight_decay'])
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_start = time.perf_counter()

    HR_history = []
    NDCG_history = []
    loss_history = []
    for epoch in range(1, train_args['epochs'] + 1):
        data = dataset.data
        user_pos_neg_pair = data.train_user_pos_neg_pair[0]
        data_loader = DataLoader(user_pos_neg_pair, shuffle=True, batch_size=train_args['batch_size'])

        model.train()
        for user_pos_neg_pair_batch in data_loader:
            u_nid, pos_i_nid, neg_i_nid = user_pos_neg_pair_batch.T
            occ_nid = np.concatenate((u_nid, pos_i_nid, neg_i_nid))
            path_index_batch = torch.from_numpy(data.path_np[0][:, np.isin(data.path_np[0][-1, :], occ_nid)]).to(train_args['device'])
            propagated_node_emb = model(path_index_batch)

            u_nid, pos_i_nid, neg_i_nid = u_nid.to(device), pos_i_nid.to(device), neg_i_nid.to(device)
            u_node_emb, pos_i_node_emb, neg_i_node_emb = propagated_node_emb[u_nid], propagated_node_emb[pos_i_nid], propagated_node_emb[neg_i_nid]
            pred_pos = (u_node_emb * pos_i_node_emb).sum(dim=1)
            pred_neg = (u_node_emb * neg_i_node_emb).sum(dim=1)
            loss = - (pred_pos - pred_neg).sigmoid().log().sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        HR, NDCG, loss = metrics(model, dataset, train_args, rec_args)

        print('Epoch: {}, HR: {}, NDCG: {}, Loss: {}'.format(epoch, HR, NDCG, loss))
    t_end = time.perf_counter()

    print('Duration: {}, HR: {}, NDCG: {}, loss: {}'.format(t_start - t_end, np.mean(HR_history), np.mean(NDCG_history), np.mean(loss_history)))

    if torch.cuda.is_available():
        torch.cuda.synchronize()


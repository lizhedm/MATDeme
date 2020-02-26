import argparse
import torch
from torch_geometric.datasets import MovieLens
from torch.optim import Adam
import time
import numpy as np

from utils import get_folder_path
from pgat import PAGAT
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
parser.add_argument("--emb_dim", type=int, default=64, help="")
parser.add_argument("--repr_dim", type=int, default=16, help="")
parser.add_argument("--hidden_size", type=int, default=64, help="")

# Train params
parser.add_argument("--device", type=str, default='cpu', help="")
parser.add_argument("--gpu_idx", type=str, default='0', help="")
parser.add_argument("--epochs", type=int, default=1000, help="")
parser.add_argument("--opt", type=str, default='adam', help="")
parser.add_argument("--loss", type=str, default='mse', help="")
parser.add_argument("--batch_size", type=int, default=256, help="")
parser.add_argument("--lr", type=float, default=1e-4, help="")
parser.add_argument("--weight_decay", type=float, default=0, help="")
parser.add_argument("--early_stopping", type=int, default=40, help="")

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
print('dataset params: {}'.format(dataset_args))
print('task params: {}'.format(model_args))
print('train params: {}'.format(train_args))

if __name__ == '__main__':
    data = MovieLens(**dataset_args).data.to(train_args['device'])
    model = PAGAT(**model_args).to(train_args['device'])
    optimizer = Adam(model.parameters(), lr=train_args['lr'], weight_decay=train_args['weight_decay'])
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_start = time.perf_counter()

    best_val_loss = float('inf')
    test_acc = 0
    val_loss_history = []


    for epoch in range(1, train_args['epochs'] + 1):
        user_pos_neg_pair = data.train_user_pos_neg_pair[0]
        data_loader = DataLoader(user_pos_neg_pair.T, batch_size=train_args['batch_size'])

        model.train()
        for user_pos_neg_pair_batch in data_loader:
            propagated_node_emb = model(data)
            u_nid, pos_i_nid, neg_i_nid = user_pos_neg_pair_batch.T
            u_node_emb, pos_i_node_emb, neg_i_node_emb = propagated_node_emb[u_nid], propagated_node_emb[pos_i_nid], propagated_node_emb[neg_i_nid]
            pred_pos = (u_node_emb * pos_i_node_emb).sum(dim=1)
            pred_neg = (u_node_emb * neg_i_node_emb).sum(dim=1)
            loss = - (pred_pos - pred_neg).sigmoid().log().sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        u_nid, pos_i_nid, neg_i_nid = data.test_user_pos_neg_pair[0]
        propagated_node_emb = model(data)
        u_node_emb, pos_i_node_emb, neg_i_node_emb = propagated_node_emb[u_nid], propagated_node_emb[pos_i_nid], \
                                                     propagated_node_emb[neg_i_nid]
        pred_pos = (u_node_emb * pos_i_node_emb).sum(dim=1)
        pred_neg = (u_node_emb * neg_i_node_emb).sum(dim=1)
        loss = - (pred_pos - pred_neg).sigmoid().log().sum().values


        eval_info = evaluate(model, data)
        eval_info['epoch'] = epoch

        if logger is not None:
            logger(eval_info)

        if eval_info['val_loss'] < best_val_loss:
            best_val_loss = eval_info['val_loss']
            test_acc = eval_info['test_acc']
            print('Epoch {}, acc {}'.format(epoch, test_acc))

        val_loss_history.append(eval_info['val_loss'])
        if early_stopping > 0 and epoch > epochs // 2:
            tmp = tensor(val_loss_history[-(early_stopping + 1):-1])
            if eval_info['val_loss'] > tmp.mean().item():
                break




    print('Accuracy of the run {} is {}'.format(_ + 1, test_acc))

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_end = time.perf_counter()

    val_losses.append(best_val_loss)
    accs.append(test_acc)
    durations.append(t_end - t_start)

loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)
    print(dataset.data.train_path[0].shape)
import argparse
import torch
from torch_geometric.datasets import MovieLens

from pgat import PAGATNet
from utils import get_folder_path

__model__ = 'PGAT'

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

# Train params
parser.add_argument("--device", type=str, default='cuda', help="")
parser.add_argument("--gpu_idx", type=str, default='0', help="")
parser.add_argument("--runs", type=int, default=10, help="")
parser.add_argument("--epochs", type=int, default=50, help="")
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
    'heads': args.heads, 'emb_dim': args.emb_dim, 'repr_dim': args.repr_dim
}
train_args = {
    'debug': args.debug, 'runs': args.runs,
    'model': __model__,
    'opt': args.opt, 'loss': args.loss,
    'epochs': args.epochs, 'batch_size': args.batch_size,
    'weight_decay': args.weight_decay, 'lr': args.lr, 'device': device,
    'weights_folder': weights_folder, 'logger_folder': logger_folder}
print('dataset params: {}'.format(dataset_args))
print('task params: {}'.format(model_args))
print('train params: {}'.format(train_args))

if __name__ == '__main__':
    data = MovieLens(**dataset_args).data.to(train_args['device'])
    paths = data.train_path[0]
    node_emb = torch.nn.Embedding(data.num_nodes[0], model_args['emb_dim'], max_norm=1, norm_type=2.0)
    model = PAGATNet(**model_args).to(train_args['device'])
from flask import render_template
from app import app
import pandas as pd
import requests
import json
import random
import argparse
import torch
from torch_geometric.datasets import MovieLens
from .pgat_recsys import PGATRecSys
from .utils import get_folder_path

parser = argparse.ArgumentParser()
default_poster_src = 'https://www.nehemiahmfg.com/wp-content/themes/dante/images/default-thumb.png'

# Dataset params
parser.add_argument("--dataset", type=str, default='movielens', help="")
parser.add_argument("--dataset_name", type=str, default='1m', help="")
parser.add_argument("--num_core", type=int, default=10, help="")
parser.add_argument("--sec_order", type=bool, default=False, help="")
parser.add_argument("--train_ratio", type=float, default=0.8, help="")
parser.add_argument("--debug", default=0.1, help="")

# Model params
parser.add_argument("--heads", type=int, default=4, help="")
parser.add_argument("--hidden_size", type=int, default=64, help="")
parser.add_argument("--emb_dim", type=int, default=300, help="")
parser.add_argument("--repr_dim", type=int, default=16, help="")

# Train params
parser.add_argument("--device", type=str, default='cuda', help="")
parser.add_argument("--gpu_idx", type=str, default='2', help="")
parser.add_argument("--runs", type=int, default=10, help="")
parser.add_argument("--epochs", type=int, default=50, help="")
parser.add_argument("--kg_opt", type=str, default='adam', help="")
parser.add_argument("--cf_opt", type=str, default='adam', help="")
parser.add_argument("--kg_loss", type=str, default='mse', help="")
parser.add_argument("--cf_loss", type=str, default='mse', help="")
parser.add_argument("--kg_batch_size", type=int, default=1028, help="")
parser.add_argument("--cf_batch_size", type=int, default=256, help="")
parser.add_argument("--sec_order_batch_size", type=int, default=1028, help="")
parser.add_argument("--lr", type=float, default=1e-4, help="")
parser.add_argument("--weight_decay", type=float, default=0, help="")
parser.add_argument("--early_stopping", type=int, default=40, help="")

args = parser.parse_args()

__model__ = 'PGAT'

# Setup data and weights file path
data_folder, weights_folder, logger_folder = get_folder_path(__model__, args.dataset + args.dataset_name)

# Setup the torch device
if not torch.cuda.is_available() or args.device == 'cpu':
    device = 'cpu'
else:
    device = 'cuda:{}'.format(args.gpu_idx)

dataset_args = {
    'root': data_folder, 'dataset': args.dataset, 'name': args.dataset_name,
    'num_core': args.num_core, 'sec_order': args.sec_order, 'train_ratio': args.train_ratio,
    'debug': args.debug
}
model_args = {
    'heads': args.heads, 'hidden_size': args.hidden_size, 'emb_dim': args.emb_dim,
    'repr_dim': args.repr_dim
}
train_args = {
    'debug': args.debug, 'runs': args.runs,
    'model': __model__,
    'kg_opt': args.kg_opt, 'kg_loss': args.kg_loss, 'cf_loss': args.cf_loss, 'cf_opt': args.cf_opt,
    'epochs': args.epochs, 'sec_order_batch_size': args.sec_order_batch_size, 'kg_batch_size': args.kg_batch_size, 'cf_batch_size': args.cf_batch_size,
    'weight_decay': args.weight_decay, 'lr': args.lr, 'device': device,
    'weights_folder': weights_folder, 'logger_folder': logger_folder}
print('dataset params: {}'.format(dataset_args))
print('task params: {}'.format(model_args))
print('train params: {}'.format(train_args))

@app.template_global()
def generateIDs(n):
    recsys = PGATRecSys(num_recs=n, train_args=train_args, model_args=model_args, dataset_args=dataset_args)
    movie_df = recsys.get_top_n_popular_items(n)
    raw_iids = [recsys.data.iid2raw_iid[0][iid] for iid in movie_df.iid]
    return raw_iids

@app.template_global()
def get_movie_name_withID(i):
    movies = pd.read_csv('/Users/lizhe/Dropbox/HCI_Master/SS19/Thesis/MADemo/app/ml-1m/movies.dat', sep='::', engine='python')
    movie_name = movies['MovieName'][i]
    return movie_name

@app.template_global()
def get_movie_poster_withID(i):

    # apikey = 'e760129c'
    apikey = 'e44e5305'
    movies = pd.read_csv('/Users/lizhe/Dropbox/HCI_Master/SS19/Thesis/MADemo/app/ml-1m/movies.dat', sep='::', engine='python')
    movie_name = get_movie_name_withID(i)
    movie_title = movie_name[0:-7]
    movie_year = movie_name[-5:-1]

    movie_url = "http://www.omdbapi.com/?" + "t=" + movie_title + "&y=" + movie_year + "&apikey=" + apikey

    try:
        r = requests.get(movie_url)
        movie_info_dic = json.loads(r.text)
        poster = movie_info_dic['Poster']
        return poster
    except:
        return default_poster_src



@app.route('/')
@app.route('/index')
def index():
    user = {'username':'Zhe'}
    return render_template('index.html', title = 'Film Recommendation', user = user)

@app.route('/explanation')
def explanation():
    return render_template('explanation.html', title = 'Film Recommendation')

@app.route('/user_information')
def user_information():
    return render_template('user_information.html', title = 'Film Recommendation')

@app.route('/user_background')
def user_background():
    return render_template('user_background.html',title = 'Film Recommendation')

@app.route('/movie_preview')
def movie_preview():
    return render_template('movie_preview.html',title = 'Film Recommendation')

@app.route('/movie_degree')
def movie_degree():
    return render_template('movie_degree.html',title = 'Film Recommendation')

@app.route('/recommendation_explanation')
def recommendation_explanation():
    return render_template('recommendation_explanation.html',title = 'Film Recommendation')

@app.route('/recommendation_evaluation')
def recommendation_evaluation():
    return render_template('recommendation_evaluation.html',title = 'Film Recommendation')

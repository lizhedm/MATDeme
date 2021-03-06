__model__ = 'PGAT'

import numpy as np
import pandas as pd
import torch
import os.path as osp
import operator


from torch_geometric.datasets import MovieLens

from pgat import PAGATNet
from torch_geometric.datasets import MovieLens
from torch_geometric import utils
from utils import get_folder_path


class PGATRecSys(object):
    def __init__(self, num_recs, dataset_args, model_args, device_args):
        self.num_recs = num_recs
        self.device_args = device_args

        self.data = MovieLens(**dataset_args).data.to(device_args['device'])

        self.node_emb = torch.nn.Embedding(self.data.num_nodes[0], model_args['emb_dim'], max_norm=1, norm_type=2.0)
        self.model = PAGATNet(**model_args).to(device_args['device'])

    def get_top_n_popular_items(self, n=10):
        """
        Get the top n movies from self.data.ratings.
        Remove the duplicates in self.data.ratings and sort it by movie count.
        After you find the top N popular movies' item id,
        look over the details information of item in self.data.movies

        :param n: the number of items, int
        :return: df: popular item dataframe, df
        """

        ratings_df = self.data.ratings[0][['iid', 'movie_count']]
        ratings_df = ratings_df.drop_duplicates()
        ratings_df = ratings_df.sort_values(by='movie_count', ascending=False)

        return ratings_df[:n]

    def build_user(self, iids, demographic_info):
        """
        Build user profiles given the historical user interactions
        :param iids: user selected item ids, list
        :param demographic_info: (gender, occupation), tuple
        :return:
        """
        self.base_iids = iids
        self.demographic_info = demographic_info
        # Build edges for new user
        self.new_user_nid = self.node_emb.weight.shape[0]
        new_user_gender_nid = self.data.e2nid[0]['gender'][demographic_info[0]]
        new_user_occ_nid = self.data.e2nid[0]['occ'][demographic_info[1]]
        row = [self.new_user_nid for i in range(len(iids) + 2)]
        col = iids + [new_user_gender_nid, new_user_occ_nid]
        self.new_edge_index = torch.from_numpy(np.array([row, col])).long().to(self.device_args['device'])

        # Build path begins and ends with
        path_from_new_user_np = utils.path.join(self.new_edge_index, self.data.edge_index)
        path_to_new_user_np = np.flip(path_from_new_user_np, axis=0)
        new_path_np = np.concatenate([path_from_new_user_np, path_to_new_user_np], axis=1)
        self.new_path = torch.from_numpy(new_path_np).long().to(self.device_args['device'])

        # Get new user embedding by applying message passing
        self.new_user_emb = torch.nn.Embedding(1, self.node_emb.weight.shape[1], max_norm=1, norm_type=2.0).weight
        new_node_emb = torch.cat((self.node_emb.weight, self.new_user_emb), dim=0)
        self.new_user_emb, _ = self.model.forward(new_node_emb, self.new_path)[-1, :]
        print('user building done...')

    def get_recommendations(self, seen_iids):
        # Estimate the feedback values and get the recommendation
<<<<<<< Updated upstream
        iids = self.get_top_n_popular_items(200).iid
        rec_iids = [iid for iid in iids if iid not in seen_iids]
        rec_iids = np.random.choice(rec_iids, 20)
        rec_nids = [self.data.e2nid[0]['iid'][iid] for iid in rec_iids]
        rec_item_emb = self.node_emb.weight[rec_nids]
        est_feedback = torch.sum(self.new_user_emb * rec_item_emb, dim=1).reshape(-1).cpu().detach().numpy()
        rec_iid_idx = np.argsort(est_feedback)[:self.num_recs]
        rec_iids = rec_iids[rec_iid_idx]

        df = self.data.items[0][self.data.items[0].iid.isin(rec_iids)]
        iids = [iid for iid in df.iids.value]
=======
        try:
            iids = self.get_top_n_popular_items(200).iid
            rec_iids = [iid for iid in iids if iid not in seen_iids]
            rec_iids = np.random.choice(rec_iids, 20)
            rec_nids = self.data.items[0].iloc[rec_iids].nid.values

            rec_item_emb = self.node_emb[rec_nids]

            est_feedback = torch.sum(self.new_user_emb * rec_item_emb, dim=1).reshape(-1).cpu().detach().numpy()
            rec_iid_idx = np.argsort(est_feedback)[:self.num_recs]
            rec_iids = rec_iids[rec_iid_idx]

            df = self.data.items[0][self.data.items[0].iid.isin(rec_iids)]
        except:
            import pdb
            pdb.set_trace()
>>>>>>> Stashed changes

        exp = [self.get_explanation(iid) for iid in iids]

        return df, exp

    def get_explanation(self, iid):
        movie_nid = self.data.iid2nid[iid]
        row = [self.new_user_nid, movie_nid]
        col = [movie_nid, self.new_user_emb]
        expl_edge_index = torch.from_numpy(np.array([row, col])).long().to(self.train_args['device'])
        new_edge_index = torch.cat([self.data.edge_index, self.new_edge_index], axis=1)
        expl_path_from_tuple = path.join(expl_edge_index, new_edge_index)
        exp_path_to_tuple = path.join(new_edge_index, expl_edge_index)
        exp_path = np.concatenate([expl_path_from_tuple, exp_path_to_tuple], axis=1)
        node_emb = torch.cat((self.model.node_emb.weight, self.new_user_emb), dim=0)
        self.new_user_emb, self.att_factor_dict = self.model.forward_(node_emb, exp_path)[-1, :]
        path = max(self.att_factor_dict.iteritems(), key=operator.itemgetter(1))[0]
        path_np = path.cpu().to_numpy()
        return str(path_np)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    default_poster_src = 'https://www.nehemiahmfg.com/wp-content/themes/dante/images/default-thumb.png'

    ########################## Define arguments ##########################
    # Dataset params
    parser.add_argument("--dataset", type=str, default='movielens', help="")
    parser.add_argument("--dataset_name", type=str, default='1m', help="")
    parser.add_argument("--num_core", type=int, default=10, help="")
    parser.add_argument("--step_length", type=int, default=2, help="")
    parser.add_argument("--train_ratio", type=float, default=False, help="")
    parser.add_argument("--debug", default=0.01, help="")

    # Model params
    parser.add_argument("--heads", type=int, default=4, help="")
    parser.add_argument("--dropout", type=float, default=0.6, help="")
    parser.add_argument("--emb_dim", type=int, default=64, help="")
    parser.add_argument("--repr_dim", type=int, default=16, help="")

    # Device params
    parser.add_argument("--device", type=str, default='cpu', help="")
    parser.add_argument("--gpu_idx", type=str, default='0', help="")

    args = parser.parse_args()

    # save id selected by users
    iid_list = []
    iid_list2 = []

    ########################## Define arguments ##########################
    data_folder, weights_folder, logger_folder = get_folder_path(args.dataset + args.dataset_name)

    ########################## Setup Device ##########################
    if not torch.cuda.is_available() or args.device == 'cpu':
        device = 'cpu'
    else:
        device = 'cuda:{}'.format(args.gpu_idx)

    ########################## Define parameters ##########################
    dataset_args = {
        'root': data_folder, 'dataset': args.dataset, 'name': args.dataset_name,
        'num_core': args.num_core, 'step_length': args.step_length, 'train_ratio': args.train_ratio,
        'debug': args.debug
    }
    model_args = {
        'heads': args.heads, 'emb_dim': args.emb_dim,
        'repr_dim': args.repr_dim, 'dropout': args.dropout
    }
    device_args = {'debug': args.debug, 'device': device, 'gpu_idx': args.gpu_idx}
    print('dataset params: {}'.format(dataset_args))
    print('task params: {}'.format(model_args))
    print('device_args params: {}'.format(device_args))

    recsys = PGATRecSys(num_recs=10, dataset_args=dataset_args, model_args=model_args, device_args=device_args)
    recsys.build_user(list(range(10)), ('M', 0))
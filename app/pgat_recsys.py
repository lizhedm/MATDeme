__model__ = 'PGAT'

import numpy as np
import pandas as pd
import torch
import os.path as osp
import operator

from torch_geometric.utils import path

from torch_geometric.datasets import MovieLens

from .pgat import PGATNetEx


class PGATRecSys(object):
    def __init__(self, num_recs, dataset_args, model_args, train_args):
        self.num_recs = num_recs
        self.train_args = train_args

        self.data = MovieLens(**dataset_args).data.to(train_args['device'])
        self.model = PGATNetEx(
                self.data.num_nodes[0],
                self.data.num_relations[0],
                **model_args
        ).to(train_args['device'])

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
        ratings_df = ratings_df.sort_index(axis=0, by='movie_count', ascending=False)

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
        self.new_user_nid = self.model.node_emb.weight.shape[0]
        new_user_gender_nid = self.data.gender2nid[0][demographic_info[0]]
        new_user_occ_nid = self.data.occ2nid[0][demographic_info[1]]
        row = [self.new_user_nid for i in range(len(iids) + 2)]
        col = iids + [new_user_gender_nid, new_user_occ_nid]
        self.new_edge_index = torch.from_numpy(np.array([row + col, col + row])).long().to(self.train_args['device'])

        # Build path begins and ends with
        path_from_new_user_np = path.join(self.new_edge_index, self.data.edge_index)
        path_to_new_user_np = path.join(self.data.edge_index, self.new_edge_index)
        new_path_np = np.concatenate([path_from_new_user_np, path_to_new_user_np], axis=1)
        self.new_path = torch.from_numpy(new_path_np).to(self.train_args['device'])
        # new_edge_index_df = pd.DataFrame({'head': new_edge_index_np[0, :], 'middle': new_edge_index_np[1, :]})
        # edge_index_np = self.data.edge_index.numpy()
        # edge_index_df = pd.DataFrame({'middle': edge_index_np[0, :], 'tail': edge_index_np[1, :]})
        # new_sec_order_edge_df = pd.merge(new_edge_index_df, edge_index_df, on='middle')
        # new_sec_order_edge_np = new_sec_order_edge_df.to_numpy()
        # new_sec_order_edge = torch.from_numpy(new_sec_order_edge_np).to(self.train_args['device']).t()

        # Get new user embedding by applying message passing
        path = torch.from_numpy(self.data.path[0]).to(self.train_args['device'])
        self.node_emb = self.model.forward_(self.model.node_emb.weight, path)
        self.new_user_emb = torch.nn.Embedding(1, self.model.node_emb.weight.shape[1]).weight
        node_emb = torch.cat((self.model.node_emb.weight, self.new_user_emb), dim=0)
        self.new_user_emb, self.att_factor = self.model.forward_(node_emb, self.new_path)[-1, :]
        print('user building done...')

    def get_recommendations(self, seen_iids):
        # Estimate the feedback values and get the recommendation
        iids = self.get_top_n_popular_items(200).iid
        rec_iids = [iid for iid in iids if iid not in seen_iids]
        rec_iids = np.random.choice(rec_iids, 20)
        rec_nids = [self.data.iid2nid[0][iid] for iid in rec_iids]
        rec_item_emb = self.node_emb[rec_nids]
        est_feedback = torch.sum(self.new_user_emb * rec_item_emb, dim=1).reshape(-1).cpu().detach().numpy()
        rec_iid_idx = np.argsort(est_feedback)[:self.num_recs]
        rec_iids = rec_iids[rec_iid_idx]

        df = self.data.items[0][self.data.items[0].iid.isin(rec_iids)]
        iids = [iid for iid in df.iids.value]

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

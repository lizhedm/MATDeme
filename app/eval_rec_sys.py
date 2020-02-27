import numpy as np
import torch
from torch.utils.data import DataLoader


def hit(hit_vec):
    if hit_vec.sum() > 0:
        return 1
    else:
        return 0


def ndcg(hit_vec):
    ndcg_vec = [np.reciprocal(np.log2(idx+2)) for idx, if_hit in enumerate(hit_vec.cpu().to_numpy()) if if_hit]
    return np.sum(ndcg_vec)


def metrics(model, dataset, train_args, rec_args):
    HR, NDCG, losses = [], [], []

    test_loader = DataLoader(dataset.data.test_user_pos_neg_pair, shuffle=True, batch_size=train_args['batch_size'])

    for u_nid, pos_i_nid, neg_i_nid in test_loader:
        propagated_node_emb = model(dataset.data)
        u_nid, pos_i_nid, neg_i_nid = u_nid.to(train_args['device']), pos_i_nid.to(train_args['device']), neg_i_nid.to(train_args['device'])
        u_node_emb, pos_i_node_emb, neg_i_node_emb = propagated_node_emb[u_nid], propagated_node_emb[pos_i_nid], \
                                                     propagated_node_emb[neg_i_nid]
        pred_pos = (u_node_emb * pos_i_node_emb).sum(dim=1)
        pred_neg = (u_node_emb * neg_i_node_emb).sum(dim=1)

        loss = - (pred_pos - pred_neg).sigmoid().log().sum().values

        _, indices = torch.topk(torch.cat([pred_pos, pred_neg], dim=0), rec_args['top_k'])
        hit_vec = indices < train_args['batch_size']

        HR.append(hit(hit_vec))
        NDCG.append(ndcg(hit_vec))
        losses.append(loss)

    return np.mean(HR), np.mean(NDCG). np.mean(losses)
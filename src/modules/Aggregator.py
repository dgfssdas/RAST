import torch.nn as nn
from modules.Hyperbolic import *
from torch_scatter import scatter_mean


class Aggregator(nn.Module):
    """
    Relational Path-aware Convolution Network
    """

    def __init__(self):
        super(Aggregator, self).__init__()

    def forward(self, ego_embed, edge_index, edge_type, relation_weight):
        head, tail = edge_index
        relation_type = edge_type
        n_entities = ego_embed.shape[0]

        head_emb = ego_embed[head]
        tail_emb = ego_embed[tail]
        relation_emb = relation_embed[relation_type]

        # hyperbolic
        # hyper_head_emb = expmap0(head_emb)
        # hyper_tail_emb = expmap(tail_emb, hyper_head_emb)
        # hyper_relation_emb = expmap(relation_emb, hyper_head_emb)
        # res = project(mobius_add(hyper_tail_emb, hyper_relation_emb))
        # res = logmap(res, hyper_head_emb)
        # entity_agg = scatter_mean(src=res, index=head, dim_size=n_entities, dim=0)

        # neigh_relation_emb = tail_emb * relation_emb
        # neigh_relation_emb_weight = self.calculate_sim_hrt(head_emb, tail_emb, relation_emb)
        # neigh_relation_emb_weight = neigh_relation_emb_weight.expand(neigh_relation_emb.shape[0],
        #                                                              neigh_relation_emb.shape[1])
        # # neigh_relation_emb_tmp = torch.matmul(neigh_relation_emb_weight, neigh_relation_emb)
        # neigh_relation_emb_weight = scatter_softmax(neigh_relation_emb_weight, index=head, dim=0)
        # neigh_relation_emb = torch.mul(neigh_relation_emb_weight, neigh_relation_emb)
        # entity_agg = scatter_sum(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0)

        # # euclidean
        # res1 = tail_emb + relation_emb
        # entity_agg = scatter_mean(src=res1, index=head, dim_size=n_entities, dim=0)

        return entity_agg

    # def calculate_sim_hrt(self, entity_emb_head, entity_emb_tail, relation_emb):
    #     tail_relation_emb = entity_emb_tail * relation_emb
    #     tail_relation_emb = tail_relation_emb.norm(dim=1, p=2, keepdim=True)
    #     head_relation_emb = entity_emb_head * relation_emb
    #     head_relation_emb = head_relation_emb.norm(dim=1, p=2, keepdim=True)
    #     att_weights = torch.matmul(head_relation_emb.unsqueeze(dim=1), tail_relation_emb.unsqueeze(dim=2)).squeeze(dim=-1)
    #     att_weights = att_weights ** 2
    #     return att_weights

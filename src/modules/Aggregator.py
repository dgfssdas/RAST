import torch.nn as nn
from modules.Hyperbolic import *
from torch_scatter import scatter_mean
from torch_scatter import scatter_softmax
from torch_scatter import scatter_sum
import torch.nn.functional as F


class Aggregator(nn.Module):
    """
    Relational Path-aware Convolution Network
    """

    def __init__(self):
        super(Aggregator, self).__init__()

    def forward(self, ego_embed, edge_index, edge_type, relation_weights):
        head, tail = edge_index
        relation_type = edge_type
        n_entities = ego_embed.shape[0]

        head_emb = ego_embed[head]
        tail_emb = ego_embed[tail]

        W_r = relation_weights[relation_type]
        h_T = head_emb.unsqueeze(1)
        transformed_head = torch.bmm(h_T, W_r).squeeze(1)
        attention_scores = F.leaky_relu((transformed_head * tail_emb).sum(dim=1))
        attention_scores_tail = torch.exp(attention_scores)
        attention_scores_sum = scatter_sum(src=attention_scores_tail, index=head, dim_size=n_entities, dim=0)[head]
        norm_attention_scores = attention_scores_tail / attention_scores_sum
        message = tail_emb
        weighted_messages = message * norm_attention_scores.unsqueeze(-1)  # [num_edges, embed_dim]
        aggregated_messages = scatter_mean(src=weighted_messages, index=head, dim=0)  # [num_nodes, embed_dim]

        return aggregated_messages

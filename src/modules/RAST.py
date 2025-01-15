import torch
import torch.nn as nn
from modules.GraphConv import GraphConv
import torch.nn.functional as F
from utils.util import L2_loss_mean
import numpy as np
import scipy.sparse as sp


class RAST(nn.Module):
    def __init__(self, args, data, device):
        super(RAST, self).__init__()
        self.device = device

        self.n_nodes = data.n_nodes
        self.n_physical_nodes = data.n_physical_nodes
        self.n_priori_nodes = data.n_priori_nodes

        self.embedding_dim = args.embedding_dim
        self.bilstm_hidden_size = args.bilstm_hidden_size
        self.bilstm_num_layers = args.bilstm_num_layers
        self.context_hops = args.context_hops
        self.lpe_dim = args.lpe_dim

        self.dropout = args.node_dropout
        self.dropout_rate = args.node_dropout_rate

        self.l2_lambda = args.l2_lambda
        self.temporal_lambda = args.temporal_lambda

        # KG relations
        self.kg_relation_weight = nn.Parameter(self.n_nodes, self.n_nodes)
        nn.init.xavier_uniform_(self.kg_relation_weight)

        # BiLSTM parameters
        self.bilstm_autoencoder = BiLSTMAutoencoder(
            input_dim=args.input_dim,
            hidden_dim=self.bilstm_hidden_size,
            num_layers=self.bilstm_num_layers,
            device=device
        )

        self.temporal_projector = nn.Linear(self.bilstm_hidden_size * 2, self.embedding_dim)
        self.gate_network = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.Sigmoid()
        )

        self.kg_edge_index, self.kg_edge_type = self.get_edges(data.kg_graph)

        self.kg_lpe = self.laplace_position_encoding(self.kg_edge_index, self.n_nodes, self.lpe_dim)
        self.random_init = torch.empty(self.n_nodes, self.embedding_dim - self.lpe_dim)
        nn.init.xavier_uniform_(self.random_init)
        self.kg_init = torch.cat([self.random_init, self.kg_lpe], dim=1)

        # Physical and prior embeddings
        self.physical_node_embedding = nn.Embedding.from_pretrained(self.kg_init[:self.n_priori_nodes], freeze=False)
        self.priori_node_embedding = nn.Embedding.from_pretrained(self.kg_init[self.n_priori_nodes:], freeze=False)

        self.gcn = GraphConv(
            embed_dim=self.embedding_dim,
            n_hops=self.context_hops,
            device=self.device,
            dropout_rate=self.dropout_rate
        )

        self.criterion = torch.nn.BCELoss()
        self.sigmoid = nn.Sigmoid()

        self.risk_projector = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )

    def get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))
        index = graph_tensor[:, :-1]
        type = graph_tensor[:, -1]
        return index.t().long().to(self.device), type.long().to(self.device)

    def forward(self, *input, mode):
        if mode == 'train':
            return self.calc_train(*input)
        if mode == 'eval':
            return self.calc_eval_score(*input)

    def calc_cf_embeddings(self, inputs, node_indexes, mode):

        """complete model"""
        ego_embed = torch.cat((self.physical_node_embedding.weight, self.priori_node_embedding.weight), dim=0).to(self.device)
        all_embeddings = self.gcn(ego_embed, self.kg_edge_index, self.kg_edge_type, self.kg_relation_weight, dropout=self.dropout)

        node_structural_emb = all_embeddings[node_indexes]

        decoded_output, node_temporal_emb = self.bilstm_autoencoder(inputs, mode=mode)

        temporal_reconstruction_loss = F.mse_loss(decoded_output, inputs)

        concatenated_features = torch.cat([node_structural_emb, node_temporal_emb], dim=-1)
        gate_signal = self.gate_network(concatenated_features)
        node_merge_emb = gate_signal * node_structural_emb + (1 - gate_signal) * node_temporal_emb

        return node_merge_emb, temporal_reconstruction_loss

    def calc_train(self, inputs, labels, node_indexes):

        node_embed, temporal_reconstruction_loss = self.calc_node_embeddings(inputs, node_indexes, mode="train")
        risk_feature = self.risk_projector(node_embed)
        risk_score = torch.sigmoid((risk_feature).sum(dim=-1)).squeeze()
        risk_loss = self.criterion(risk_score, labels)

        l2_loss = L2_loss_mean(node_embed)
        loss = risk_loss + self.l2_lambda * l2_loss + self.temporal_lambda * temporal_reconstruction_loss
        return loss

    def calc_eval_score(self, inputs, node_indexes):

        node_embed = self.calc_node_embeddings(inputs, node_indexes, mode="eval")
        risk_feature = self.risk_projector(node_embed)
        risk_score = torch.sigmoid((risk_feature).sum(dim=1)).squeeze()

        return risk_score

    def laplace_position_encoding(self, edge_index, num_nodes, k):

        adj = sp.coo_matrix((np.ones(len(edge_index[0])), (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes), dtype=np.float32)

        deg = np.array(adj.sum(axis=1)).flatten()
        deg_inv_sqrt = np.power(deg, -0.5).flatten()
        deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(deg_inv_sqrt)
        laplacian = sp.eye(num_nodes) - d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

        eigvals, eigvecs = sp.linalg.eigsh(laplacian, k=k, which='SM')
        return torch.tensor(eigvecs, dtype=torch.float32)


class BiLSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, device):
        super(BiLSTMAutoencoder, self).__init__()

        self.device = device
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.decoder = nn.LSTM(hidden_dim * 2, input_dim, num_layers, batch_first=True, bidirectional=True)

    def forward(self, x, mode="train"):

        batch_size, seq_len, input_dim = x.size()

        encoded_output, (hidden, cell) = self.encoder(x)
        decoder_input = torch.zeros(batch_size, seq_len, input_dim).to(self.device)
        decoded_output, _ = self.decoder(decoder_input, (hidden, cell))

        if mode == "train":
            return decoded_output, encoded_output.mean(dim=1)
        elif mode == "eval":
            return encoded_output.mean(dim=1)

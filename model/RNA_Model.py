import torch
import torch.nn as nn
from .module import MPNNLayer
from .feature import RNAFeatures
from torch_geometric.nn import LayerNorm

def gather_nodes(nodes, neighbor_idx):
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.device = args.device
        self.node_features = self.edge_features = args.hidden
        self.hidden_dim = args.hidden

        self.features = RNAFeatures(
            args.hidden, args.hidden,
            top_k=args.k_neighbors,
            dropout=args.dropout,
            node_feat_types=args.node_feat_types,
            edge_feat_types=args.edge_feat_types,
            args=args
        )

        layer = MPNNLayer
        # encoder
        self.W_s = nn.Embedding(args.vocab_size, self.hidden_dim)
        self.encoder_layers = nn.ModuleList([
            layer(self.hidden_dim, self.hidden_dim*2, dropout=args.dropout)
            for _ in range(args.num_encoder_layers)])
        self.mlp_skip_encoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                LayerNorm(self.hidden_dim)
            ) for _ in range(args.num_decoder_layers)
        ])
        # decoder
        self.decoder_layers = nn.ModuleList([
            layer(self.hidden_dim, self.hidden_dim*2, dropout=args.dropout)
            for _ in range(args.num_decoder_layers)])
        self.mlp_skip_decoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                LayerNorm(self.hidden_dim)
            ) for _ in range(args.num_decoder_layers)
        ])

        self.projection_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        )

        self.readout = nn.Linear(self.hidden_dim, args.vocab_size, bias=True)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, X, S, mask):
        X, S, h_V, h_E, E_idx, batch_id = self.features(X, S, mask)

        for i, (enc_layer, skip) in enumerate(zip(self.encoder_layers, self.mlp_skip_encoder)):
            h_EV = torch.cat([h_E, h_V[E_idx[0]], h_V[E_idx[1]]], dim=-1)
            h_res = enc_layer(h_V, h_EV, E_idx, batch_id)
            h_V = h_V + skip(h_res)

        for i, (dec_layer, skip) in enumerate(zip(self.decoder_layers, self.mlp_skip_decoder)):
            h_EV = torch.cat([h_E, h_V[E_idx[0]], h_V[E_idx[1]]], dim=-1)
            h_des = dec_layer(h_V, h_EV, E_idx, batch_id)
            h_V = h_V + skip(h_des)

        graph_embs = []
        for b_id in range(batch_id[-1].item()+1):
            b_data = h_V[batch_id == b_id].mean(0)
            graph_embs.append(b_data)
        graph_embs = torch.stack(graph_embs, dim=0)
        graph_prjs = self.projection_head(graph_embs)

        logits = self.readout(h_V)
        return logits, S, graph_prjs
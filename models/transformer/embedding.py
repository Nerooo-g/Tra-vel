from torch import nn
from models.transformer.gat import GAT
from models.transformer.gcn import GCN


class SpatialEmbedding(nn.Module):
    def __init__(self, layers, heads, d_model, drop_rate=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gat = GAT(num_of_layers=layers, num_heads_per_layer=heads, num_features_per_layer=d_model)
        # self.norm = LayerNorm(d_model)
        # self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, data):
        x = self.gat(data)
        # x = self.norm(x)
        return x


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, layers, drop_rate=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gcn = GCN(in_features=d_model, n_hid=d_model*2, n_hid2=d_model*2, out_features=d_model, dropout=0.3,
                       layers_num=layers)
        # self.norm = LayerNorm(d_model)

    def forward(self, x, adj):
        x = self.gcn(x, adj)
        return x


class STLayer(nn.Module):
    def __init__(self, layers, heads, s_dim,t_dim,drop_rate=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spatial_embedding = SpatialEmbedding(layers, heads, s_dim, drop_rate)
        self.temporal_embedding = TemporalEmbedding(t_dim, layers, drop_rate)

        # self.alpha = nn.Parameter(torch.tensor([1]))
        # self.beta = nn.Parameter(torch.tensor([1]))

    def forward(self,sp_data,t_emb, t_adj):
        sp = self.spatial_embedding(sp_data)
        tp = self.temporal_embedding(t_emb, t_adj)
        return sp, tp

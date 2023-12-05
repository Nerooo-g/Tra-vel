import torch
from torch import nn
from torch.cuda.amp import autocast

from models.transformer.decoder import Decoder
from models.transformer.embedding import STLayer
from models.transformer.encoder import Encoder
import torch.nn.functional as F


class Travel(nn.Module):
    """
    Transformer model from Attention is All You Need paper.

    Args:
        enc_size (int): Source class size.
        dec_size (int): Target class size.
        d_model (int): Embedding dimension size.
        ffn_hidden (int): Feedforward hidden layer size.
        n_head (int): Number of attention heads.
        n_layers (int): Number of encoder/decoder layers.
        drop_prob (float): Dropout probability.
        norm_type (str): Type of layer normalization.
        pe (str): Positional encoding type.
        tie_emb (bool): Tie encoder/decoder embeddings.
        pad_idx (int): Padding index.

    """

    def __init__(self, enc_size, dec_size, d_model, ffn_hidden, n_head, n_layers, sp_data, t_emb, t_adj, drop_prob
                 , norm_type='pre',
                 pe='relative', tie_emb=False, pad_idx=0, norm_bias=True, no_time=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.no_time_gap_module = no_time
        self.pad_idx = pad_idx
        self.sp_emb, self.sp_adj = sp_data
        self.sp_emb = nn.Embedding.from_pretrained(embeddings=self.sp_emb, freeze=False, padding_idx=0)
        self.t_emb = nn.Embedding.from_pretrained(embeddings=t_emb, freeze=False, padding_idx=0)
        self.t_adj = t_adj
        self.st_fc = nn.Linear(2 * d_model, d_model)
        if no_time:
            self.t_fc = nn.Linear(32, 64)
        else:
            self.t_fc = nn.Linear(1, 32)
        self.st_layer = STLayer(layers=1, heads=4, s_dim=128, t_dim=32, drop_rate=0.1)
        self.crds_fc = nn.Linear(2, 64)
        self.encoder = Encoder(enc_size=enc_size, d_model=d_model,
                               n_head=n_head, ffn_hidden=ffn_hidden, drop_prob=drop_prob,
                               n_layers=n_layers, norm_type=norm_type, pe=pe, tie_emb=tie_emb,
                               norm_bias=norm_bias
                               )

        self.decoder = Decoder(dec_size=dec_size, d_model=d_model, n_head=n_head, ffn_hidden=ffn_hidden,
                               drop_prob=drop_prob,
                               n_layers=n_layers, norm_type=norm_type, pe=pe, tie_emb=tie_emb, norm_bias=norm_bias)

    @autocast()
    def forward(self, src, crds, time_span, time_gaps, trg, t_trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        s_emb, t_emb = self.st_layer((self.sp_emb.weight, self.sp_adj.cuda()), self.t_emb.weight,
                                     self.t_adj.cuda())
        crds = self.crds_fc(crds)
        s_emb = s_emb[0]
        src = F.embedding(src, s_emb, padding_idx=0)
        src = torch.cat((src, crds), dim=-1)
        time_span = F.embedding(time_span, t_emb, padding_idx=0)
        time_gaps = self.t_fc(time_gaps.unsqueeze(-1))
        time_span = time_span.expand(-1, time_gaps.shape[1], -1)
        time_info = torch.cat((time_span, time_gaps), dim=2)
        src = self.st_fc(torch.cat((src, time_info), dim=-1))
        trg = F.embedding(trg, s_emb, padding_idx=0)
        t_trg = self.t_fc(t_trg.unsqueeze(-1))
        enc_hid = self.encoder(src, src_mask)
        output, t_output = self.decoder(trg, t_trg, enc_hid, trg_mask, src_mask)
        return output, t_output

    @autocast()
    def get_hidden_state(self, src, crds, time_span, time_gaps):
        src_mask = self.make_src_mask(src)
        s_emb, t_emb = self.st_layer((self.sp_emb.weight, self.sp_adj.cuda()), self.t_emb.weight,
                                     self.t_adj.cuda())
        crds = self.crds_fc(crds)
        s_emb = s_emb[0]
        src = F.embedding(src, s_emb, padding_idx=0)
        src = torch.cat((src, crds), dim=-1)
        time_span = F.embedding(time_span, t_emb, padding_idx=0)
        time_gaps = self.t_fc(time_gaps.unsqueeze(-1))
        time_span = time_span.expand(-1, time_gaps.shape[1], -1)
        time_info = torch.cat((time_span, time_gaps), dim=2)
        src = self.st_fc(torch.cat((src, time_info), dim=-1))
        enc_hid = self.encoder(src, src_mask)
        return enc_hid

    def make_src_mask(self, src):
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).cuda()
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

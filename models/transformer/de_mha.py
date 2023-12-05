import torch
from torch import nn


def make_table(batch_size, n_heads, length_q, length_k, max_relative_position):
    range_vec_q = torch.arange(length_q)
    range_vec_k = torch.arange(length_k)
    distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
    distance_mat_clipped = torch.clamp(distance_mat, max_relative_position, max_relative_position)
    final_mat = distance_mat_clipped + max_relative_position  # NxN
    final_mat = torch.LongTensor(final_mat).unsqueeze(0).unsqueeze(0).repeat(batch_size, n_heads, 1, 1).cuda()

    return final_mat


class RPEMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()

        self.hid_dim = d_model
        self.n_heads = n_head
        self.head_dim = d_model
        self.max_relative_position = 16

        self.fc_q = nn.Linear(d_model, d_model * n_head)
        self.fc_k = nn.Linear(d_model, d_model * n_head)
        self.fc_v = nn.Linear(d_model, d_model * n_head)

        self.fc_o = nn.Linear(d_model * n_head, d_model)
        self.k_r = nn.Linear(d_model, d_model)
        self.q_r = nn.Linear(d_model, d_model)
        self.p = nn.Parameter(torch.Tensor(self.max_relative_position * 2 + 1, d_model), requires_grad=True)

        self.scale = torch.sqrt(torch.FloatTensor([3 * self.head_dim]))

    def forward(self, query, key, value, mask=None):
        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]
        batch_size = query.shape[0]
        len_k = key.shape[1]
        len_q = query.shape[1]
        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)

        r_q1 = query.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        r_k1 = key.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        attn1 = torch.matmul(r_q1, r_k1.permute(0, 1, 3, 2))

        k_r = self.k_r(self.p)  # [max_relative_position * 2 + 1, d_model]
        table = torch.matmul(r_q1, k_r.transpose(0, 1))  # [batch_size, n_heads, len_q, max_relative_position * 2 + 1]
        p_q = make_table(batch_size, self.n_heads, len_q, len_k, self.max_relative_position)
        attn2 = table.gather(-1, p_q)  # [batch_size, n_heads, len_q, len_k]

        q_r = self.q_r(self.p)  # [max_relative_position * 2 + 1, d_model]
        table = torch.matmul(r_k1, q_r.transpose(0, 1))  # [batch_size, n_heads, len_k, max_relative_position * 2 + 1]
        p_r = make_table(batch_size, self.n_heads, len_k, len_q, self.max_relative_position)
        attn3 = table.gather(-1, p_r)  # [batch_size, n_heads, len_k, len_q]
        attn = (attn1 + attn2 + attn3) / self.scale.cuda()
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e4)

        attn = torch.softmax(attn, dim=-1)
        r_v1 = value.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        x = torch.matmul(attn, r_v1)
        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim * self.n_heads)

        # x = [batch size, query len, hid dim]

        x = self.fc_o(x)

        # x = [batch size, query len, hid dim]

        return x

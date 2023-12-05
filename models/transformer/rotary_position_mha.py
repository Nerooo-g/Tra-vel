import math

from torch import nn
from models.transformer.rotary_embedding_torch import RotaryEmbedding


class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention
    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self, dim, pe=True):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        if pe:
            self.rpe = RotaryEmbedding(dim)
        else:
            self.register_parameter('rpe', None)

    def forward(self, q, k, v, mask=None):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()
        if self.rpe is not None:
            q = self.rpe.rotate_queries_or_keys(q)
            k = self.rpe.rotate_queries_or_keys(k)
        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        # rpe = RotaryPE(d_tensor)
        # score = rpe.forward(query=q.permute(2, 3, 1, 0),key=k.permute(2, 3, 1, 0)) / math.sqrt(d_tensor)
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e4)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head, pe=True):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention(d_model, pe=pe)
        self.w_q = nn.Linear(d_model, d_model * n_head)
        self.w_k = nn.Linear(d_model, d_model * n_head)
        self.w_v = nn.Linear(d_model, d_model * n_head)
        self.w_concat = nn.Linear(d_model * n_head, d_model)

    def forward(self, query, key, value, mask=None):
        # 1. dot product with weight matrices
        query, key, value = self.w_q(query), self.w_k(key), self.w_v(value)

        # 2. split tensor by number of heads
        query, key, value = self.split(query), self.split(key), self.split(value)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(query, key, value, mask=mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        return out

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor

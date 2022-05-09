import torch
import torch.nn as nn
import numpy as np

def get_subsequent_mask(seq):
    seq_len = seq.shape[1]
    ones = torch.ones((seq_len, seq_len), dtype=torch.int32, device=seq.device)
    mask = 1 - torch.triu(ones, diagonal=1)

    return mask

class PositionalEncoding(nn.Module):
    def __init__(self, in_dim, out_dim, n_position=50):
        super(PositionalEncoding, self).__init__()

        self.linear = nn.Linear(in_features=in_dim, out_features=out_dim)
        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, out_dim))

    def _get_sinusoid_encoding_table(self, n_position, out_dim):
        ''' Sinusoid position encoding table '''
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / out_dim) for hid_j in range(out_dim)]
        
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        x = self.linear(x)
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, q, k, v, mask=None):
        d_k = q.shape[-1]

        # (B, n_head, T, out_dim) x (B, n_head, out_dim, T) -> (B, n_head, T, T)
        scores = torch.matmul(q / (d_k ** 0.5), k.transpose(2, 3))
        
        if mask is not None:
            # print(mask.unsqueeze(0).unsqueeze(0).shape, scores.shape)
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0)==0, -1e9)
        
        scores = torch.nn.Softmax(dim=-1)(scores) #(B, n_head, T, T)
        output = torch.matmul(scores, v) # (B, n_head, T, out_dim)
        
        return output, scores


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, in_dim, out_dim):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.out_dim = out_dim

        self.linear_q = nn.Linear(in_features=in_dim, out_features=n_head*out_dim)
        self.linear_k = nn.Linear(in_features=in_dim, out_features=n_head*out_dim)
        self.linear_v = nn.Linear(in_features=in_dim, out_features=n_head*out_dim)

        self.scaled_dot_production_attention = ScaledDotProductAttention()
        self.linear = nn.Linear(in_features=n_head*out_dim, out_features=out_dim) #这里out_features可以随意指定，这个就是encoder最终输出的qkv的维度，为了简便和out_dim一致

    def forward(self, q, k, v, mask=None):
        batch_size, len_q, len_kv = q.shape[0], q.shape[1], k.shape[1] #k和v的长度一直一致，但是在解码中，会出现q和kv长度不同的情况

        #(B, T, in_dim) -> (B, T, n_head * out_dim) -> (B, T, n_head, out_dim)
        q = self.linear_q(q).view(batch_size, len_q, self.n_head, self.out_dim)
        k = self.linear_k(k).view(batch_size, len_kv, self.n_head, self.out_dim)
        v = self.linear_v(v).view(batch_size, len_kv, self.n_head, self.out_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        output, scores = self.scaled_dot_production_attention(q, k, v, mask=mask)
        
        # (B, n_head, T, out_dim) -> (B, T, n_head, out_dim) -> (B, T, n_head * out_dim) -> (B, T, out_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        output = self.linear(output)

        return output, scores


class PositionWiseFeedForward(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(PositionWiseFeedForward, self).__init__()
        self.linear_1 = nn.Linear(in_features=in_dim, out_features=hidden_dim)
        self.linear_2 = nn.Linear(in_features=hidden_dim, out_features=in_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        
        return x


class Encoder(nn.Module):
    def __init__(self, n_head, in_dim, out_dim):
        super(Encoder, self).__init__()

        self.position_enc = PositionalEncoding(in_dim, out_dim)
        self.multi_head_attention_1 = MultiHeadAttention(n_head=n_head, in_dim=out_dim, out_dim=out_dim)
        self.layer_norm_1_1 = nn.LayerNorm(out_dim)

        self.position_wise_feed_forward_1 = PositionWiseFeedForward(out_dim, hidden_dim=128)
        self.layer_norm_1_2 = nn.LayerNorm(out_dim)
        
        self.scores_for_paint = None

    def forward(self, x):
        # (B, T, en_dim) -> (B, T, 64)
        qkv = self.position_enc(x)
        residual = qkv
        outputs, scores = self.multi_head_attention_1(qkv, qkv, qkv)
        self.scores_for_paint = scores.detach().cpu().numpy()
        outputs = self.layer_norm_1_1(outputs + residual)

        residual = outputs
        outputs = self.position_wise_feed_forward_1(outputs)
        outputs = self.layer_norm_1_2(outputs + residual)

        return outputs


class Decoder(nn.Module):
    def __init__(self, n_head,in_dim, out_dim):
        super(Decoder, self).__init__()

        self.position_enc = PositionalEncoding(in_dim, out_dim)
        self.multi_head_attention_1_1 = MultiHeadAttention(n_head=n_head, in_dim=out_dim, out_dim=out_dim)
        self.layer_norm_1_1 = torch.nn.LayerNorm(out_dim)

        self.multi_head_attention_1_2 = MultiHeadAttention(n_head=n_head, in_dim=out_dim, out_dim=out_dim)
        self.layer_norm_1_2 = torch.nn.LayerNorm(out_dim)

        self.position_wise_feed_forward_1 = PositionWiseFeedForward(out_dim, hidden_dim=128)
        self.layer_norm_1_3 = torch.nn.LayerNorm(out_dim)

        self.scores_for_paint = None

    def forward(self, enc_outputs, target):
        # (B, T, de_dim) -> (B, T, 64)
        qkv = self.position_enc(target)
        residual = qkv
        outputs, scores = self.multi_head_attention_1_1(qkv, qkv, qkv, mask=get_subsequent_mask(target))
        outputs = self.layer_norm_1_1(outputs + residual)

        residual = outputs
        outputs, scores = self.multi_head_attention_1_2(outputs, enc_outputs, enc_outputs)
        self.scores_for_paint = scores.detach().cpu().numpy()
        outputs = self.layer_norm_1_2(outputs + residual)

        residual = outputs
        outputs = self.position_wise_feed_forward_1(outputs)
        outputs = self.layer_norm_1_3(outputs + residual)

        return outputs


class Transformer(nn.Module):
    def __init__(self, n_head, en_dim, de_dim):
        super(Transformer, self).__init__()

        self.encoder = Encoder(n_head, in_dim=en_dim, out_dim=64)
        self.decoder = Decoder(n_head, in_dim=de_dim, out_dim=64)
        self.linear = nn.Linear(in_features=64, out_features=5)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        enc_outputs = self.encoder(x)
        outputs = self.decoder(enc_outputs, y)
        outputs = self.linear(outputs)
        outputs = self.softmax(outputs)

        return outputs

    def size(self):
        size = sum([p.numel() for p in self.parameters()])
        print('%.2fKB' % (size * 4 / 1024))


if __name__ == '__main__':
    model = Transformer(4, 20, 10)
    # (B, T, dim)
    batch_x = torch.randn(16, 10, 20)
    batch_y = torch.randn(16, 5, 10)
    pred = model(batch_x, batch_y)
    print(pred.shape)

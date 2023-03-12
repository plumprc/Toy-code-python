import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
from matplotlib import pyplot as plt

def get_subsequent_mask(seq):
    seq_len = seq.shape[1]
    ones = torch.ones((seq_len, seq_len), dtype=torch.int32, device=seq.device)
    mask = 1 - torch.triu(ones, diagonal=1)

    return mask


class PositionalEncoding(nn.Module):
    def __init__(self, in_dim, out_dim, n_position=50, pos=True):
        super(PositionalEncoding, self).__init__()

        self.linear = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.pos = pos
        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, out_dim))

    def _get_sinusoid_encoding_table(self, n_position, out_dim):
        ''' Sinusoid position encoding table '''
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / out_dim) for hid_j in range(out_dim)]
        
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        x = self.linear(x)

        return x if self.pos == False else x + self.pos_table[:, :x.size(1)].clone().detach()


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
        self.linear = nn.Linear(in_features=n_head*out_dim, out_features=out_dim)

    def forward(self, q, k, v, mask=None):
        batch_size, len_q, len_kv = q.shape[0], q.shape[1], k.shape[1]

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
    def __init__(self, n_head, in_dim, out_dim, pos):
        super(Encoder, self).__init__()

        self.position_enc = PositionalEncoding(in_dim, out_dim, pos=pos)
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
    def __init__(self, n_head,in_dim, out_dim, pos):
        super(Decoder, self).__init__()
        self.position_enc = PositionalEncoding(in_dim, out_dim, pos=pos)
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
    def __init__(self, n_head, en_dim, de_dim, out_features, pos):
        super(Transformer, self).__init__()

        self.encoder = Encoder(n_head, in_dim=en_dim, out_dim=64, pos=pos)
        self.decoder = Decoder(n_head, in_dim=de_dim, out_dim=64, pos=pos)
        self.linear = nn.Linear(in_features=64, out_features=out_features)
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        enc_outputs = self.encoder(x)
        outputs = self.decoder(enc_outputs, y)
        outputs = self.linear(outputs)
        # outputs = self.softmax(outputs)

        return outputs

    def size(self):
        size = sum([p.numel() for p in self.parameters()])
        print('%.2fKB' % (size * 4 / 1024))


class DataEmbedding(nn.Module):
    def __init__(self, in_dim=1, out_dim=4, mode='linear'):
        super(DataEmbedding, self).__init__()
        
        self.linear = nn.Linear(in_dim, out_dim)
        self.conv = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, padding=1)
        # self.norm = nn.BatchNorm1d(out_dim)
        self.activation = nn.ELU()
        self.mode = mode

    def forward(self, x):
        assert self.mode in ['linear', 'conv']

        if self.mode == 'linear':
            x = self.linear(x)
            # x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        
        else:
            x = self.conv(x.transpose(1, 2))
            # x = self.norm(x)
            x = x.transpose(1, 2)

        x = self.activation(x)
        return x


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    model = Transformer(n_head=4, en_dim=4, de_dim=4, out_features=1, pos=False)
    data_embd = DataEmbedding(1, 4, mode='linear')

    # x -> sin(alpha * x) + beta * x + outliers
    alpha = 200
    beta = 3
    base = np.linspace(-1000, 1000, 5000) / 1000
    outlier = np.random.normal(0, 0.01, 5000)
    x = torch.Tensor(base.reshape(250, 20)).unsqueeze(2)
    y = torch.Tensor((np.sin(alpha * base) + beta * base + outlier).reshape(250, 20)).unsqueeze(2)
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()

    data_set = Data.TensorDataset(x, y)
    data_loader = Data.DataLoader(
        dataset=data_set,
        batch_size=32,
        shuffle=True
    )

    model_optim = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(10):
        for step, (batch_x, batch_y) in enumerate(data_loader):
            x_embed = data_embd(batch_x)
            y_embed = data_embd(batch_y)

            model_optim.zero_grad()
            pred = model(x_embed, y_embed)
            loss = F.mse_loss(pred, batch_y)
            loss.backward()

            if step % 10 == 0:
                print('epoch:', epoch, 'step:', step, 'loss:', loss.item())
            
            model_optim.step()

    model.eval()
    x_embed = data_embd(x)
    y_embed = data_embd(y)
    pred = model(x_embed, y_embed)

    plt.plot(y.reshape(-1)[2000:3000])
    plt.plot(pred.detach().reshape(-1)[2000:3000])
    plt.show()

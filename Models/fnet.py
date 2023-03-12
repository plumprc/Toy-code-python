import torch
import torch.nn as nn

class FNetBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)

    def fourier_transform(self, x):
        return torch.fft.fft(x, dim=-1).real

    def forward(self, x):
        residual = x
        x = self.fourier_transform(x)
        x = self.norm_1(x + residual)
        residual = x
        x = self.mlp(x)
        out = self.norm_2(x + residual)

        return out


class FNet(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, e_layer=3):
        super().__init__()
        self.encoder = nn.ModuleList([
            FNetBlock(d_model, d_ff, dropout) for _ in range(e_layer)
        ])

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)

        return x
    

if __name__ == '__main__':
    x = torch.rand(32, 96, 7)
    model = FNet(7, 512)
    y = model(x)
    print(x.shape, y.shape)

import torch
import torch.nn as nn

class MLPBlock(nn.Module):
    def __init__(self, input_dim, mlp_dim=512) :
        super().__init__()
        self.fc1 = nn.Linear(input_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, input_dim)
    
    def forward(self,x):
        # [B, #tokens, D] or [B, D, #tokens]
        return self.fc2(self.gelu(self.fc1(x)))


class MixerBlock(nn.Module):
    def __init__(self, tokens_mlp_dim=16, channels_mlp_dim=1024, tokens_hidden_dim=32, channels_hidden_dim=1024):
        super().__init__()
        self.tokens_mlp_block = MLPBlock(tokens_mlp_dim, mlp_dim=tokens_hidden_dim)
        self.channels_mlp_block = MLPBlock(channels_mlp_dim, mlp_dim=channels_hidden_dim)
        self.norm = nn.LayerNorm(channels_mlp_dim)

    def forward(self,x):
        # token-mixing [B, D, #tokens]
        y = self.norm(x).transpose(1, 2)
        y = self.tokens_mlp_block(y)

        # channel-mixing [B, #tokens, D]
        y = y.transpose(1, 2) + x
        res = y
        y = self.norm(y)
        y = res + self.channels_mlp_block(y)

        return y


class MLPMixer(nn.Module):
    def __init__(self, num_classes, num_blocks, patch_size, tokens_hidden_dim, channels_hidden_dim, tokens_mlp_dim, channels_mlp_dim):
        super().__init__()
        self.embed = nn.Conv2d(3, channels_mlp_dim, kernel_size=patch_size, stride=patch_size) 
        self.mlp_blocks = nn.ModuleList([
            MixerBlock(tokens_mlp_dim, channels_mlp_dim, tokens_hidden_dim, channels_hidden_dim) for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.fc = nn.Linear(channels_mlp_dim, num_classes)

    def forward(self,x):
        # [B, C, H, W] -> [B, D, patch_h, patch_w] -> [B, #tokens, D]
        y = self.embed(x)
        B, D, _, _ = y.shape
        y = y.view(B, D, -1).transpose(1, 2)

        for block in self.mlp_blocks:
            y = block(y)
        
        y = self.norm(y)
        y = torch.mean(y, dim=1, keepdim=False) # [B, D]
        probs = self.fc(y) # [B, #class]

        return probs


if __name__ == '__main__':
    mlp_mixer = MLPMixer(num_classes=100, num_blocks=1, patch_size=16, tokens_hidden_dim=32, channels_hidden_dim=1024, tokens_mlp_dim=64, channels_mlp_dim=1024)
    img = torch.randn(32, 3, 128, 128)
    pred = mlp_mixer(img)
    print(pred.shape)
    
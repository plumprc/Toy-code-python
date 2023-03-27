import torch
import torch.nn as nn

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvMixer(nn.Module):
    def __init__(self, dim, depth, kernel_size=9, patch_size=16, n_classes=1000):
        super().__init__()
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )
        self.conv_mixer_blocks = nn.ModuleList([nn.Sequential(
            # Depthwise Convolution
            Residual(nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                nn.GELU(),
                nn.BatchNorm2d(dim)
            )),
            # Pointwise Convolution
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        ) for _ in range(depth)])
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(dim, n_classes)
        )

    def forward(self, x):
        x = self.patch_embed(x)
        for blk in self.conv_mixer_blocks:
            x = blk(x)

        x = self.projection(x)

        return x
    

if __name__ == '__main__':
    x = torch.rand(32, 3, 128, 128)
    model = ConvMixer(512, 3)
    y = model(x)
    print(y.shape)
    
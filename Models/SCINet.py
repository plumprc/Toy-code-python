import torch
import torch.nn as nn
import argparse

class SCIBlock(nn.Module):
    def __init__(self, in_planes, kernel_size=3, dilation=1, dropout=0.5, hidden_size=64):
        super(SCIBlock, self).__init__()        
        pad_l = dilation * (kernel_size - 1) // 2 + 1 if kernel_size % 2 != 0 else dilation * (kernel_size - 2) // 2 + 1
        pad_r = dilation * (kernel_size - 1) // 2 + 1 if kernel_size % 2 != 0 else dilation * (kernel_size) // 2 + 1

        self.phi = nn.Sequential(
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(in_planes, hidden_size, kernel_size=kernel_size, dilation=dilation),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, in_planes, kernel_size=3),
            nn.Tanh()
        )
        self.psi = nn.Sequential(
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(in_planes, hidden_size, kernel_size=kernel_size, dilation=dilation),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, in_planes, kernel_size=3),
            nn.Tanh()
        )
        self.rho = nn.Sequential(
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(in_planes, hidden_size, kernel_size=kernel_size, dilation=dilation),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, in_planes, kernel_size=3),
            nn.Tanh()
        )
        self.eta = nn.Sequential(
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(in_planes, hidden_size, kernel_size=kernel_size, dilation=dilation),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, in_planes, kernel_size=3),
            nn.Tanh()
        )

    def forward(self, x):
        x_even = x[:, ::2, :].transpose(1, 2)
        x_odd = x[:, 1::2, :].transpose(1, 2)

        x_odd_s = x_odd.mul(torch.exp(self.phi(x_even)))
        x_even_s = x_even.mul(torch.exp(self.psi(x_odd)))

        x_even_update = x_even_s + self.eta(x_odd_s)
        x_odd_update = x_odd_s - self.rho(x_even_s)

        return x_even_update.transpose(1, 2), x_odd_update.transpose(1, 2)


class SCITree(nn.Module):
    def __init__(self, level, in_planes, kernel_size=3, dilation=1, dropout=0.5, hidden_size=64):
        super(SCITree, self).__init__()
        self.level = level
        self.block = SCIBlock(
            in_planes=in_planes,
            kernel_size=kernel_size,
            dropout=dropout,
            dilation=dilation,
            hidden_size=hidden_size,
        )
        if level != 0:
            self.SCINet_odd = SCITree(level - 1, in_planes, kernel_size, dilation, dropout, hidden_size)
            self.SCINet_even = SCITree(level - 1, in_planes, kernel_size, dilation, dropout, hidden_size)
    
    def zip_up_the_pants(self, even, odd):
        assert even.shape[1] == odd.shape[1]

        even = even.transpose(0, 1)
        odd = odd.transpose(0, 1)
        merge = []

        for i in range(even.shape[0]):
            merge.append(even[i].unsqueeze(0))
            merge.append(odd[i].unsqueeze(0))

        return torch.cat(merge, 0).transpose(0, 1) # [B, L, D]
        
    def forward(self, x):
        # [B, L, D]
        x_even_update, x_odd_update = self.block(x)

        if self.level == 0:
            return self.zip_up_the_pants(x_even_update, x_odd_update)
        else:
            return self.zip_up_the_pants(self.SCINet_even(x_even_update), self.SCINet_odd(x_odd_update))


class SCINet(nn.Module):
    def __init__(self, input_len, output_len, level, in_planes, kernel_size=3, dilation=1, dropout=0.5, hidden_size=64):
        super(SCINet, self).__init__()
        self.encoder = SCITree(level, in_planes, kernel_size, dilation, dropout, hidden_size)
        self.projection = nn.Conv1d(input_len, output_len, kernel_size=1, stride=1, bias=False)
    
    def forward(self, x):
        res = x
        x = self.encoder(x)
        x += res
        x = self.projection(x)

        return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_len', type=int, default=96)
    parser.add_argument('--output_len', type=int, default=48)
    parser.add_argument('--level', type=int, default=3)
    parser.add_argument('--in_planes', type=int, default=8)
    parser.add_argument('--kernel_size', default=3, type=int, help='kernel size')
    parser.add_argument('--dilation', default=1, type=int, help='dilation')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--hidden_size', default=64, type=int, help='hidden channel of module')
    args = parser.parse_args()
    
    model = SCINet(args.input_len, args.output_len, args.level, args.in_planes, args.kernel_size, args.dilation, args.dropout, args.hidden_size)
    x = torch.randn(32, 96, 8)
    y = model(x)
    print(y.shape)

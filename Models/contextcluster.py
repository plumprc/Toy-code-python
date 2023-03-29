import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class PointReducer(nn.Module):
    def __init__(self, patch_size, padding, in_chans, embed_dim):
        super().__init__()

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, padding=padding)
        self.norm = nn.GroupNorm(1, embed_dim)

    def forward(self, x):
        # x: [B, C, W, H] -> [B, C, W / patch_size, H / patch_size]
        x = self.proj(x)
        x = self.norm(x)
        
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        x = self.fc1(x.permute(0, 2, 3, 1))
        x = self.act(x)
        x = self.fc2(x).permute(0, 3, 1, 2)

        return x


def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor):
    """
    return pair-wise similarity matrix between two tensors
    :param x1: [B, ..., M, D]
    :param x2: [B, ..., N, D]
    :return: similarity matrix [B, ..., M, N]
    """
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)

    return torch.matmul(x1, x2.transpose(-2, -1))


class Cluster(nn.Module):
    def __init__(self, dim, out_dim, proposal_w=2, proposal_h=2, fold_w=2, fold_h=2, heads=4, head_dim=24):
        super().__init__()

        self.heads = heads
        self.fold_w = fold_w
        self.fold_h = fold_h
        self.center_proj = nn.Conv2d(dim, heads * head_dim, kernel_size=1)
        self.patch_proj = nn.Conv2d(dim, heads * head_dim, kernel_size=1)
        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))
        self.centers_proposal = nn.AdaptiveAvgPool2d((proposal_w, proposal_h))
        self.proj = nn.Conv2d(heads * head_dim, out_dim, kernel_size=1)

    def forward(self, x):
        # [B, C, W, H] -> [B, head * head_dim, W, H]
        value = self.patch_proj(x)
        x = self.center_proj(x)
        # [B, head * head_dim, W, H] -> [B * head, head_dim, W, H]
        value = rearrange(value, "b (e c) w h -> (b e) c w h", e=self.heads)
        x = rearrange(x, "b (e c) w h -> (b e) c w h", e=self.heads)

        if self.fold_w > 1 and self.fold_h > 1:
            # split feature maps into local regions for clustering
            _, _, w0, h0 = x.shape
            assert w0 % self.fold_w == 0 and h0 % self.fold_h == 0
            # [B * head, head_dim, W, H] -> [B * head * (fold_w * fold_h), head_dim, W / fold_w, H / fold_h]
            value = rearrange(value, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w, f2=self.fold_h)
            x = rearrange(x, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w, f2=self.fold_h)

        b, c, w, _ = x.shape
        # [b, c, w, h] -> [b, c, cluster_w, cluster_h] -> [b, #cluster, c]
        centers = self.centers_proposal(x)
        value_centers = rearrange(self.centers_proposal(value), 'b c w h -> b (w h) c')
        # [b, #cluster, #patch], similarity between centers_proposal and points in local region
        sim = torch.sigmoid(
            self.sim_beta +
            self.sim_alpha * pairwise_cos_sim(
                centers.reshape(b, c, -1).permute(0, 2, 1),
                x.reshape(b, c, -1).permute(0, 2, 1)
            )
        )
        # assign each point to one center, similar as NMS in detection
        _, sim_max_idx = sim.max(dim=1, keepdim=True)
        mask = torch.zeros_like(sim)
        mask.scatter_(1, sim_max_idx, 1.)
        sim = sim * mask
        # [b, c, w, h] -> [b, #patch, c]
        value = rearrange(value, 'b c w h -> b (w h) c')

        # feature aggregating in eq.(1), [b, #cluster, c]
        cluster_update = ((value.unsqueeze(dim=1) * sim.unsqueeze(dim=-1)).sum(dim=2) + value_centers) / (
                sim.sum(dim=-1, keepdim=True) + 1.0)

        # feature dispatching in eq.(2), [b, #patch, c] -> [b, c, w, h]
        patches = (cluster_update.unsqueeze(dim=2) * sim.unsqueeze(dim=-1)).sum(dim=1)
        patches = rearrange(patches, "b (w h) c -> b c w h", w=w)
        
        # [[B * head * (fold_w * fold_h), head_dim, W / fold_w, H / fold_h]] -> [B * head, head_dim, W, H]
        if self.fold_w > 1 and self.fold_h > 1:
            # recover the splited regions back to feature maps
            patches = rearrange(patches, "(b f1 f2) c w h -> b c (f1 w) (f2 h)", f1=self.fold_w, f2=self.fold_h)
        
        # [B * head, head_dim, W, H] -> [B, head * head_dim, W, H] -> [B, C, W, H]
        patches = rearrange(patches, "(b e) c w h -> b (e c) w h", e=self.heads)
        patches = self.proj(patches)

        return patches


class ClusterBlock(nn.Module):
    def __init__(self, dim, mlp_ratio, proposal_w, proposal_h, fold_w, fold_h, heads, head_dim):

        super().__init__()

        self.token_mixer = Cluster(dim, dim, proposal_w, proposal_h, fold_w, fold_h, heads, head_dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.channel_mixer = MLP(dim, int(dim * mlp_ratio))
        self.norm2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        x = x + self.token_mixer(self.norm1(x))
        x = x + self.channel_mixer(self.norm2(x))
        
        return x


class ContextCluster(nn.Module):
    def __init__(self, layers=3, pad=0, num_classes=1000, 
            # see the original paper for recommended parameters
            embed_dims=[5, 32, 64, 128], mlp_ratios=[2, 2, 2], proposal_w=[2, 2, 2], proposal_h=[2, 2, 2], 
            fold_w=[4, 2, 1], fold_h=[4, 2, 1], heads=[2, 2, 4], head_dim=[16, 32, 32],
        ):

        super().__init__()

        self.cocs_blocks = nn.ModuleList([
            nn.Sequential(
                PointReducer(
                    patch_size=2 if idx != 0 else 4,
                    padding=pad,
                    in_chans=embed_dims[idx], embed_dim=embed_dims[idx + 1]
                ),
                ClusterBlock(
                    embed_dims[idx + 1], mlp_ratios[idx], proposal_w=proposal_w[idx], proposal_h=proposal_h[idx], 
                    fold_w=fold_w[idx], fold_h=fold_h[idx], heads=heads[idx], head_dim=head_dim[idx]
                )
            ) for idx in range(layers)
        ])

        # Classifier head
        self.norm = nn.BatchNorm2d(embed_dims[-1])
        self.projection = nn.Linear(embed_dims[-1], num_classes)

    def img2points(self, x):
        _, _, img_w, img_h = x.shape
        # register positional information buffer.
        range_w = torch.arange(0, img_w, step=1) / (img_w - 1.0)
        range_h = torch.arange(0, img_h, step=1) / (img_h - 1.0)
        # fea_pos: [B, W, H, 2]
        fea_pos = torch.stack(torch.meshgrid(range_w, range_h, indexing='ij'), dim=-1).float()
        fea_pos = fea_pos.to(x.device)
        fea_pos = fea_pos - 0.5
        # pos: [B, 2, W, H]
        pos = fea_pos.permute(2, 0, 1).unsqueeze(dim=0).expand(x.shape[0], -1, -1, -1)

        return pos

    def forward(self, x):
        # x: [B, 3, W, H] -> [B, 5, W, H]; color + pos
        x = torch.cat([x, self.img2points(x)], dim=1)
        for blk in self.cocs_blocks:
            x = blk(x)

        x = self.norm(x)
        cls_out = self.projection(x.mean([-2, -1]))

        return cls_out
    

if __name__ == '__main__':
    x = torch.rand(4, 3, 128, 128)
    model = ContextCluster(3)
    y = model(x)
    print(y.shape)

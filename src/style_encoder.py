import torch
import torch.nn as nn
import torch.nn.functional as F

class AlphaPredictor(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, in_ch // 4, 1), nn.ReLU(True),
            nn.Conv2d(in_ch // 4, 1, 1), nn.Sigmoid()
        )

    def forward(self, feat, sim):
        a = self.net(feat)  # (B,1,1,1)
        return a * (0.5 + 0.5 * sim)

class TinyStyleEncoder(nn.Module):

    def __init__(self, in_ch=3, dim=64):
        super().__init__()

        def dw_block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cin, 3, 2, 1, groups=cin, bias=False),
                nn.BatchNorm2d(cin), nn.ReLU(inplace=True),
                nn.Conv2d(cin, cout, 1, 1, 0, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True)
            )

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
        )
        self.b1 = dw_block(16, 32)
        self.b2 = dw_block(32, 64)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(64, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x_ref):
        x = F.interpolate(x_ref, size=(64, 64), mode='bilinear', align_corners=False)
        x = self.stem(x)
        x = self.b1(x)
        x = self.b2(x)
        s = self.pool(x).flatten(1)
        s = self.proj(s)
        return self.norm(s)

class SharedFiLM(nn.Module):

    def __init__(self, style_dim=64, base_c=128):
        super().__init__()
        hid = max(64, style_dim)
        self.mlp = nn.Sequential(
            nn.Linear(style_dim, hid), nn.ReLU(inplace=True),
            nn.Linear(hid, base_c * 2)
        )

        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, s):
        gb = self.mlp(s)
        gamma, beta = gb.chunk(2, dim=1)
        return gamma, beta, self.alpha.clamp(0, 1)

class LightFiLMInject(nn.Module):

    def __init__(self, base_c=128, out_ch=128):
        super().__init__()
        self.map_g = nn.Linear(base_c, out_ch)
        self.map_b = nn.Linear(base_c, out_ch)

    def forward(self, x, gamma_base, beta_base, alpha, gate=None, scale=1.0):
        B, C, H, W = x.shape
        g = self.map_g(gamma_base).view(B, C, 1, 1)
        b = self.map_b(beta_base).view(B, C, 1, 1)

        if gate is None:
            gate = 1.0

        if isinstance(scale, torch.Tensor):
            scale = scale.view(B, 1, 1, 1)

        return x + (alpha * scale) * gate * (x * (1 + g) + b - x)

class AlphaPredictor(nn.Module):

    def __init__(self, in_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, in_ch // 4, 1), nn.ReLU(True),
            nn.Conv2d(in_ch // 4, 1, 1), nn.Sigmoid()  
        )

    def forward(self, feat, sim):
        # sim (B,1,1,1)
        a = self.net(feat)
        return a * (0.5 + 0.5 * sim)  

class ReliabilityPredictor(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        in_dim = 13
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1), nn.Sigmoid()
        )

    @staticmethod
    def _global_stats(img):
        mu = img.mean(dim=(2, 3))               
        std = img.std(dim=(2, 3), unbiased=False) 
        return mu, std

    def forward(self, img, ref, sim):

        if sim.dim() == 4:
            sim_flat = sim.view(sim.size(0), 1)
        else:
            sim_flat = sim

        mu_x, std_x = self._global_stats(img)
        mu_r, std_r = self._global_stats(ref)

        feat = torch.cat([mu_x, std_x, mu_r, std_r, sim_flat], dim=1)  # (B,13)
        rel = self.mlp(feat)  # (B,1)
        return rel.view(-1, 1, 1, 1)


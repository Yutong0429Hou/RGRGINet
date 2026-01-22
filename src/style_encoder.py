import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Tiny Style Encoder
# ============================================================
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
    """
    提取参考图风格特征的轻量编码器。
    输出一个风格向量 s ∈ R^{B×dim}。
    """
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
        # 将输入图像缩放到 64×64，降低计算量
        x = F.interpolate(x_ref, size=(64, 64), mode='bilinear', align_corners=False)
        x = self.stem(x)
        x = self.b1(x)
        x = self.b2(x)
        s = self.pool(x).flatten(1)
        s = self.proj(s)
        return self.norm(s)


# ============================================================
# Shared FiLM 调制器
# ============================================================
class SharedFiLM(nn.Module):
    """
    根据风格向量生成 (γ, β, α) 参数。
    α 是一个可学习的全局门控（初值 0.1，可调节整体风格强度）。
    """
    def __init__(self, style_dim=64, base_c=128):
        super().__init__()
        hid = max(64, style_dim)
        self.mlp = nn.Sequential(
            nn.Linear(style_dim, hid), nn.ReLU(inplace=True),
            nn.Linear(hid, base_c * 2)
        )
        # α 可学习，约束在 [0,1]
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, s):
        gb = self.mlp(s)
        gamma, beta = gb.chunk(2, dim=1)
        return gamma, beta, self.alpha.clamp(0, 1)


# ============================================================
# LightFiLMInject —— 支持 gate / scale 的增强版
# ============================================================
class LightFiLMInject(nn.Module):
    """
    轻量 FiLM 风格注入模块。
    - x: 主干特征 (B,C,H,W)
    - gamma_base / beta_base: 来自 SharedFiLM
    - alpha: 全局学习门控 (标量)
    - gate:  空间门控 (B,1,H,W)，控制注入区域 (通常来自 mask)
    - scale: 相似度权重 (B,1,1,1)，控制风格强度
    """
    def __init__(self, base_c=128, out_ch=128):
        super().__init__()
        self.map_g = nn.Linear(base_c, out_ch)
        self.map_b = nn.Linear(base_c, out_ch)

    def forward(self, x, gamma_base, beta_base, alpha, gate=None, scale=1.0):
        B, C, H, W = x.shape
        g = self.map_g(gamma_base).view(B, C, 1, 1)
        b = self.map_b(beta_base).view(B, C, 1, 1)

        # 若未提供 gate，则全区域注入
        if gate is None:
            gate = 1.0

        # 若 scale 是张量则自动 broadcast
        if isinstance(scale, torch.Tensor):
            scale = scale.view(B, 1, 1, 1)

        # 逐层自适应：alpha 可学习 + gate 空间控制 + scale 相似度控制
        return x + (alpha * scale) * gate * (x * (1 + g) + b - x)


# ============================================================
# 可选: AlphaPredictor（逐层自适应α预测器）
# ============================================================
class AlphaPredictor(nn.Module):
    """
    根据当前层特征 + 全局相似度预测逐层注入强度 α。
    可在 networks.py 中调用，用于替代固定 α。
    """
    def __init__(self, in_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, in_ch // 4, 1), nn.ReLU(True),
            nn.Conv2d(in_ch // 4, 1, 1), nn.Sigmoid()  # 输出 (B,1,1,1)
        )

    def forward(self, feat, sim):
        # sim (B,1,1,1)
        a = self.net(feat)
        return a * (0.5 + 0.5 * sim)  # 放大或抑制，范围 (0,1)


class ReliabilityPredictor(nn.Module):
    """
    输出参考可靠度 rel ∈ (0,1)，用于替代 heuristic sim 做所有 gating。
    只用非常便宜的全局统计：mean/std + sim。
    """
    def __init__(self, hidden=32):
        super().__init__()
        # 输入维度：mu_x(3)+std_x(3)+mu_r(3)+std_r(3)+sim(1)=13
        in_dim = 13
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1), nn.Sigmoid()
        )

    @staticmethod
    def _global_stats(img):
        # img: (B,3,H,W), 假设范围已在[0,1]或[-1,1]，这里不强行改范围
        mu = img.mean(dim=(2, 3))                # (B,3)
        std = img.std(dim=(2, 3), unbiased=False) # (B,3)
        return mu, std

    def forward(self, img, ref, sim):
        """
        sim: (B,1,1,1) 或 (B,1)
        return: rel (B,1,1,1)
        """
        if sim.dim() == 4:
            sim_flat = sim.view(sim.size(0), 1)
        else:
            sim_flat = sim

        mu_x, std_x = self._global_stats(img)
        mu_r, std_r = self._global_stats(ref)

        feat = torch.cat([mu_x, std_x, mu_r, std_r, sim_flat], dim=1)  # (B,13)
        rel = self.mlp(feat)  # (B,1)
        return rel.view(-1, 1, 1, 1)

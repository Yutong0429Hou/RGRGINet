# src/modules/edffn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class EDFFN(nn.Module):
    """
    Enhanced Dual-FFN with learnable Spectral Modulation.
    放置位置：SandwichBlock 的 FFN 处（建议替换第二个 FFN）。
    特性：
      - Pre-LN + 残差
      - 自动 pad 到 patch_size 的倍数再还原
      - AMP 兼容（torch.fft.*）
    """
    def __init__(self, dim, patch_size=8, ffn_expansion_factor=4, bias=True, drop=0.0):
        super().__init__()
        self.dim = dim
        self.patch_size = int(patch_size)
        self.hidden = int(dim * ffn_expansion_factor)

        # Pre-norm
        self.ln = nn.LayerNorm(dim)

        # 1x1 → DWConv → Gated → 1x1
        self.project_in  = nn.Conv2d(dim, self.hidden * 2, kernel_size=1, bias=bias)
        self.dwconv      = nn.Conv2d(self.hidden * 2, self.hidden * 2, kernel_size=3, stride=1, padding=1,
                                     groups=self.hidden * 2, bias=bias)
        self.project_out = nn.Conv2d(self.hidden, dim, kernel_size=1, bias=bias)

        # 频域增益（只调幅值，数值更稳）
        P = self.patch_size
        self.spectral_gain = nn.Parameter(torch.ones(dim, 1, 1, P, P // 2 + 1))

        self.drop = nn.Dropout(drop) if drop and drop > 1e-6 else nn.Identity()

        self.current_sim = None  # 由外部网络（HINT）在 forward 前设置

    @staticmethod
    def _pad_to_multiple(x, multiple):
        H, W = x.shape[-2:]
        pad_h = (multiple - H % multiple) % multiple
        pad_w = (multiple - W % multiple) % multiple
        if pad_h == 0 and pad_w == 0:
            return x, (0, 0, H, W)
        x = F.pad(x, (0, pad_w, 0, pad_h))  # (left,right,top,bottom)
        return x, (pad_h, pad_w, H, W)

    @staticmethod
    def _unpad(x, pad_info):
        pad_h, pad_w, H, W = pad_info
        if pad_h == 0 and pad_w == 0:
            return x
        return x[..., :H, :W]

    def forward(self, x):
        """
        x: (B, C=dim, H, W)
        """
        identity = x

        # LayerNorm 作用在通道维：临时换轴再换回
        x_ln = self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()

        # —— 空域 FFN（Conv-Gated-Conv）——
        y = self.project_in(x_ln)
        y1, y2 = self.dwconv(y).chunk(2, dim=1)
        y = F.gelu(y1) * y2
        y = self.project_out(y)

        # —— 频域增强（按 patch rFFT/irFFT）——
        # y_pad, info = self._pad_to_multiple(y, self.patch_size)
        # P = self.patch_size
        # y_blk = rearrange(y_pad, 'b c (h p1) (w p2) -> b c h w p1 p2', p1=P, p2=P)
        # y_fft = torch.fft.rfft2(y_blk.float())  # AMP 兼容
        #
        # gain = F.softplus(self.spectral_gain)   # 保证正增益
        # y_fft = y_fft * gain                    # 广播到 (b,c,h,w,p1,p2//2+1)
        #
        # y_blk = torch.fft.irfft2(y_fft, s=(P, P))
        y_pad, info = self._pad_to_multiple(y, self.patch_size)
        P = self.patch_size
        y_blk = rearrange(y_pad, 'b c (h p1) (w p2) -> b c h w p1 p2', p1=P, p2=P)
        y_fft = torch.fft.rfft2(y_blk.float())  # AMP 兼容

        # 频域增益
        gain = F.softplus(self.spectral_gain)   # (C,1,1,P,P//2+1)

        # ===== NEW: 根据 current_sim 调节频域增强强度 =====
        # current_sim 预期形状为 (B,1,1,1)，数值在 [0,1]
        if getattr(self, "current_sim", None) is not None:
            sim = self.current_sim  # (B,1,1,1)
            # 把相似度映射到 [0.5, 1.5] 这样的缩放，用于调节增强强度
            # sim_factor = 0.5 + sim  # (B,1,1,1)，相似度高 → >1
            sim_factor = 1.0 + 0.8 * sim

            # 扩展维度到和 y_fft 一样多，用于广播
            while sim_factor.dim() < y_fft.dim():
                sim_factor = sim_factor.unsqueeze(-1)
            # 直接在频谱上按 batch 进行缩放
            y_fft = y_fft * sim_factor

        # 再乘通道维的 learnable gain
        y_fft = y_fft * gain                    # 广播到 (b,c,h,w,p1,p2//2+1)

        y_blk = torch.fft.irfft2(y_fft, s=(P, P))

        y_spec = rearrange(y_blk, 'b c h w p1 p2 -> b c (h p1) (w p2)', p1=P, p2=P)
        y_spec = self._unpad(y_spec, info)

        # 残差
        out = identity + self.drop(y_spec)
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class EDFFN(nn.Module):
 
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

        P = self.patch_size
        self.spectral_gain = nn.Parameter(torch.ones(dim, 1, 1, P, P // 2 + 1))

        self.drop = nn.Dropout(drop) if drop and drop > 1e-6 else nn.Identity()

        self.current_sim = None 

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


        x_ln = self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


        y = self.project_in(x_ln)
        y1, y2 = self.dwconv(y).chunk(2, dim=1)
        y = F.gelu(y1) * y2
        y = self.project_out(y)

        y_pad, info = self._pad_to_multiple(y, self.patch_size)
        P = self.patch_size
        y_blk = rearrange(y_pad, 'b c (h p1) (w p2) -> b c h w p1 p2', p1=P, p2=P)
        y_fft = torch.fft.rfft2(y_blk.float()) 


        gain = F.softplus(self.spectral_gain)  

        if getattr(self, "current_sim", None) is not None:
            sim = self.current_sim  # (B,1,1,1)

            sim_factor = 1.0 + 0.8 * sim

            while sim_factor.dim() < y_fft.dim():
                sim_factor = sim_factor.unsqueeze(-1)
            y_fft = y_fft * sim_factor

        y_fft = y_fft * gain                

        y_blk = torch.fft.irfft2(y_fft, s=(P, P))

        y_spec = rearrange(y_blk, 'b c h w p1 p2 -> b c (h p1) (w p2)', p1=P, p2=P)
        y_spec = self._unpad(y_spec, info)

        out = identity + self.drop(y_spec)
        return out


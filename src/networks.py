import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from einops import rearrange
from .style_encoder import AlphaPredictor
from .style_encoder import TinyStyleEncoder, SharedFiLM, LightFiLMInject
from .edffn import EDFFN 
from .style_encoder import ReliabilityPredictor

class TextureRefineBlock(nn.Module):
    def __init__(self, in_ch, num_layers=3):
        super().__init__()
        layers = []
        ch = in_ch
        for i in range(num_layers):
            layers.append(nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True))
            if i != num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.body(x)


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)
    return module


class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # Spatial branch
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True),
            LayerNorm(dim, 'WithBias'),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True),
            LayerNorm(dim, 'WithBias'),
            nn.ReLU(inplace=True)
        )
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        # Channel branch (MHSA)
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # Spatial branch
        y = self.avg_pool(x)
        y = self.conv(y)
        y = self.upsample(y)
        out = y * out

        out = self.project_out(out)
        return out

class SandwichBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        ffn_expansion_factor,
        bias,
        LayerNorm_type,
        use_edffn: bool = False,   
        edffn_patch: int = 8,       
        edffn_drop: float = 0.0   
    ):
        super(SandwichBlock, self).__init__()

        self.norm1_1 = LayerNorm(dim, LayerNorm_type)
        self.ffn1 = FeedForward(dim, ffn_expansion_factor, bias)

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)

        self.norm2 = LayerNorm(dim, LayerNorm_type)

        self.use_edffn = use_edffn
        if self.use_edffn:
            self.ffn2 = EDFFN(dim=dim, patch_size=edffn_patch,
                              ffn_expansion_factor=ffn_expansion_factor,
                              bias=bias, drop=edffn_drop)
        else:
            self.ffn2 = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.ffn1(self.norm1_1(x))
        x = x + self.attn(self.norm1(x))
        if self.use_edffn:
            x = self.ffn2(x)
        else:
            x = x + self.ffn2(self.norm2(x))
        return x

class GatedEmb(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(GatedEmb, self).__init__()
        self.gproj1 = nn.Conv2d(in_c, embed_dim * 2, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.gproj1(x)
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return x
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )
        self.body2 = nn.PixelUnshuffle(2)
        self.proj = nn.Conv2d(n_feat * 4, n_feat * 2, kernel_size=3, stride=1, padding=1, groups=n_feat * 2, bias=False)

    def forward(self, x, mask):
        out = self.body(x)
        out_mask = self.body2(mask)

        b, n, h, w = out.shape
        device = out.device
        t = torch.zeros((b, 2 * n, h, w), device=device)

        for i in range(n):
            t[:, 2 * i, :, :] = out[:, i, :, :]
        for i in range(n):
            if i <= 3:
                t[:, 2 * i + 1, :, :] = out_mask[:, i, :, :]
            else:
                t[:, 2 * i + 1, :, :] = out_mask[:, (i % 4), :, :]

        return self.proj(t)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x, mask):
        return self.body(x)
        
def _cfg_get(cfg, key, default):
    if cfg is None:
        return default
    v = getattr(cfg, key, default)
    return default if v is None else v

def _cfg_bool(cfg, key, default=False):
    v = _cfg_get(cfg, key, default)
    if isinstance(v, str):
        return v.lower() in ("1", "true", "yes", "on")
    return bool(v)

def _cfg_int(cfg, key, default=0):
    v = _cfg_get(cfg, key, default)
    try:
        return int(v)
    except Exception:
        return int(default)

def _cfg_float(cfg, key, default=0.0):
    v = _cfg_get(cfg, key, default)
    try:
        return float(v)
    except Exception:
        return float(default)

def build_lowpass_mask(h, w, ratio=0.25, device='cpu', dtype=torch.float32):
    yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    dist = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r = ratio * (min(h, w) / 2.0)
    mask = (dist <= r).to(dtype)
    return mask  # (H,W)


def lowpass_filter(x, ratio=0.25):
    B, C, H, W = x.shape
    X = torch.fft.fft2(x, dim=(-2, -1))
    X = torch.fft.fftshift(X, dim=(-2, -1))
    mask = build_lowpass_mask(H, W, ratio=ratio, device=x.device, dtype=x.dtype)  # (H,W)
    X = X * mask[None, None, :, :]
    X = torch.fft.ifftshift(X, dim=(-2, -1))
    x_low = torch.fft.ifft2(X, dim=(-2, -1)).real
    return x_low

class RGRG(nn.Module):
    def __init__(self,
                 inp_channels=4,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 config=None):
        super(RGRG, self).__init__()

        # ===== Style-conditioning =====
        use_style = _cfg_bool(config, "USE_STYLE", False)
        if use_style:
            style_dim = _cfg_int(config, "STYLE_DIM", 64)
            self.style_encoder = TinyStyleEncoder(in_ch=3, dim=style_dim)
            self.shared_film = SharedFiLM(style_dim=style_dim, base_c=128)
            self.style_inject = LightFiLMInject(base_c=128, out_ch=int(dim * 2))
        else:
            self.style_encoder = None

        self.sim_tau_low = _cfg_float(config, "STYLE_SIM_TAU_LOW", 0.4)    
        self.sim_min = _cfg_float(config, "STYLE_MIN_SIM", 0.2)              
        self.gate_smooth = int(_cfg_int(config, "STYLE_GATE_SMOOTH", 3))     
        self.lowfreq_ratio = _cfg_float(config, "STYLE_LOWFREQ_RATIO", 0.25) 

        self.lm_r_min = _cfg_float(config, "LM_R_MIN", 0.1)
        self.lm_r_max = _cfg_float(config, "LM_R_MAX", 0.8)
        self.lm_gamma = _cfg_float(config, "LM_GAMMA", 1.5)

        self.lm_alpha = nn.Parameter(torch.tensor(5.0))  
        self.lm_rho   = nn.Parameter(torch.tensor(0.0))   
        self.alpha_predictor = AlphaPredictor(in_ch=int(dim * 2))  
        self.reliability_predictor = ReliabilityPredictor(hidden=_cfg_int(config, "REL_HID", 32))

        use_edffn = _cfg_bool(config, "USE_EDFFN", False)
        edffn_patch = _cfg_int(config, "EDFFN_PATCH", 8)
        edffn_drop = _cfg_float(config, "EDFFN_DROP", 0.0)

        self.patch_embed = GatedEmb(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            SandwichBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                          bias=bias, LayerNorm_type=LayerNorm_type,
                          use_edffn=False, edffn_patch=edffn_patch, edffn_drop=edffn_drop)
            for _ in range(num_blocks[0])
        ])

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            SandwichBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                          bias=bias, LayerNorm_type=LayerNorm_type,
                          use_edffn=False, edffn_patch=edffn_patch, edffn_drop=edffn_drop)
            for _ in range(num_blocks[1])
        ])

        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential(*[
            SandwichBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                          bias=bias, LayerNorm_type=LayerNorm_type,
                          use_edffn=use_edffn, edffn_patch=edffn_patch, edffn_drop=edffn_drop)  
        ])

        self.down3_4 = Downsample(int(dim * 2 ** 2))
        self.latent = nn.Sequential(*[
            SandwichBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                          bias=bias, LayerNorm_type=LayerNorm_type,
                          use_edffn=use_edffn, edffn_patch=edffn_patch, edffn_drop=edffn_drop)   
            for _ in range(num_blocks[3])
        ])

        # ----- Decoder -----
        self.up4_3 = Upsample(int(dim * 2 ** 3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            SandwichBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                          bias=bias, LayerNorm_type=LayerNorm_type,
                          use_edffn=use_edffn, edffn_patch=edffn_patch, edffn_drop=edffn_drop) 
            for _ in range(num_blocks[2])
        ])

        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            SandwichBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                          bias=bias, LayerNorm_type=LayerNorm_type,
                          use_edffn=False, edffn_patch=edffn_patch, edffn_drop=edffn_drop)
            for _ in range(num_blocks[1])
        ])

        self.up2_1 = Upsample(int(dim * 2 ** 1))
        self.decoder_level1 = nn.Sequential(*[
            SandwichBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                          bias=bias, LayerNorm_type=LayerNorm_type,
                          use_edffn=False, edffn_patch=edffn_patch, edffn_drop=edffn_drop)
            for _ in range(num_blocks[0])
        ])
        self.texture_refine = TextureRefineBlock(in_ch=int(dim * 2 ** 1))

        self.output = nn.Sequential(
            nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        )

    @staticmethod
    def global_color_sim(img, ref):
        if img.min() < 0:
            img = (img + 1.0) * 0.5
        if ref.min() < 0:
            ref = (ref + 1.0) * 0.5
        mu_x = img.mean(dim=(2, 3))  # (B,3)
        mu_r = ref.mean(dim=(2, 3))  # (B,3)
        cos = F.cosine_similarity(mu_x, mu_r, dim=1).clamp(-1, 1)  # (B,)
        sim = (cos + 1.0) * 0.5
        return sim.view(-1, 1, 1, 1)

    def learnable_mask(self, sim, H, W, device=None, dtype=None):
   
        device = device or sim.device
        dtype = dtype or sim.dtype
        B = sim.shape[0]
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype),
            indexing='ij'
        )
        dist = torch.sqrt(yy ** 2 + xx ** 2)              
        dist = dist / dist.max()                         
        dist = dist.view(1, 1, H, W)                   
        r = self.lm_r_min + (self.lm_r_max - self.lm_r_min) * (1.0 - sim) ** self.lm_gamma

        alpha = F.softplus(self.lm_alpha)   
        rho   = self.lm_rho                 

        M = torch.sigmoid(alpha * (r - dist - rho))       # (B,1,H,W)
        return M

    def _set_edffn_sim(self, module, sim):
      
        for m in module.modules():
            if isinstance(m, EDFFN):
                m.current_sim = sim

    def forward(self, inp_img, mask_whole, mask_half, mask_quarter, mask_tiny, img_ref=None):

        sim = None
        if img_ref is not None:
            sim = self.global_color_sim(inp_img, img_ref)  # (B,1,1,1)
            sim = torch.clamp(sim, min=self.sim_min, max=1.0)

        rel = None
        gate_sig = sim  

        if (img_ref is not None) and (sim is not None) and (self.reliability_predictor is not None):
            rel = self.reliability_predictor(inp_img, img_ref, sim)  # (B,1,1,1)

            rel_min = float(getattr(self, "rel_min", 0.0) or 0.0)
            rel = torch.clamp(rel, min=rel_min, max=1.0)

            if not self.training:
    
                gate_sig = sim * rel
       
            else:
                it = int(getattr(self, "iteration", 0))

                mix_start = float(getattr(self, "rel_mix_start", 1.0) or 1.0)
                mix_end = float(getattr(self, "rel_mix_end", 0.3) or 0.3)
                warmup = int(getattr(self, "rel_mix_warmup", 20000) or 20000)

                if warmup > 0:
                    t = min(max(it / float(warmup), 0.0), 1.0)
                    alpha = mix_start + (mix_end - mix_start) * t
                else:
                    alpha = mix_end

                gate_sig = alpha * sim + (1.0 - alpha) * rel  # (B,1,1,1)

        if gate_sig is not None:
            gate_sig = torch.clamp(gate_sig, 0.0, 1.0)

        self.last_sim = sim.detach() if sim is not None else None
        self.last_rel = rel.detach() if rel is not None else None
        self.last_gate = gate_sig.detach() if gate_sig is not None else None

        if gate_sig is not None:
            self._set_edffn_sim(self.encoder_level3, gate_sig)
            self._set_edffn_sim(self.latent, gate_sig)
            self._set_edffn_sim(self.decoder_level3, gate_sig)

        # ----- Encoder -----
        inp_enc_level1 = self.patch_embed(torch.cat((inp_img, mask_whole), dim=1))
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1, mask_whole)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2, mask_half)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3, mask_quarter)
        latent = self.latent(inp_enc_level4)

        # ----- Decoder -----
        inp_dec_level3 = self.up4_3(latent, mask_tiny)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], dim=1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3, mask_quarter)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], dim=1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2, mask_half)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], dim=1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        if (self.style_encoder is not None) and (img_ref is not None):

            s_vec = self.style_encoder(img_ref)  # [B, STYLE_DIM]

            gamma, beta, alpha_base = self.shared_film(s_vec)

            gate = F.interpolate(mask_whole, size=out_dec_level1.shape[-2:], mode='bilinear', align_corners=False)
            k = max(1, int(self.gate_smooth))
            if k > 1:
                pad = k // 2
                gate = F.avg_pool2d(gate, kernel_size=k, stride=1, padding=pad)

            B, C, H, W = out_dec_level1.shape
            if gate_sig is None:
                gate_sig = torch.ones((B, 1, 1, 1), device=out_dec_level1.device, dtype=out_dec_level1.dtype)

            M = self.learnable_mask(gate_sig, H, W,
                                    device=out_dec_level1.device,
                                    dtype=out_dec_level1.dtype)
            low = lowpass_filter(out_dec_level1, ratio=self.lowfreq_ratio)
            high = out_dec_level1 - low
            feat_for_inject = M * low + (1.0 - M) * high

            alpha_dynamic = self.alpha_predictor(out_dec_level1, gate_sig)  # (B,1,1,1)

            out_dec_level1 = self.style_inject(
                feat_for_inject, gamma, beta, alpha_dynamic,
                gate=gate,
                scale=gate_sig
            )

        feat = self.texture_refine(out_dec_level1)
        out = self.output(feat)
        out = (torch.tanh(out) + 1) / 2
        return out



import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .networks import RGRG, Discriminator
from .loss import AdversarialLoss, PerceptualLoss, StyleLoss, TextureLoss


class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()
        self.name = name
        self.config = config
        self.iteration = 0

        self.gen_weights_path = os.path.join(config.PATH, name + '_gen.pth')
        self.dis_weights_path = os.path.join(config.PATH, name + '_dis.pth')
        os.makedirs(self.config.PATH, exist_ok=True)

    def load(self):
        if os.path.exists(self.gen_weights_path):
            print('Loading %s generator...' % self.name)
            data = torch.load(self.gen_weights_path, map_location='cpu')
            self.generator.load_state_dict(data['generator'], strict=False)
            self.iteration = data.get('iteration', 0)

        if self.config.MODE == 1 and os.path.exists(self.dis_weights_path):
            print('Loading %s discriminator...' % self.name)
            data = torch.load(self.dis_weights_path, map_location='cpu')
            self.discriminator.load_state_dict(data['discriminator'])

    def save(self):
        print(f'\nSaving {self.name} at iter {self.iteration}...\n')

        gen_ckpt = os.path.join(
            self.config.PATH,
            f"{self.name}_gen_{self.iteration:08d}.pth"
        )
        dis_ckpt = os.path.join(
            self.config.PATH,
            f"{self.name}_dis_{self.iteration:08d}.pth"
        )

        torch.save({
            'iteration': self.iteration,
            'generator': self.generator.state_dict()
        }, gen_ckpt)

        torch.save({
            'discriminator': self.discriminator.state_dict()
        }, dis_ckpt)

        torch.save({
            'iteration': self.iteration,
            'generator': self.generator.state_dict()
        }, self.gen_weights_path)

        torch.save({
            'discriminator': self.discriminator.state_dict()
        }, self.dis_weights_path)


class InpaintingModel(BaseModel):
    def __init__(self, config):
        super(InpaintingModel, self).__init__('InpaintingModel', config)

        generator = RGRG(config=config)
        discriminator = Discriminator(in_channels=3, use_sigmoid=config.GAN_LOSS != 'hinge')

        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, config.GPU)
            discriminator = nn.DataParallel(discriminator, config.GPU)

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)
        self.add_module('l1_loss', nn.L1Loss())
        self.add_module('perceptual_loss', PerceptualLoss())
        self.add_module('style_loss', StyleLoss())
        self.add_module('adversarial_loss', AdversarialLoss(type=config.GAN_LOSS))
        self.add_module('texture_loss', TextureLoss())

        self.ema_l1 = 0.0
        self.ema_style = 0.0
        self.ema_tex = 0.0
        self.ema_perc = 0.0
        self.ema_gan = 0.0
        self.ema_momentum = 0.01  

        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32)
        sobel_y = torch.tensor([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]], dtype=torch.float32)

        self.sobel_x = sobel_x.view(1, 1, 3, 3).to(config.DEVICE)
        self.sobel_y = sobel_y.view(1, 1, 3, 3).to(config.DEVICE)

        rel_lr_mult = getattr(config, "REL_LR_MULT", 5.0)
        if rel_lr_mult is None:
            rel_lr_mult = 5.0
        rel_lr_mult = float(rel_lr_mult)

        gen_mod = generator.module if isinstance(generator, nn.DataParallel) else generator

        rel_params = []
        base_params = []
        for n, p in gen_mod.named_parameters():
            if not p.requires_grad:
                continue
            if "reliability_predictor" in n:
                rel_params.append(p)
            else:
                base_params.append(p)
        print("rel params:", len(rel_params), "base params:", len(base_params))

        self.gen_optimizer = optim.Adam(
            [
                {"params": base_params, "lr": float(config.LR)},
                {"params": rel_params, "lr": float(config.LR) * rel_lr_mult},
            ],
            betas=(config.BETA1, config.BETA2)
        )

        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR),
            betas=(config.BETA1, config.BETA2)
        )

    def process(self, images, masks, image_ref=None):
        self.iteration += 1
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()

        outputs_img = self(images, masks, image_ref=image_ref)


        dis_real, _ = self.discriminator(images)
        dis_fake, _ = self.discriminator(outputs_img.detach())

        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss = (dis_real_loss + dis_fake_loss) / 2

        gen_fake, _ = self.discriminator(outputs_img)
        adv_raw = self.adversarial_loss(gen_fake, True, False)
        gen_gan_loss = adv_raw * self.config.INPAINT_ADV_LOSS_WEIGHT

        eps = 1e-6
        diff = torch.abs(outputs_img - images) * masks  
        num = torch.sum(masks) + eps                   
        l1_raw = diff.sum() / num
        gen_l1_loss = l1_raw * self.config.L1_LOSS_WEIGHT


        perc_raw = self.perceptual_loss(outputs_img, images)
        gen_content_loss = perc_raw * self.config.CONTENT_LOSS_WEIGHT

        sim = None
        style_weight = float(self.config.STYLE_LOSS_WEIGHT)

        gate_sig = None
        if image_ref is not None:
            gate_sig = getattr(self.generator, "last_gate", None)

        if (image_ref is not None) and (gate_sig is not None):
            gamma = float(getattr(self.config, "STYLE_GAMMA", 2.0))
            s_min = float(getattr(self.config, "STYLE_MIN_SIM_WEIGHT", 0.1))

            gate_nl = torch.clamp(gate_sig ** gamma, min=s_min)
            style_weight = float(self.config.STYLE_LOSS_WEIGHT) * gate_nl.mean()

        else:
            if (image_ref is not None) and hasattr(self.generator, "global_color_sim"):
                sim = self.generator.global_color_sim(images, image_ref)  # (B,1,1,1)
                sim = torch.clamp(sim, min=0.0, max=1.0)

                gamma = float(getattr(self.config, "STYLE_GAMMA", 2.0))
                s_min = float(getattr(self.config, "STYLE_MIN_SIM_WEIGHT", 0.1))

                sim_nl = torch.clamp(sim ** gamma, min=s_min)
                style_weight = float(self.config.STYLE_LOSS_WEIGHT) * sim_nl.mean()
            else:
                sim = None
                style_weight = float(self.config.STYLE_LOSS_WEIGHT)

        style_raw = self.style_loss(outputs_img * masks, images * masks)
        gen_style_loss = style_raw * style_weight

        tex_w_cfg = float(getattr(self.config, "TEXTURE_LOSS_WEIGHT", 0.0) or 0.0)
        if tex_w_cfg > 0 and image_ref is not None:
            tex_raw = self.texture_loss(outputs_img * masks, image_ref * masks)
            gen_texture_loss = tex_raw * tex_w_cfg
        else:
            tex_raw = torch.tensor(0.0, device=images.device)
            gen_texture_loss = torch.tensor(0.0, device=images.device)
        w_cons = float(getattr(self.config, "CONSIST_LOSS_WEIGHT", 0.0))
        if w_cons > 0 and image_ref is not None:
            with torch.no_grad():
                base_out = self.forward(images, masks, image_ref=None)  

            def sobel_edges(x):
                gray = x.mean(dim=1, keepdim=True)
                ex = F.conv2d(gray, self.sobel_x, padding=1)
                ey = F.conv2d(gray, self.sobel_y, padding=1)
                return torch.sqrt(ex ** 2 + ey ** 2 + 1e-6)

            edges_styl = sobel_edges(outputs_img)
            edges_base = sobel_edges(base_out)

            loss_l1_cons = F.l1_loss(outputs_img, base_out)
            loss_edge_cons = F.l1_loss(edges_styl, edges_base)
            loss_perc_cons = self.perceptual_loss(outputs_img, base_out)

            l_cons = (
                1.0 * loss_l1_cons +
                0.5 * loss_edge_cons +
                0.2 * loss_perc_cons
            )
            cons_loss = w_cons * l_cons
        else:
            cons_loss = torch.tensor(0.0, device=images.device)

        with torch.no_grad():
            cur_l1 = float(gen_l1_loss.detach().item())
            cur_style = float(gen_style_loss.detach().item())
            cur_tex = float(gen_texture_loss.detach().item())
            cur_perc = float(gen_content_loss.detach().item())
            cur_gan = float(gen_gan_loss.detach().item())

            m = self.ema_momentum
            if self.ema_l1 == 0.0:
                self.ema_l1 = cur_l1
                self.ema_style = cur_style
                self.ema_tex = cur_tex
                self.ema_perc = cur_perc
                self.ema_gan = cur_gan
            else:
                self.ema_l1 = (1 - m) * self.ema_l1 + m * cur_l1
                self.ema_style = (1 - m) * self.ema_style + m * cur_style
                self.ema_tex = (1 - m) * self.ema_tex + m * cur_tex
                self.ema_perc = (1 - m) * self.ema_perc + m * cur_perc
                self.ema_gan = (1 - m) * self.ema_gan + m * cur_gan

            ema_list = [v for v in [self.ema_l1, self.ema_style,
                                    self.ema_tex, self.ema_perc,
                                    self.ema_gan] if v > 0]
            if len(ema_list) > 0:
                avg_mag = sum(ema_list) / len(ema_list)
            else:
                avg_mag = 1.0

            eps_ema = 1e-8

            def make_scale(ema_val):
                if ema_val <= 0:
                    return 1.0
                s = avg_mag / (ema_val + eps_ema)
    
                s = max(0.5, min(2.0, s))
                return s

            scale_l1 = make_scale(self.ema_l1)
            scale_style = make_scale(self.ema_style)
            scale_tex = make_scale(self.ema_tex)
            scale_perc = make_scale(self.ema_perc)
            scale_gan = make_scale(self.ema_gan)

    
        gen_loss = (
            gen_gan_loss * scale_gan +
            gen_l1_loss * scale_l1 +
            gen_content_loss * scale_perc +
            gen_style_loss * scale_style +
            gen_texture_loss * scale_tex +
            cons_loss
        )

        logs = [
            ("gen_total", float(gen_loss.item())),
            ("dis_total", float(dis_loss.item())),
            ("gan_loss", float(gen_gan_loss.item())),
            ("l1_loss", float(gen_l1_loss.item())),
            ("content_loss", float(gen_content_loss.item())),
            ("style_loss", float(gen_style_loss.item())),
            ("texture_loss", float(gen_texture_loss.item())),
        ]

        logs += [
            ("scale_gan", float(scale_gan)),
            ("scale_l1", float(scale_l1)),
            ("scale_perc", float(scale_perc)),
            ("scale_style", float(scale_style)),
            ("scale_tex", float(scale_tex)),
        ]

        # if image_ref is not None and sim is not None:
        #     logs.append(("style_sim", float(sim.mean().item())))
        if image_ref is not None:
            last_sim = getattr(self.generator, "last_sim", None)
            last_rel = getattr(self.generator, "last_rel", None)
            last_gate = getattr(self.generator, "last_gate", None)

            if last_sim is not None:
                logs.append(("style_sim", float(last_sim.mean().item())))
            if last_rel is not None:
                logs.append(("style_rel", float(last_rel.mean().item())))
            if last_gate is not None:
                logs.append(("style_gate", float(last_gate.mean().item())))

        return outputs_img, gen_loss, dis_loss, logs, gen_gan_loss, gen_l1_loss, gen_content_loss, gen_style_loss

    def forward(self, images, masks, image_ref=None):

        images_masked = (images * (1 - masks).float()) + masks
        inputs = images_masked

        scaled_masks_tiny = F.interpolate(masks, size=[masks.shape[2] // 8, masks.shape[3] // 8], mode='nearest')
        scaled_masks_quarter = F.interpolate(masks, size=[masks.shape[2] // 4, masks.shape[3] // 4], mode='nearest')
        scaled_masks_half = F.interpolate(masks, size=[masks.shape[2] // 2, masks.shape[3] // 2], mode='nearest')

        outputs_img = self.generator(
            inputs, masks, scaled_masks_half, scaled_masks_quarter, scaled_masks_tiny,
            img_ref=image_ref
        )
        return outputs_img

    def backward(self, gen_loss=None, dis_loss=None):
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 5.0)
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 5.0)

        dis_loss.backward(retain_graph=True)
        gen_loss.backward()

        self.dis_optimizer.step()
        self.gen_optimizer.step()

    def backward_joint(self, gen_loss=None, dis_loss=None):
        dis_loss.backward()
        self.dis_optimizer.step()
        gen_loss.backward()
        self.gen_optimizer.step()

def abs_smooth(x):
    absx = torch.abs(x)
    minx = torch.min(absx, other=torch.ones_like(absx))
    r = 0.5 * ((absx - 1) * minx + absx)
    return r

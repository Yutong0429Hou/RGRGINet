import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class AdversarialLoss(nn.Module):
    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        super(AdversarialLoss, self).__init__()
        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()
        elif type == 'lsgan':
            self.criterion = nn.MSELoss()
        elif type == 'hinge':
            self.criterion = nn.ReLU()
        else:
            raise ValueError(f"Unsupported GAN type: {type}")

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()
        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            return self.criterion(outputs, labels)

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features

        self.relu1_1 = nn.Sequential(*features[0:2])   
        self.relu2_1 = nn.Sequential(*features[0:7])  
        self.relu3_1 = nn.Sequential(*features[0:12])  
        self.relu4_1 = nn.Sequential(*features[0:21])  
        self.relu5_1 = nn.Sequential(*features[0:30])  

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        out = {}
        out['relu1_1'] = self.relu1_1(x)
        out['relu2_1'] = self.relu2_1(x)
        out['relu3_1'] = self.relu3_1(x)
        out['relu4_1'] = self.relu4_1(x)
        out['relu5_1'] = self.relu5_1(x)
        return out

class PerceptualLoss(nn.Module):
    def __init__(self, weights=[1, 1, 1, 1, 1]):
        super(PerceptualLoss, self).__init__()
        self.vgg = VGG19().eval()
        self.l1 = nn.L1Loss()
        self.weights = weights

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0.0
        for i, k in enumerate(['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']):
            loss += self.weights[i] * self.l1(x_vgg[k], y_vgg[k].detach())
        return loss

class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()
        self.vgg = VGG19().eval()
        self.l1 = nn.L1Loss()

    def gram_matrix(self, feat):
        (b, ch, h, w) = feat.size()
        feat = feat.view(b, ch, h * w)
        gram = torch.bmm(feat, feat.transpose(1, 2))
        return gram / (ch * h * w)

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0.0
        for k in ['relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']:
            gram_x = self.gram_matrix(x_vgg[k])
            gram_y = self.gram_matrix(y_vgg[k])
            loss += self.l1(gram_x, gram_y.detach())
        return loss

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))
        return loss

class ConsistencyLoss(nn.Module):
    def __init__(self, weight=0.1):
        super(ConsistencyLoss, self).__init__()
        self.weight = weight
        self.l1 = nn.L1Loss()

    def forward(self, styled_out, base_out):
        return self.weight * self.l1(styled_out, base_out)

class TextureLoss(nn.Module):

    def __init__(self,
                 layers=('relu2_1', 'relu3_1', 'relu4_1'),
                 weights=(1.0, 1.0, 1.0),
                 eps=1e-6):
        super(TextureLoss, self).__init__()
        self.vgg = VGG19().eval()
        self.layers = layers
        self.weights = weights
        self.eps = eps
        self.l1 = nn.L1Loss()

    def _moments(self, feat):
        B, C, H, W = feat.shape
        feat_flat = feat.view(B, C, -1)
        mu = feat_flat.mean(dim=-1)                           
        var = feat_flat.var(dim=-1, unbiased=False)           
        sigma = torch.sqrt(var + self.eps)                   
        return mu, sigma

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0.0
        for layer, w in zip(self.layers, self.weights):
            fx = x_vgg[layer]
            fy = y_vgg[layer]

            mu_x, sig_x = self._moments(fx)
            mu_y, sig_y = self._moments(fy)
            l_mu = self.l1(mu_x, mu_y.detach())
            l_sigma = self.l1(sig_x, sig_y.detach())

            loss = loss + w * (l_mu + l_sigma)
        return loss

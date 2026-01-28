import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image
import cv2

from .utils import create_mask


def _to_abs_path(p):
    if p is None:
        return None
    p = os.path.expanduser(str(p))
    return os.path.abspath(p)


class Dataset(Dataset):
    def __init__(
        self,
        config,
        flist,
        mask_flist,
        augment=True,
        training=True,
        ref_flist=None,          
    ):
        super().__init__()
        self.config = config
        self.augment = augment
        self.training = training

        self.data = self.load_flist(flist)
        self.mask_data = self.load_flist(mask_flist)

        self.ref_paths = self.load_flist(ref_flist) if ref_flist else None

        self.image_paths = self.data

        self.input_size = int(getattr(config, "INPUT_SIZE", 0) or 0)
        self.mask = int(getattr(config, "MASK", 0) or 0)


        self.ref_strategy = str(getattr(config, "STYLE_REF_STRATEGY", "random") or "random").lower()
        self.same_ratio = float(getattr(config, "STYLE_SAME_RATIO", 1.0) or 1.0)


    def __len__(self):
        return len(self.data)

    def load_name(self, index):
        return os.path.basename(self.data[index])

    @staticmethod
    def load_flist(flist):
        if flist is None:
            return []
        flist = _to_abs_path(flist)
        if not os.path.exists(flist):
            raise FileNotFoundError(f"找不到路径/清单: {flist}")

        images = []
        if os.path.isdir(flist):
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp"):
                images.extend(glob.glob(os.path.join(flist, "**", ext), recursive=True))
        else:
      
            candidates = []
            if "," in flist:
                candidates = [x.strip() for x in flist.split(",")]
            else:
                candidates = [flist]

            for c in candidates:
                c = _to_abs_path(c)
                if os.path.isdir(c):
                    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp"):
                        images.extend(glob.glob(os.path.join(c, "**", ext), recursive=True))
                else:
            
                    with open(c, "r", encoding="utf-8") as f:
                        for line in f:
                            p = line.strip()
                            if not p:
                                continue
                            images.append(_to_abs_path(p))

        images = sorted(list(set(images)))
        return images


    def __getitem__(self, index):

        img_t, mask_t, img_path = self.load_item(index)
        ref_t = self._sample_ref(img_path)

        return img_t, mask_t, ref_t

    def load_item(self, index):
        size = self.input_size

        img_path = self.data[index]
        img = self.imread_rgb(img_path)  # HWC, uint8, RGB

        if size != 0:
            img = self.resize(img, size, size, centerCrop=True)

        mask = self.load_mask(img, index)

        img_t = self.to_tensor_img(img)
        mask_t = self.to_tensor_mask(mask)

        return img_t, mask_t, img_path

    def _sample_ref(self, content_path):
       
        size = self.input_size
        def _finalize(path):
            img = self.imread_rgb(path)
            if size != 0:
                img = self.resize(img, size, size, centerCrop=True)
            return self.to_tensor_img(img)

        if self.ref_paths and self.ref_strategy in ("same_dir", "ref_flist"):
            if self.ref_strategy == "same_dir":
  
                content_dir = os.path.dirname(content_path)
                cand = [p for p in self.ref_paths if os.path.dirname(p) == content_dir]
                base = os.path.basename(content_path)
                cand = [p for p in cand if os.path.basename(p) != base] or cand

                if len(cand) == 0:
                    path = random.choice(self.ref_paths)
                else:
                    path = random.choice(cand)
            else:
                path = random.choice(self.ref_paths)

            return _finalize(path)

        ref_idx = random.randint(0, len(self.image_paths) - 1)
        path = self.image_paths[ref_idx]
        return _finalize(path)
        
    def load_mask(self, img, index):
        imgh, imgw = img.shape[:2]
        mask_type = self.mask


        if mask_type == 5:
            mask_type = 0 if np.random.uniform(0, 1) >= 0.5 else 4


        if mask_type == 4:
            mask_type = 1 if np.random.binomial(1, 0.5) == 1 else 3

        # no mask
        if mask_type == 0:
            return np.zeros((imgh, imgw), dtype=np.uint8)

        # random block
        if mask_type == 1:
            m = create_mask(imgw, imgh, imgw // 2, imgh // 2)  # 0/1
            return (m * 255).astype(np.uint8)

        # center mask
        if mask_type == 2:
            m = create_mask(imgw, imgh, imgw // 2, imgh // 2, x=imgw // 4, y=imgh // 4)
            return (m * 255).astype(np.uint8)

        # external
        if mask_type == 3:
            midx = random.randint(0, len(self.mask_data) - 1)
            m = self.imread_gray(self.mask_data[midx])
            m = self.resize(m, imgh, imgw, keep3ch=False, centerCrop=True)
            m = (m > 0).astype(np.uint8) * 255
            return m

        # external non-random
        if mask_type == 6:
            m = self.imread_gray(self.mask_data[index % len(self.mask_data)])
            m = self.resize(m, imgh, imgw, keep3ch=False, centerCrop=False)
            m = (m > 0).astype(np.uint8) * 255
            return m

        # fallback
        return np.zeros((imgh, imgw), dtype=np.uint8)

    def imread_rgb(self, path):

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

   
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def imread_gray(self, path):

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"读掩码失败: {path}")
        return img

    def to_tensor_img(self, img_rgb_uint8):
        """HWC uint8 RGB -> tensor float [0,1] 3xHxW"""
        pil = Image.fromarray(img_rgb_uint8)
        return TF.to_tensor(pil).float()

    def to_tensor_mask(self, mask_uint8):
        """HxW uint8 (0/255) -> float [0,1] 1xHxW"""
        if mask_uint8.ndim == 3:
            mask_uint8 = cv2.cvtColor(mask_uint8, cv2.COLOR_BGR2GRAY)
        mask_uint8 = (mask_uint8 > 0).astype(np.uint8) * 255
        pil = Image.fromarray(mask_uint8)
        t = TF.to_tensor(pil).float()  # 1xHxW
        return t

    def resize(self, img, height, width, keep3ch=True, centerCrop=True):
       
        if img.ndim == 3:
            imgh, imgw, _ = img.shape
        else:
            imgh, imgw = img.shape[:2]

        x = img
        if centerCrop and imgh != imgw:
            side = min(imgh, imgw)
            top = (imgh - side) // 2
            left = (imgw - side) // 2
            if img.ndim == 3:
                x = x[top:top + side, left:left + side, :]
            else:
                x = x[top:top + side, left:left + side]

        pil_mode = "RGB" if (x.ndim == 3 and (x.shape[2] == 3 or (keep3ch and x.ndim == 3))) else "L"
        pil = Image.fromarray(x, mode=pil_mode)
        pil = pil.resize((width, height), resample=Image.BILINEAR)
        arr = np.array(pil)

        return arr

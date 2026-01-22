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
    """
    返回三元组: (img_t, mask_t, ref_t)
      - img_t:   内容图 (3xH xW, float[0,1])
      - mask_t:  掩码   (1xH xW, float[0,1])
      - ref_t:   参考图 (3xH xW, float[0,1])；若未启用风格库，则回退为同库随机
    """
    def __init__(
        self,
        config,
        flist,
        mask_flist,
        augment=True,
        training=True,
        ref_flist=None,           # ✨ 新增：独立风格库（可为 None）
    ):
        super().__init__()
        self.config = config
        self.augment = augment
        self.training = training

        # 主数据集（内容图）/ 掩码
        self.data = self.load_flist(flist)
        self.mask_data = self.load_flist(mask_flist)

        # 参考数据集（风格库，可选）
        self.ref_paths = self.load_flist(ref_flist) if ref_flist else None

        # 与 __getitem__ 兼容的名字
        self.image_paths = self.data

        # 关键配置
        self.input_size = int(getattr(config, "INPUT_SIZE", 0) or 0)
        self.mask = int(getattr(config, "MASK", 0) or 0)

        # 参考策略
        self.ref_strategy = str(getattr(config, "STYLE_REF_STRATEGY", "random") or "random").lower()
        self.same_ratio = float(getattr(config, "STYLE_SAME_RATIO", 1.0) or 1.0)

        # 自检
        assert len(self.data) > 0, f"图像 flist 为空或未正确读取：{flist}"
        if self.mask in (3, 4, 6):  # 需要外部 mask
            assert len(self.mask_data) > 0, f"掩码 flist 为空或未正确读取：{mask_flist}"

    # ---------------- 基础 IO ----------------
    def __len__(self):
        return len(self.data)

    def load_name(self, index):
        return os.path.basename(self.data[index])

    @staticmethod
    def load_flist(flist):
        """
        支持:
        - 目录：自动递归收集图像
        - 文本清单：逐行路径
        - 逗号分隔的多个路径
        """
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
            # 可能是逗号分隔的多个文件
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
                    # 文本清单
                    with open(c, "r", encoding="utf-8") as f:
                        for line in f:
                            p = line.strip()
                            if not p:
                                continue
                            images.append(_to_abs_path(p))

        images = sorted(list(set(images)))
        return images

    # ---------------- 主取样 ----------------
    def __getitem__(self, index):
        # 主样本：图像 & 掩码（tensor）
        img_t, mask_t, img_path = self.load_item(index)

        # 参考图：优先用独立风格库（若存在且启用）；否则回退为同库随机
        ref_t = self._sample_ref(img_path)

        return img_t, mask_t, ref_t

    # ---------------- 加载一条内容样本 ----------------
    def load_item(self, index):
        size = self.input_size

        # 1) 读主图（RGB）
        img_path = self.data[index]
        img = self.imread_rgb(img_path)  # HWC, uint8, RGB

        # 2) 统一到输入尺寸（0 表示保留原尺寸；默认 center crop 到正方形）
        if size != 0:
            img = self.resize(img, size, size, centerCrop=True)

        # 3) 构造/读取掩码（uint8 0/255, HxW）
        mask = self.load_mask(img, index)

        # 4) 转 tensor：图像 3xHxW、掩码 1xHxW，范围 [0,1]
        img_t = self.to_tensor_img(img)
        mask_t = self.to_tensor_mask(mask)

        return img_t, mask_t, img_path

    # ---------------- 参考图采样（相似但不同） ----------------
    def _sample_ref(self, content_path):
        """
        根据策略选择参考图：
          - same_dir: 参考图优先与内容图同目录；若找不到则回退到 ref_flist / 同库随机
          - ref_flist: 仅从独立风格库随机
          - random/disabled: 从内容库随机
        """
        size = self.input_size

        def _finalize(path):
            img = self.imread_rgb(path)
            if size != 0:
                img = self.resize(img, size, size, centerCrop=True)
            return self.to_tensor_img(img)

        # 1) 使用独立风格库的分支
        if self.ref_paths and self.ref_strategy in ("same_dir", "ref_flist"):
            if self.ref_strategy == "same_dir":
                # 同目录优先（“相似但不同”的近域采样）
                content_dir = os.path.dirname(content_path)
                cand = [p for p in self.ref_paths if os.path.dirname(p) == content_dir]
                # 避免与内容图同名（仍保留候选避免空集）
                base = os.path.basename(content_path)
                cand = [p for p in cand if os.path.basename(p) != base] or cand

                if len(cand) == 0:
                    # 回退：整个 ref 库随机
                    path = random.choice(self.ref_paths)
                else:
                    path = random.choice(cand)
            else:
                # ref_flist: 直接在风格库随机
                path = random.choice(self.ref_paths)

            return _finalize(path)

        # 2) 回退：同库随机
        ref_idx = random.randint(0, len(self.image_paths) - 1)
        path = self.image_paths[ref_idx]
        return _finalize(path)

    # ---------------- 掩码构造 ----------------
    def load_mask(self, img, index):
        imgh, imgw = img.shape[:2]
        mask_type = self.mask

        # 50% no mask, 25% random block, 25% external
        if mask_type == 5:
            mask_type = 0 if np.random.uniform(0, 1) >= 0.5 else 4

        # external + random block（二选一）
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

        # external（随机）
        if mask_type == 3:
            midx = random.randint(0, len(self.mask_data) - 1)
            m = self.imread_gray(self.mask_data[midx])
            m = self.resize(m, imgh, imgw, keep3ch=False, centerCrop=True)
            m = (m > 0).astype(np.uint8) * 255
            return m

        # external non-random（按 index 对应）
        if mask_type == 6:
            m = self.imread_gray(self.mask_data[index % len(self.mask_data)])
            m = self.resize(m, imgh, imgw, keep3ch=False, centerCrop=False)
            m = (m > 0).astype(np.uint8) * 255
            return m

        # fallback
        return np.zeros((imgh, imgw), dtype=np.uint8)

    # ---------------- IO / 变换 ----------------
    def imread_rgb(self, path):
        """读取为 RGB（丢弃 alpha），HWC uint8"""
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        # if img is None:
        #     raise RuntimeError(f"读图失败: {path}")
        if img is None:
            raise RuntimeError(f"读图失败: {path}")
        if np.isnan(img).any() or np.isinf(img).any():
            raise RuntimeError(f"检测到非法像素: {path}")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def imread_gray(self, path):
        """读取灰度，HxW uint8"""
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
        """
        使用 PIL resize（size=(width,height)）
        支持 img 为 HxW 或 HxWxC；默认做正方形 center crop 后再缩放
        """
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

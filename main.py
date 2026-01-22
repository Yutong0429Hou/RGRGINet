# main.py
import os
import cv2
import random
import numpy as np
import torch
import argparse
from shutil import copyfile
from contextlib import nullcontext

from src.config import Config
from src.HINT import HINT

# 可选依赖
try:
    from torchinfo import summary as torchinfo_summary
except Exception:
    torchinfo_summary = None

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

try:
    import wandb
except Exception:
    wandb = None


# =========================
# 配置加载
# =========================
def load_config(mode=None, config_file=None):
    project_root = os.path.dirname(os.path.abspath(__file__))
    default_ckpt_dir = os.path.join(project_root, "checkpoints")
    example_cfg_1 = os.path.join(project_root, "config.yml.example")
    example_cfg_2 = os.path.join(project_root, "config.yml")

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=default_ckpt_dir, help='model checkpoints path')
    parser.add_argument('--model', type=int, default=2, choices=[2])

    if mode == 2:
        parser.add_argument('--input', type=str, help='path to input images')
        parser.add_argument('--mask', type=str, help='path to masks')
        parser.add_argument('--output', type=str, help='path to outputs')

    args, _ = parser.parse_known_args()

    if config_file is not None:
        config_path = os.path.abspath(os.path.expanduser(config_file))
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
    else:
        args.path = os.path.abspath(os.path.expanduser(args.path))
        config_path = os.path.join(args.path, 'config.yml')
        os.makedirs(args.path, exist_ok=True)
        if not os.path.exists(config_path):
            if os.path.exists(example_cfg_1):
                copyfile(example_cfg_1, config_path)
            elif os.path.exists(example_cfg_2):
                copyfile(example_cfg_2, config_path)
            else:
                raise FileNotFoundError("找不到配置模板，请检查 config.yml 或 config.yml.example")

    config = Config(config_path)
    print("✅ 使用配置文件:", config_path)

    if mode == 1:
        config.MODE = 1
        if args.model:
            config.MODEL = args.model
    elif mode == 2:
        config.MODE = 2
        config.MODEL = args.model
        if args.input:
            config.TEST_INPAINT_IMAGE_FLIST = args.input
        if args.mask:
            config.TEST_MASK_FLIST = args.mask
        if args.output:
            config.RESULTS = args.output

    return config


# =========================
# 稳健初始化 W&B
# =========================
def _config_to_dict(cfg_obj):
    d = {}
    for k, v in cfg_obj.__dict__.items():
        if k.isupper():
            d[k] = v
    return d


def init_wandb_with_fallback(config):
    if wandb is None:
        print("wandb 未安装，跳过监控。")
        return nullcontext()

    if os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true"):
        print("W&B 已禁用。")
        return nullcontext()

    cfg_dict = _config_to_dict(config)

    try:
        print("W&B: 尝试在线初始化...")
        return wandb.init(project="HINT", config=cfg_dict)
    except Exception as e1:
        print(f"⚠️ W&B 在线失败: {e1}\n切换到离线模式。")
        try:
            return wandb.init(project="HINT", mode="offline", config=cfg_dict)
        except Exception as e2:
            print(f"⚠️ W&B 离线失败: {e2}\n完全禁用。")
            os.environ["WANDB_DISABLED"] = "true"
            return nullcontext()


# =========================
# 可视化：结构 / Summary / TensorBoard
# =========================
def show_text_architecture(model_wrap):
    print("\n=== InpaintingModel ===")
    print(model_wrap)
    try:
        print("\n=== Generator (HINT) ===")
        print(model_wrap.generator)
    except Exception:
        pass
    try:
        print("\n=== Discriminator ===")
        print(model_wrap.discriminator)
    except Exception:
        pass


def show_model_summary(model_wrap, input_size=256, use_style=True, device=None):
    if torchinfo_summary is None:
        print("[torchinfo] 未安装，跳过 summary。")
        return
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_wrap = model_wrap.to(device).eval()
    images = torch.randn(1, 3, input_size, input_size, device=device)
    masks = torch.randint(0, 2, (1, 1, input_size, input_size), device=device).float()
    ref = torch.randn(1, 3, input_size, input_size, device=device) if use_style else None

    print("\n>>> torchinfo.summary on InpaintingModel")
    torchinfo_summary(
        model_wrap,
        input_data=(images, masks, ref),
        verbose=2,
        col_names=("kernel_size", "num_params", "mult_adds", "output_size")
    )


def add_graph_to_tensorboard(model_wrap, logdir, input_size=256, use_style=True, device=None):
    if SummaryWriter is None:
        print("[TensorBoard] 未安装，跳过图结构。")
        return
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=logdir)
    model_wrap = model_wrap.to(device).eval()
    with torch.no_grad():
        images = torch.randn(1, 3, input_size, input_size, device=device)
        masks = torch.randint(0, 2, (1, 1, input_size, input_size), device=device).float()
        ref = torch.randn(1, 3, input_size, input_size, device=device) if use_style else None
        writer.add_graph(model_wrap, (images, masks, ref))
    writer.close()
    print(f"[TB] 计算图已保存: {logdir}\n启动命令: tensorboard --logdir {logdir}")


# =========================
# 主入口
# =========================
def main(mode=None, config_file=None):
    config = load_config(mode, config_file)

    with init_wandb_with_fallback(config) as run:
        # 设备设置
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)
        if torch.cuda.is_available():
            print('✅ CUDA 可用')
            config.DEVICE = torch.device("cuda")
            torch.backends.cudnn.benchmark = False  # <--- 改为 False
            torch.backends.cudnn.deterministic = True  # <--- 新增这行
        else:
            print('⚠️ CUDA 不可用，使用 CPU')
            config.DEVICE = torch.device("cpu")

        # 随机种子
        cv2.setNumThreads(0)
        torch.manual_seed(config.SEED)
        np.random.seed(config.SEED)
        random.seed(config.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.SEED)

        # 构建模型
        model = HINT(config)
        model.load()

        # 打印架构
        try:
            show_text_architecture(model.inpaint_model)
            show_model_summary(
                model.inpaint_model,
                input_size=config.INPUT_SIZE,
                use_style=getattr(config, "USE_STYLE", False),
                device=config.DEVICE
            )
            add_graph_to_tensorboard(
                model.inpaint_model,
                logdir=f"{config.PATH}/tb_graph",
                input_size=config.INPUT_SIZE,
                use_style=getattr(config, "USE_STYLE", False),
                device=config.DEVICE
            )
        except Exception as viz_e:
            print(f"[可视化] 跳过: {viz_e}")

        # W&B watch
        if (wandb is not None and
            os.getenv("WANDB_DISABLED", "").lower() not in ("1", "true") and
            wandb.run is not None):
            try:
                wandb.watch(getattr(model, "inpaint_model", model), log="all", log_freq=10)
            except Exception as werr:
                print(f"[W&B] watch 跳过: {werr}")

        # 模式选择
        if config.MODE == 1:
            config.print()
            print('\n🚀 开始训练...\n')
            model.train()
        elif config.MODE == 2:
            print('\n🧪 开始测试...\n')
            model.test()


if __name__ == "__main__":
    # 默认训练阶段
    main(mode=1, config_file="config.yml")

    # 可选微调阶段
    # main(mode=1, config_file="finetune.yml")

import os
import glob

def make_flist(data_dir, output_path, exts=['.jpg', '.png', '.jpeg']):
    """
    生成 .flist 文件，每一行是图片的绝对路径
    """
    data_dir = os.path.abspath(data_dir)
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(data_dir, '**', f'*{ext}'), recursive=True))

    files = sorted(files)
    print(f"找到 {len(files)} 张图片，写入 {output_path}")

    with open(output_path, 'w', encoding='utf-8') as f:
        for file in files:
            f.write(file.replace('\\', '/') + '\n')  # 确保是 / 路径

if __name__ == "__main__":
    # 根据你的数据路径生成 flist
    make_flist("/root/autodl-tmp/XGGGD/data/data512/train_images",
               "/root/autodl-tmp/XGGGD/data/data512/train_images.flist")

    make_flist("/root/autodl-tmp/XGGGD/data/data512/train_masks",
               "/root/autodl-tmp/XGGGD/data/data512/train_masks.flist")

    make_flist("/root/autodl-tmp/XGGGD/data/data512/test_images",
               "/root/autodl-tmp/XGGGD/data/data512/test_images.flist")

    make_flist("/root/autodl-tmp/XGGGD/data/data512/test_masks",
               "/root/autodl-tmp/XGGGD/data/data512/test_masks.flist")

    make_flist("/root/autodl-tmp/XGGGD/data/data512/style_images",
               "/root/autodl-tmp/XGGGD/data/data512/style_images.flist")

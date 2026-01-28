import os
import glob

def make_flist(data_dir, output_path, exts=['.jpg', '.png', '.jpeg']):

    data_dir = os.path.abspath(data_dir)
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(data_dir, '**', f'*{ext}'), recursive=True))

    files = sorted(files)


    with open(output_path, 'w', encoding='utf-8') as f:
        for file in files:
            f.write(file.replace('\\', '/') + '\n') 

if __name__ == "__main__":
    
    make_flist("/data/data512/train_images",
               "/data/data512/train_images.flist")

    make_flist("/data/data512/train_masks",
               "/data/data512/train_masks.flist")

    make_flist("/data/data512/test_images",
               "/data/data512/test_images.flist")

    make_flist("/data512/test_masks",
               "/data512/test_masks.flist")

    make_flist("/data512/style_images",
               "/data/data512/style_images.flist")

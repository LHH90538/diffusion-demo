import os
import numpy as np
from PIL import Image

src = "./stl10_binary"       # 解压后的 ST-10 文件目录
dst = "./stl10_png"          # 输出目录
os.makedirs(dst, exist_ok=True)

IMG_SIZE = 96
NUM_CHANNELS = 3
IMG_BYTES = IMG_SIZE * IMG_SIZE * NUM_CHANNELS

def load_stl_images(file_path):
    with open(file_path, "rb") as f:
        data = np.fromfile(f, dtype=np.uint8)
        total_images = data.shape[0] // IMG_BYTES

        data = data.reshape(total_images, NUM_CHANNELS, IMG_SIZE, IMG_SIZE)
        data = np.transpose(data, (0, 2, 3, 1))   # CHW → HWC
        return data

def save_images(images, prefix):
    for i, img in enumerate(images):
        Image.fromarray(img).save(f"{dst}/{prefix}_{i:05d}.png")

print("Loading STL10 train set...")
train_imgs = load_stl_images(os.path.join(src, "train_X.bin"))
save_images(train_imgs, "train")

print("Loading STL10 test set...")
test_imgs = load_stl_images(os.path.join(src, "test_X.bin"))
save_images(test_imgs, "test")

print("Loading STL10 unlabeled set...")
unlabeled_imgs = load_stl_images(os.path.join(src, "unlabeled_X.bin"))
save_images(unlabeled_imgs, "unlabeled")

print("Done! PNG saved in ./stl10_png")

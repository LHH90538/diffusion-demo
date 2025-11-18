import pickle, numpy as np, os
from PIL import Image

src = "./cifar_data/cifar-10-batches-py"
dst = "./cifar_png"
os.makedirs(dst, exist_ok=True)

def load_batch(file):
    with open(file, "rb") as f:
        datadict = pickle.load(f, encoding="bytes")
        X = datadict[b'data']
        X = X.reshape(10000, 3, 32, 32)
        return X

all_images = []
for i in range(1, 6):
    batch_file = f"{src}/data_batch_{i}"
    imgs = load_batch(batch_file)
    all_images.append(imgs)

images = np.concatenate(all_images, axis=0)

print("Saving PNG images...")
for i, img in enumerate(images):
    img = np.transpose(img, (1, 2, 0))           # CHW → HWC
    Image.fromarray(img).save(f"{dst}/{i:05d}.png")

print("Done! PNG saved to ./cifar_png")

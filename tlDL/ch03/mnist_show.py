# coding: utf-8
import sys, os
# Ensure the project root (parent of this file's directory) is on sys.path so
# imports like `from dataset.mnist import load_mnist` work regardless of the
# current working directory or how the script is launched.
here = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(here)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label)  # 5

print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 把图像的形状变为原来的尺寸
print(img.shape)  # (28, 28)

img_show(img)


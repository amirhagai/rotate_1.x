# Copyright (c) OpenMMLab. All rights reserved.
from PIL import Image
import numpy as np

im = Image.open("/app/data/test_injected/trainval/images/pres_dota.png")
im = np.array(im)
print(im.shape)

im2 = Image.open(
    "/app/data/test_injected/trainval/images/P0005__1024__0___0.png"
)
im2 = np.array(im2)
print(im2.shape)
print(im2.max())

# 定义读取图片的函数, 并将其resize成width*height尺寸大小
from PIL import Image
import os.path
import glob
import numpy as np
def read_img(image_path):
    cate = [path+f for f in os.listdir(image_path) if os.path.isdir(image_path+f)]
    imgs = []
    labels = []
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder+'/*.jpg'):
            print('reading the images:%s' % im)
            img = io.imread(im)
            try:
                if img.shape[2] == 3:
                    img = transform.resize(img, (width, height))
                    imgs.append(img)
                    labels.append(idx)
            except:
                continue
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)
for jpgfile in glob.glob("G:\img\img\*.jpg"):
    read_img("G:\img\img")

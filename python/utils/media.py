#!/usr/bin/env python3

# import re
import random
import numpy as np
# import os.path
import scipy.misc
from scipy import ndimage
# import shutil
# import zipfile
# import time
# import tensorflow as tf
# from glob import glob
# from urllib.request import urlretrieve
# from tqdm import tqdm

def img_size(image):
    return image.shape[0], image.shape[1]

def crop_image(image, gt_image):
    h, w = img_size(image)
    nw = random.randint(1150, w-5)  # Random crop size
    nh = int(nw / 3.3) # Keep original aspect ration
    x1 = random.randint(0, w - nw)  # Random position of crop
    y1 = random.randint(0, h - nh)
    return image[y1:(y1+nh), x1:(x1+nw), :], gt_image[y1:(y1+nh), x1:(x1+nw), :]


def flip_image(image, gt_image):
    return np.flip(image, axis=1), np.flip(gt_image, axis=1)

def bc_img(img, s=1.0, m=0.0):
    img = img.astype(np.int)
    img = img * s + m
    img[img > 255] = 255
    img[img < 0] = 0
    img = img.astype(np.uint8)
    return img


def process_gt_image(gt_image):
    background_color = np.array([255, 0, 0])

    gt_bg = np.all(gt_image == background_color, axis=2)
    gt_bg = gt_bg.reshape(gt_bg.shape[0], gt_bg.shape[1], 1)

    gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)
    return gt_image

def denoise_img(img):
    eroded_img = ndimage.binary_erosion(img)
    return ndimage.binary_propagation(eroded_img, mask=img)


def paste_mask(street_im, im_soft_max, image_shape, color, obj_color_schema):
    im_soft_max_r = im_soft_max[0][:, color].reshape(image_shape[0], image_shape[1])
    segmentation_r = (im_soft_max_r > 0.5).reshape(image_shape[0], image_shape[1], 1)
    mask = np.dot(segmentation_r, np.array(obj_color_schema))
    mask = scipy.misc.toimage(mask, mode="RGBA")
    street_im.paste(mask, box=None, mask=mask)

    return street_im

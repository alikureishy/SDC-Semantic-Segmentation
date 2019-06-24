#!/usr/bin/env python3

import re
import random
import numpy as np
import os.path
import scipy.misc
from scipy import ndimage
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm
from moviepy.editor import VideoFileClip

def predict_video(sess, image_shape, logits, keep_prob, input_image):
    video_dir = r"./test_video//"
    video_library =   [["GOPR0706_cut1.mp4", [210, 470]],
                        ["GOPR0706_cut2.mp4", [210, 470]],
                        ["GOPR0707_cut1.mp4", [316, 576]],
                        ["GOPR0708_cut1.mp4", [316, 576]],
                        ["GOPR0732_cut1.mp4", [316, 576]],
                        ["GOPR0732_cut2.mp4", [316, 576]],
                        ["GOPR0732_cut3.mp4", [316, 576]]
                        ]
    for video_data in video_library:
        rect = video_data[1]
        video_output = video_data[0][:-4] +"_out.mp4"
        clip1 = VideoFileClip(video_dir + video_data[0])
        video_clip = clip1.fl_image(lambda frame: predict_frame(frame, rect, sess, image_shape, logits, keep_prob, input_image))
        video_clip.write_videofile(video_output, audio=False)

def predict_frame(im, rect, sess, image_shape, logits, keep_prob, image_pl):
    original = im
    roi = im[rect[0]:rect[1],0:720]

    image = scipy.misc.imresize(roi, image_shape)

    im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
    im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
    segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
    mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
    mask = scipy.misc.toimage(mask, mode="RGBA")
    street_im = scipy.misc.toimage(image)
    street_im.paste(mask, box=None, mask=mask)
    upscale_pred = scipy.misc.imresize(street_im, (rect[1]-rect[0],720))
    original[rect[0]:rect[1], 0:720] = upscale_pred
    return original


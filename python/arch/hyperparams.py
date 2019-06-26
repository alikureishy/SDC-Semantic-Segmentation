#!/usr/bin/env python3

import numpy as np

###################################
#           CONSTANTS             #
###################################
L2_REG = 1e-5
STDEV = 1e-2
KEEP_PROB = 0.8
LEARNING_RATE = 1e-4
EPOCHS = 60
BATCH_SIZE = 32
IMAGE_SHAPE_KITI = (160, 576)
NUM_CLASSES = 2

###################################
#           GRAPH NODES           #
###################################
vgg_tag = 'vgg16'
vgg_input_tensor_name = 'image_input:0'
vgg_keep_prob_tensor_name = 'keep_prob:0'
vgg_layer3_out_tensor_name = 'layer3_out:0'
vgg_layer4_out_tensor_name = 'layer4_out:0'
vgg_layer7_out_tensor_name = 'layer7_out:0'

###################################
#           INPUT                 #
###################################
DATA_DIR = '../data'
TEST_DIR = '../data/data_road/testing'
RUNS_DIR = '../runs'
MODELS_DIR = "../models"
MODEL_FILE_PATTERN = "model.ckpt"

#!/usr/bin/env python3

import numpy as np

###################################
#           CONSTANTS             #
###################################
L2_REG = 1e-5
STDEV = 1e-2
KEEP_PROB = 0.8
LEARNING_RATE = 1e-4
EPOCHS = 15
BATCH_SIZE = 8
IMAGE_SHAPE_KITI = (160, 576)
NUM_CLASSES = 2

###################################
#           INPUT                 #
###################################
DATA_DIR = '../data'
RUNS_DIR = '../runs'
MODELS_DIR = "../models"

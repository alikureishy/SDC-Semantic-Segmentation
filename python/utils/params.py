#!/usr/bin/env python3

import numpy as np

###################################
#           IMAGE PARAMS          #
###################################
DEBUG_AUGMENTATION_LEVEL = 0
BACKGROUND_COLOR = np.array([255, 0, 0])
CROP_PROBABILITY = 0.5
FLIP_PROBABILITY = 0.5
RESIZE_PROBABILITY = 0.5
EDIT_PROBABILITY = 0.5
EDIT_CONTRAST_RANGE = [0.85, 1.15]
EDIT_BRIGHTNESS_RANGE = [-45, 30]

###################################
#           VIDEO PARAMS          #
###################################

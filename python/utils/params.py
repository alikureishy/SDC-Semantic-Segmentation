#!/usr/bin/env python3

import numpy as np

##################################
#           DEBUGGING            #
##################################
DEBUG_AUGMENTATION_LEVEL = 0
DEBUG_INFERENCE_LEVEL = 0
DEBUG_MODEL = True
DEBUG_DIR = '../debug_keep'

###################################
#           IMAGE PARAMS          #
###################################
COLOR_FOR_ONLY_NON_ROAD_PIXELS = np.array([255, 0, 0])
CROP_PROBABILITY = 0.5
FLIP_PROBABILITY = 0.5
RESIZE_PROBABILITY = 0.0
EDIT_PROBABILITY = 0.7
EDIT_CONTRAST_RANGE = [0.40, 1.60] #[0.85, 1.15]
EDIT_BRIGHTNESS_RANGE = [-100, 100] #[-45, 30]


NON_ROAD_RGBA_FOR_ILLUSTRATION = np.array([0, 0, 0, 0])                 # Blank color (alpha = 0)
ROAD_RGBA_FOR_ILLUSTRATION = np.array([0, 127, 0, 1])                   # Mask color (alpha = 1) - RED
LABEL_TO_MASK_TRANSFORM = np.vstack((NON_ROAD_RGBA_FOR_ILLUSTRATION,    # 2 x 3 matrix
                                     ROAD_RGBA_FOR_ILLUSTRATION))       # Yields a (?, ?, 4) matrix from
                                                                        #   labeled image (?, ?, 2) DOT transform (2, 4)

###################################
#           VIDEO PARAMS          #
###################################

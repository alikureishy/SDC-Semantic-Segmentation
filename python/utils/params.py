#!/usr/bin/env python3

import numpy as np

###################################
#           IMAGE PARAMS          #
###################################
DEBUG_AUGMENTATION_LEVEL = 0
DEBUG_SHAPES = False
COLOR_FOR_ONLY_NON_ROAD_PIXELS = np.array([255, 0, 0])
CROP_PROBABILITY = 0.5
FLIP_PROBABILITY = 0.5
RESIZE_PROBABILITY = 0.5
EDIT_PROBABILITY = 0.5
EDIT_CONTRAST_RANGE = [0.85, 1.15]
EDIT_BRIGHTNESS_RANGE = [-45, 30]


NON_ROAD_RGBA_FOR_ILLUSTRATION = np.array([0, 0, 0, 0])                 # Blank color (alpha = 0)
ROAD_RGBA_FOR_ILLUSTRATION = np.array([0, 127, 0, 1])                   # Mask color (alpha = 1)
LABEL_TO_MASK_TRANSFORM = np.vstack((NON_ROAD_RGBA_FOR_ILLUSTRATION,    # 2 x 3 matrix
                                     ROAD_RGBA_FOR_ILLUSTRATION))       # Yields a (?, ?, 4) matrix from
                                                                        #   labeled image (?, ?, 2) DOT transform (2, 4)

###################################
#           VIDEO PARAMS          #
###################################

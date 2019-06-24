#!/usr/bin/env python3

import warnings
from distutils.version import LooseVersion
import tensorflow as tf

# Check TensorFlow Version
def check_tensorflow_version():
    assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
    print('TensorFlow Version: {}'.format(tf.__version__))

def check_gpu():
    # Check for a GPU
    assert tf.test.is_gpu_available(), 'No GPU found. Please use a GPU to train your neural network.'
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))



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

def plot_loss(runs_dir, loss, folder_name):
    _, axes = plt.subplots()
    plt.plot(range(0, len(loss)), loss)
    plt.title('Cross-entropy loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    if os.path.exists(runs_dir):
        shutil.rmtree(runs_dir)
    os.makedirs(runs_dir)

    output_file = os.path.join(runs_dir, folder_name + ".png")
    plt.savefig(output_file)

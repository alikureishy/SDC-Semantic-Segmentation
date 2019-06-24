#!/usr/bin/env python3

import os.path
from datetime import timedelta
import tensorflow as tf
from utils.input import *
from hyperparams import *
from tests import project_tests as tests
from utils.gpu import *

###################################
#           INPUT                 #
###################################
DATA_DIR = '../data'
RUNS_DIR = '../runs'
MODELS_DIR = "../models"

check_tensorflow_version()
check_gpu()

def run(model_file=None, reload_model=True):
    num_classes = 2
    image_shape = IMAGE_SHAPE_KITI  # KITTI dataset uses 160x576 images

    with tf.Session() as sess:
        maybe_download_pretrained_vgg(DATA_DIR)

        # Path to vgg model
        vgg_path = os.path.join(DATA_DIR, 'vgg')

        sess.run(tf.global_variables_initializer())

        # TODO: Save inference data using helper.save_inference_samples
        print ("Saving inferrence samples...", end="")
        save_inference_samples(RUNS_DIR, DATA_DIR, sess, image_shape, logits, keep_prob, input_layer)
        print ("...DONE")

        if model_file is not None:
            # builder = tf.saved_model.builder.SavedModelBuilder()
            # builder.add_meta_graph_and_variables()
            save_path = saver.save(sess, os.path.join(MODELS_DIR, model_file))
            print ("Model saved to file: " + save_path)
        else:
            print ("No model file provided. This was a dry run.")

if __name__ == '__main__':
    run(reload_model=False, model_file="Semantic_seg_trained.ckpt")

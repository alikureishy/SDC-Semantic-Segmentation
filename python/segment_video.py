#!/usr/bin/env python3
import tensorflow as tf
import input_utils
import warnings
from distutils.version import LooseVersion

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.is_gpu_available():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


###################################
#           CONSTANTS             #
###################################
BATCH_SIZE = 8
IMAGE_SHAPE_KITI = (160, 576)
NUM_CLASSES = 2

###################################
#           INPUT                 #
###################################
DATA_DIR = './data'
RUNS_DIR = './runs'
MODELS_DIR = "./models"


def run():
    num_classes = 2
    image_shape = IMAGE_SHAPE_KITI  # KITTI dataset uses 160x576 images

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # TODO: Save inference data using helper.save_inference_samples
        input_utils.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_layer)

        # OPTIONAL: Apply the trained model to a video
        predict_video(sess, image_shape, logits, keep_prob, input_layer)
        input_utils.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_layer)
        save_path = tf.train.Saver().save(sess, MODELS_DIR + "Semantic_seg_trained.ckpt")

if __name__ == '__main__':
    run()

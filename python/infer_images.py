#!/usr/bin/env python3

import scipy.misc
import tensorflow as tf
import time
import os.path
from glob import glob
from tests import project_tests as tests
from arch.model import build_model
from arch.hyperparams import *
from utils.gpu import check_gpu
import shutil
import argparse

def segment_images(sess, logits, keep_prob, input_image, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep probability
    :param image_pl: TF Placeholder for the image placeholder
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in glob(os.path.join(TEST_DIR, 'image_2', '*.png')):
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

        # Run inference
        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, input_image: [image]})
        # Splice out second column (road), reshape output back to image_shape
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        # If road softmax > 0.5, prediction is road
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        # Create mask based on segmentation to apply to original image
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), np.array(street_im)


def run_inference(sess, image_shape, logits, keep_prob, input_image):
    """
    Save test images with semantic masks of lane predictions to runs_dir.
    :param sess: TF session
    :param image_shape: Tuple - Shape of image
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep probability
    :param input_image: TF Placeholder for the image placeholder
    """
    # Make folder for current run
    print("Running inference and saving output ...")
    output_dir = os.path.join(RUNS_DIR, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    image_outputs = segment_images(sess, logits, keep_prob, input_image, image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)

    print("Inference complete!")

def run(model_file):
    image_shape = IMAGE_SHAPE_KITI  # KITTI dataset uses 160x576 images

    assert model_file is not None, "Model file must be provided for inference to be run"

    with tf.Session() as sess:
        input_layer, keep_prob, correct_label, learning_rate, output_layer, logits, _, _ = build_model(model_file=model_file, reload_model=True, sess=sess)
        run_inference(sess, image_shape, logits, keep_prob, input_layer)

if __name__ == '__main__':
    print("###############################################")
    print("#         IMAGE SEGMENTATION                  #")
    print("###############################################")

    current_time_millis = lambda: int(round(time.time() * 1000))

    parser = argparse.ArgumentParser(description='Image Segmentation Inference')
    parser.add_argument('-o', dest='model_folder', default=current_time_millis(), type=str, help='Location of model on disk')
    args = parser.parse_args()

    check_gpu()

    model_file = os.path.join(MODELS_DIR, args.model_folder, MODEL_FILE_PATTERN);
    run(model_file=model_file)

#!/usr/bin/env python3

import scipy.misc
import tensorflow as tf
import time
import os.path
from glob import glob
from arch.model import build_model
from arch.hyperparams import *
from utils.gpu import check_gpu
from utils.params import *
import imageio
import shutil
import argparse

from tensorflow.python import debug as tf_debug

def paste_road_mask(street_im, im_softmax):

    # If road softmax > 0.5, prediction is road:
    #   For channel 0 (NON-ROAD pixels), this will yield a True for pixels that are more "non-road" than road
    #   For channel 1 (ROAD pixels), this will yield a True for pixels that are more "road" than non-road
    #   Which essentially matches the cell values and meaning of the expected labeled image
    # segmentation = ((im_softmax > 0.5)[0, :, :, :]).asType(np.uint8) # This produces a labeled image of shape (?, ?, 2)
    segmentation = ((im_softmax > 0.5)[0, :, :, :]) # This produces a labeled image of shape (?, ?, 2)

    # Create mask for the road-sections
    mask = np.dot(segmentation, LABEL_TO_MASK_TRANSFORM)
    mask = scipy.misc.toimage(mask, mode="RGBA")

    if DEBUG_INFERENCE_LEVEL >= 1:
        print ("Segmented shape: ", segmentation.shape)
        print ("Transform shape: ", LABEL_TO_MASK_TRANSFORM.shape)
        print ("Mask shape: ", mask.shape)

    street_im.paste(mask, box=None, mask=mask)

    return street_im, mask

# def paste_mask(street_im, im_soft_max, image_shape, color, obj_color_schema):
#     im_soft_max_r = im_soft_max[0][:, color].reshape(image_shape[0], image_shape[1])
#     segmentation_r = (im_soft_max_r > 0.5).reshape(image_shape[0], image_shape[1], 1)
#     mask = np.dot(segmentation_r, np.array(obj_color_schema))
#     mask = scipy.misc.toimage(mask, mode="RGBA")
#     street_im.paste(mask, box=None, mask=mask)
#
#     return street_im


def segment_images(sess, logits, keep_prob, input_image, image_shape, count=None):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep probability
    :param image_pl: TF Placeholder for the image placeholder
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    count = count if count is not None else -1
    counter = 0
    for image_file in glob(os.path.join(TEST_DIR, 'image_2', '*.png')):
        if (counter == count):
            break

        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
        street_im = scipy.misc.toimage(image) # Numpy array -> PIL image

        # Run inference
        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, input_image: [image]})

        im_softmax = im_softmax[0] # list [[1, X, Y, 2]] ==> [1, X, Y, 2] (including the batch row)

        if DEBUG_INFERENCE_LEVEL >= 1:
            print ("---")
            print ("Image shape: " , image.shape)
            print ("Logits shape: ", logits.shape)
            print ("Softmax shape: ", im_softmax.shape)

        street_im, mask = paste_road_mask(street_im, im_softmax)

        counter = counter + 1

        yield os.path.basename(image_file), np.array(street_im), np.array(mask)


def run_inference(sess, image_shape, logits, keep_prob, input_image, count=None):
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
    image_outputs = segment_images(sess, logits, keep_prob, input_image, image_shape, count=count)
    for name, image, mask in image_outputs:
        imageio.imwrite(os.path.join(output_dir, name), image)
        imageio.imwrite(os.path.join(output_dir, "mask-" + name), mask)

    print("Inference complete!")

def run(model_file, count = None):
    image_shape = IMAGE_SHAPE_KITI  # KITTI dataset uses 160x576 images

    assert model_file is not None, "Model file must be provided for inference to be run"

    with tf.Session() as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        input_layer, keep_prob, correct_label, learning_rate, output_layer, logits, _, _ = build_model(model_file=model_file, reload_model=True, sess=sess)
        run_inference(sess, image_shape, logits, keep_prob, input_layer, count=count)

if __name__ == '__main__':
    print("###############################################")
    print("#         IMAGE SEGMENTATION                  #")
    print("###############################################")

    current_time_millis = lambda: int(round(time.time() * 1000))

    parser = argparse.ArgumentParser(description='Image Segmentation Inference')
    parser.add_argument('-o', dest='model_folder', default=current_time_millis(), type=str, help='Location of model on disk')
    parser.add_argument('-n', dest='count', default=None, type=int, help='Number of images to segment (default: All)')
    args = parser.parse_args()

    check_gpu()

    model_file = os.path.join(MODELS_DIR, args.model_folder, MODEL_FILE_PATTERN);
    run(model_file=model_file, count=args.count)

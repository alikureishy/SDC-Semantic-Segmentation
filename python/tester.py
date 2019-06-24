#!/usr/bin/env python3

import scipy.misc
from scipy import ndimage
import tensorflow as tf
import time
import os.path
from glob import glob
import numpy as np
from tests import project_tests as tests
from model import layers, load_vgg, optimize, train_nn
from hyperparams import *
import shutil
from utils.input import maybe_download_pretrained_vgg

tests.test_load_vgg(load_vgg, tf)
print ("VGG tests complete")

tests.test_layers(layers)
print ("Network layers test complete")

tests.test_optimize(optimize)
print ("Optimizer test complete...")

tests.test_train_nn(train_nn)
print ("Training test complete...")

def run_samples_inference(sess, logits, keep_prob, image_pl, data_folder, image_shape):
	"""
	Generate test output using the test images
	:param sess: TF session
	:param logits: TF Tensor for the logits
	:param keep_prob: TF Placeholder for the dropout keep probability
	:param image_pl: TF Placeholder for the image placeholder
	:param data_folder: Path to the folder that contains the datasets
	:param image_shape: Tuple - Shape of image
	:return: Output for for each test image
	"""
	for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
		image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

		# Run inference
		im_softmax = sess.run(
			[tf.nn.softmax(logits)],
			{keep_prob: 1.0, image_pl: [image]})
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


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
	"""
	Save test images with semantic masks of lane predictions to runs_dir.
	:param runs_dir: Directory to save output images
	:param data_dir: Path to the directory that contains the datasets
	:param sess: TF session
	:param image_shape: Tuple - Shape of image
	:param logits: TF Tensor for the logits
	:param keep_prob: TF Placeholder for the dropout keep probability
	:param input_image: TF Placeholder for the image placeholder
	"""
	# Make folder for current run
	output_dir = os.path.join(runs_dir, str(time.time()))
	if os.path.exists(output_dir):
		shutil.rmtree(output_dir)
	os.makedirs(output_dir)

	# Run NN on test images and save them to HD
	print('Training Finished. Saving test images to: {}'.format(output_dir))
	image_outputs = run_samples_inference(sess, logits, keep_prob, input_image, os.path.join(data_dir, 'data_road/testing'), image_shape)
	for name, image in image_outputs:
		scipy.misc.imsave(os.path.join(output_dir, name), image)

	print("Saving complete!")

def run(model_file=None):
    image_shape = IMAGE_SHAPE_KITI  # KITTI dataset uses 160x576 images

	assert model_file is not None, "Model file must be provided for inference to be run"

    with tf.Session() as sess:
        # Load VGG
        maybe_download_pretrained_vgg(DATA_DIR)
        vgg_path = os.path.join(DATA_DIR, 'vgg')
        input_layer, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)

        saver = tf.train.Saver()
		output_layer = layers(layer3, layer4, layer7, NUM_CLASSES)
		correct_label = tf.placeholder(dtype=tf.float32, shape=(None, None, None, NUM_CLASSES), name='correct_label')
		learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')
		logits, train_op, cross_entropy_loss = optimize(output_layer, correct_label, learning_rate, NUM_CLASSES)

		# TODO: Train NN using the train_nn function
		print ("Training the network...", end="")
		sess.run(tf.global_variables_initializer())
		train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, cross_entropy_loss, input_layer, correct_label, keep_prob, learning_rate)

		# TODO: Save inference data using helper.save_inference_samples
		print ("Saving inferrence samples...", end="")
		save_inference_samples(RUNS_DIR, DATA_DIR, sess, image_shape, logits, keep_prob, input_layer)

if __name__ == '__main__':
    run(reload_model=False, model_file="Semantic_seg_trained.ckpt")

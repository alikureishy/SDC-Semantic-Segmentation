#!/usr/bin/env python3

'''
You should not edit input.py as part of your submission.

This file is used primarily to download vgg if it has not yet been,
give you the progress of the download, get batches for your training,
as well as around generating and saving the image outputs.
'''

import re
import os.path
import scipy.misc
from glob import glob
from utils.media import *
from utils.params import *

def prob_choice(threshold):
	return random.uniform(0, 1) > threshold

def randomly_adjust_shape(image, segmented_image):
	# Crop (with probability = CROP_PROBABILITY):
	if prob_choice(CROP_PROBABILITY):
		image, segmented_image = crop_image(image, segmented_image)
		if DEBUG_AUGMENTATION_LEVEL >= 2:
			print("\t\tCrop:")
			print("\t\t\t=> Image: (" + str(image.shape) + ") / GT_Image: (" + str(segmented_image.shape) + ")")

	return image, segmented_image

def randomly_adjust_content(image, segmented_image):
	# Flip (with probability = FLIP_PROBABILITY):
	if prob_choice(FLIP_PROBABILITY):
		image, segmented_image = flip_image(image, segmented_image)
		if DEBUG_AUGMENTATION_LEVEL >= 2:
			print("\t\tFlip:")
			print("\t\t\t=> Image: (" + str(image.shape) + ") / Segmented: (" + str(segmented_image.shape) + ")")

	# Change contrast (with probability = SUNLIGHT_PROBABILITY):
	if prob_choice(EDIT_PROBABILITY):
		contrast = random.uniform(*EDIT_CONTRAST_RANGE)  # Contrast augmentation
		brightness = random.randint(*EDIT_BRIGHTNESS_RANGE)  # Brightness augmentation
		image = bc_img(image, contrast, brightness)

	return image, segmented_image

def adjust_to_target_shape(image, segmented_image, target_shape):
	# Resize (with probability = RESIZE_PROBABILITY)
	# image, gt_image = cv2.resize(image, (image_shape[1], image_shape[0])), cv2.resize(gt_image, (image_shape[1], image_shape[0]))
	image, segmented_image = scipy.misc.imresize(image, target_shape), scipy.misc.imresize(segmented_image, target_shape)
	if DEBUG_AUGMENTATION_LEVEL >= 2:
		print("\t\tResize:")
		print("\t\t\t => Image: (" + str(image.shape) + ") / Segmented: (" + str(segmented_image.shape) + ")")

	return image, segmented_image

def convert_to_labels(segmented_image):
	background = np.all(segmented_image == BACKGROUND_COLOR, axis=2)
	background = background.reshape(*background.shape, 1)
	background_inverted = np.invert(background)
	labeled_image = np.concatenate((background, background_inverted), axis=2)
	if DEBUG_AUGMENTATION_LEVEL >= 2:
		print("\t\tBackground: " + str(background.shape) + ") ++ (" + str(background_inverted.shape) + ") ")
		print("\t\t\t=> Labeled: (" + str(labeled_image.shape) + ")")
	return labeled_image

#
# Augment images as follows:
# - Flip
# - Brightness/contrast
# -
def gen_batch_function(data_folder, image_shape):
	"""
	Generate function to create batches of training data
	:param data_folder: Path to folder that contains all the datasets
	:param image_shape: Tuple - Shape of image
	:return:
	"""
	def get_batches_fn(batch_size):
		"""
		Create batches of training data
		:param batch_size: Batch Size
		:return: Batches of training data
		"""
		# Grab image and label paths
		image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
		label_paths = {
			re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path for path in
				glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))
		}

		# Shuffle training data
		random.shuffle(image_paths)

		# Loop through batches and grab images, yielding each batch
		for batch_i in range(0, len(image_paths), batch_size):
			images = []
			segmented_images = []
			for image_file in image_paths[batch_i:batch_i+batch_size]:
				segmented_image_file = label_paths[os.path.basename(image_file)]

				# Read image:
				image, segmented_image = scipy.misc.imread(image_file), scipy.misc.imread(segmented_image_file)
				# image, segmented_image = cv2.imread(image_file), cv2.imread(segmented_image_file)
				if DEBUG_AUGMENTATION_LEVEL >= 1:
					print ("\tImage: (" + str(image.shape) + ") / Segmented: (" + str(segmented_image.shape) + ")")

				image, segmented_image = randomly_adjust_shape(image, segmented_image)
				image, segmented_image = randomly_adjust_content(image, segmented_image)
				image, segmented_image = adjust_to_target_shape(image, segmented_image, image_shape)

				# Create "one-hot-like" labels by channel
				segmented_with_labels = convert_to_labels(segmented_image)

				images.append(image)
				segmented_images.append(segmented_with_labels)

			yield np.array(images), np.array(segmented_images)

	return get_batches_fn


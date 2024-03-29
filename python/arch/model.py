#!/usr/bin/env python3

from utils.gpu import *
import time
import os
import os.path
import glob
from arch.hyperparams import *
from datetime import timedelta
import shutil
import zipfile
from urllib.request import urlretrieve
from tqdm import tqdm
from utils.params import *
from utils.plot import plot_loss

# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
# import logging
# logging.getLogger("tensorflow").setLevel(logging.ERROR)
# logging.getLogger("scipy").setLevel(logging.ERROR)

tf.logging.set_verbosity(tf.logging.ERROR)

check_tensorflow_version()

class DLProgress(tqdm):
    """
    Report download progress to the terminal.
    :param tqdm: Information fed to the tqdm library to estimate progress.
    """
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        """
        Store necessary information for tracking progress.
        :param block_num: current block of the download
        :param block_size: size of current block
        :param total_size: total download size, if known
        """
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)  # Updates progress
        self.last_block = block_num

def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   (DONE) Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag],vgg_path)
    graph = tf.get_default_graph()

    w1 = graph.get_tensor_by_name(vgg_input_tensor_name)

    # w1.set_shape((None, 160, 576, 3))

    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    if DEBUG_MODEL:
        print ("VGG:")
        print ("\tInput: ", w1.shape)
        print ("\tKeep: ", keep.shape)
        print ("\tLayer 3: ", layer3_out.shape)
        print ("\tLayer 4: ", layer4_out.shape)
        print ("\tLayer 7: ", layer7_out.shape)

    return w1, keep, layer3_out, layer4_out, layer7_out

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function

    ######################################################################
    # Prepare the Skip layers by controlling number of filters put out:
    ######################################################################
    # Reduce dimensionality from VGG's last convolutional layer:
    layer7_1x1_out = tf.layers.conv2d(
                    inputs=vgg_layer7_out,
                    filters=256,    # Cap at 256
                    kernel_size=1,
                    padding='same',
                    kernel_initializer=tf.random_normal_initializer(stddev=STDEV), #0.01
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG)) #1e-3
    # 1x1 convolution of vgg layer 4
    layer4_1x1_out = tf.layers.conv2d(
                    inputs=vgg_layer4_out,
                    filters=128,    # Cap at 128
                    kernel_size=1,
                    padding='same',
                    kernel_initializer=tf.random_normal_initializer(stddev=STDEV),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG))
    # 1x1 convolution of vgg layer 3
    layer3_1x1_out = tf.layers.conv2d(
                    inputs=vgg_layer3_out,
                    filters=64,     # Cap at 64
                    kernel_size=1,
                    padding='same',
                    kernel_initializer=tf.random_normal_initializer(stddev=STDEV),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG))


    ##################################
    # Prepare the deconvolution layers
    ##################################
    layer_4_deconv = tf.layers.conv2d_transpose(
                    inputs=layer7_1x1_out, # Or do 1x1 before this on that output?
                    filters=128,
                    kernel_size=4,
                    strides=(2, 2),
                    padding='same',
                    activation="relu",
                    kernel_initializer=tf.random_normal_initializer(stddev=STDEV),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG))
    # layer4_skip = tf.add(x=layer4_1x1_out, y=layer_4_deconv)
    layer4_skip = tf.concat(axis=len(layer4_1x1_out.shape)-1, values=[layer4_1x1_out, layer_4_deconv])


    layer_3_deconv = tf.layers.conv2d_transpose(
                    inputs=layer4_skip,
                    filters=64,
                    kernel_size=4,
                    strides=(2, 2),
                    padding='same',
                    activation="relu",
                    kernel_initializer=tf.random_normal_initializer(stddev=STDEV),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG))
    # layer3_skip = tf.add(x=layer3_1x1_out, y=layer_3_deconv)
    layer3_skip = tf.concat(axis=len(layer3_1x1_out.shape)-1, values=[layer3_1x1_out, layer_3_deconv])


    output = tf.layers.conv2d_transpose(
                    inputs=layer3_skip,
                    filters=num_classes,
                    kernel_size=8,
                    strides=(8, 8),
                    padding='same',
                    kernel_initializer=tf.random_normal_initializer(stddev=STDEV),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG))

    return output

def build_model(model_file, reload_model, sess):



    maybe_download_pretrained_vgg(DATA_DIR)
    vgg_path = os.path.join(DATA_DIR, 'vgg')
    vgg_path = os.path.join(DATA_DIR, 'vgg')
    input_layer, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
    output_layer = layers(layer3, layer4, layer7, NUM_CLASSES)
    correct_label = tf.placeholder(dtype=tf.float32, shape=(None, None, None, NUM_CLASSES), name='correct_label')
    learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')

    # Reshape:
    logits = output_layer
    # logits = tf.reshape(output_layer, (-1, NUM_CLASSES))
    # correct_label = tf.reshape(correct_label, (-1, NUM_CLASSES))

    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss);

    # Load the model variables into the model:
    if reload_model and model_exists(model_file):
        saver = load_model(model_file, sess)
        print("Reloading model from file...")
    else:
        print("Initialized fresh model...")
        sess.run(tf.global_variables_initializer())

    return input_layer, keep_prob, correct_label, learning_rate, output_layer, logits, train_op, cross_entropy_loss

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    losses = []
    for epoch in range(epochs):
        batch_loss = 0.0
        s_time = time.time()
        for image, labeled in get_batches_fn(batch_size):
            _, loss = sess.run(
                    [train_op, cross_entropy_loss],
                    feed_dict={input_image: image,
                               correct_label: labeled,
                               keep_prob: KEEP_PROB,
                               learning_rate: LEARNING_RATE}
            )
            batch_loss += loss
        losses.append(batch_loss)
        print("[Epoch: {0}/{1} Loss: {2:4f} Time: {3}]".format(epoch + 1, epochs, batch_loss,
                                                               str(timedelta(seconds=(time.time() - s_time)))))
        plot_loss(RUNS_DIR, losses, "loss_graph")

    pass

def save_model(sess, model_file):
    saver = tf.train.Saver()
    # builder = tf.saved_model.builder.SavedModelBuilder()
    # builder.add_meta_graph_and_variables()
    save_path = saver.save(sess, os.path.join(MODELS_DIR, model_file))

    return save_path

def load_model(model_file, sess):
    assert model_exists(model_file), "Checkpoint asset provided does not exist"
    saver = tf.train.Saver()
    saver.restore(sess, os.path.join(MODELS_DIR, model_file))
    return saver

def model_exists(model_file):
    return len(glob.glob(model_file + "*")) > 0



# # 1x1 convolution layer (to replaace the fully connected layer) : To reduce VGG-16's 4096 output classes/filters to a smaller number
# layer = conv_1x1 = tf.layers.conv2d(inputs=vgg_layer7_out, filters=256, kernel_size=1, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
#
# # Deconvolution layers:
#
# # Skip connection here:
# layer = deconv7_out = tf.layers.conv2d_transpose(inputs=layer, filters=num_classes, kernel_size=4, strides=2, padding='same', name="deconv7_out")
# layer = tf.add(layer, vgg_layer7_out)
# # layer = tf.concat((layer, vgg_layer7_out), axis=1)
#
# # Skip connection here:
# layer = deconv4_out = tf.layers.conv2d_transpose(inputs=layer, filters=num_classes, kernel_size=64, strides=32, padding='same', name="deconv4_out")
# layer = tf.add(layer, vgg_layer4_out)
# # layer = tf.concat((layer, vgg_layer4_out), axis=0)
#
# # Skip connection here:
# layer = deconv3_out = tf.layers.conv2d_transpose(inputs=layer, filters=num_classes, kernel_size=128, strides=64, padding='same', name="deconv3_out")
# layer = tf.add(layer, vgg_layer3_out)
# # layer = tf.concat((layer, vgg_layer3_out), axis=0)
#
# # No skip connections here:
# layer = deconv6_out = tf.layers.conv2d_transpose(inputs=layer, filters=num_classes, kernel_size=16, strides=8, padding='same', name="deconv6_out")
# layer = deconv5_out = tf.layers.conv2d_transpose(inputs=layer, filters=num_classes, kernel_size=32, strides=16, padding='same', name="deconv5_out")
#
# layer = tf.nn.dropout(layer, keep_prob=0.75)


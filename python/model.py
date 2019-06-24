#!/usr/bin/env python3

from utils.gpu import *
import time
import os.path
import glob
from hyperparams import *
from datetime import timedelta

check_tensorflow_version()
check_gpu()

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
    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

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

    # Reduce dimensionality from VGG's last convolutional layer:
    layer7a_out = tf.layers.conv2d(
                    inputs=vgg_layer7_out,
                    filters=64,
                    kernel_size=1,
                    padding='same',
                    kernel_initializer=tf.random_normal_initializer(stddev=STDEV), #0.01
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG)) #1e-3

    # upsample
    layer4a_in1 = tf.layers.conv2d_transpose(
                    inputs=layer7a_out,
                    filters=32,
                    kernel_size=4,
                    strides=(2, 2),
                    padding='same',
                    kernel_initializer=tf.random_normal_initializer(stddev=STDEV),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG))

    # 1x1 convolution of vgg layer 4
    layer4a_in2 = tf.layers.conv2d(
                    inputs=vgg_layer4_out,
                    filters=32,
                    kernel_size=1,
                    padding='same',
                    kernel_initializer=tf.random_normal_initializer(stddev=STDEV),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG))

    # skip connection (element-wise addition)
    layer4a_out = tf.add(x=layer4a_in1, y=layer4a_in2)
    # upsample
    layer3a_in1 = tf.layers.conv2d_transpose(
                    inputs=layer4a_out,
                    filters=16,
                    kernel_size=4,
                    strides=(2, 2),
                    padding='same',
                    kernel_initializer=tf.random_normal_initializer(stddev=STDEV),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG))

    # 1x1 convolution of vgg layer 3
    layer3a_in2 = tf.layers.conv2d(
                    inputs=vgg_layer3_out,
                    filters=16,
                    kernel_size=1,
                    padding='same',
                    kernel_initializer=tf.random_normal_initializer(stddev=STDEV),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG))

    # skip connection (element-wise addition)
    layer3a_out = tf.add(x=layer3a_in1, y=layer3a_in2)
    # upsample
    nn_last_layer = tf.layers.conv2d_transpose(
                    inputs=layer3a_out,
                    filters=num_classes,
                    kernel_size=8,
                    strides=(8, 8),
                    padding='same',
                    kernel_initializer=tf.random_normal_initializer(stddev=STDEV),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG))

    return nn_last_layer

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1,num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss);

    return logits, train_op, cross_entropy_loss

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
        loss = None
        s_time = time.time()
        for image, labels in get_batches_fn(batch_size):
            _, loss = sess.run(
                    [train_op, cross_entropy_loss],
                    feed_dict={input_image: image,
                               correct_label: labels,
                               keep_prob: KEEP_PROB,
                               learning_rate: LEARNING_RATE}
            )
            losses.append(loss)
        print("[Epoch: {0}/{1} Loss: {2:4f} Time: {3}]".format(epoch + 1, epochs, loss,
                                                               str(timedelta(seconds=(time.time() - s_time)))))
    # helper.plot_loss(RUNS_DIR, losses, "loss_graph")

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



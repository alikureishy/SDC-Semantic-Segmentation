#!/usr/bin/env python3

import os.path
from utils.input import *
from hyperparams import *
from tests import project_tests as tests
from utils.gpu import *
from model import layers, load_vgg, optimize, train_nn, load_model, save_model, model_exists


def run(model_file=None, reload_model=True):
    num_classes = 2
    image_shape = IMAGE_SHAPE_KITI  # KITTI dataset uses 160x576 images
    tests.test_for_kitti_dataset(DATA_DIR)

    assert model_file is not None, "A model file name must be provided"

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        maybe_download_pretrained_vgg(DATA_DIR)
        vgg_path = os.path.join(DATA_DIR, 'vgg')
        input_layer, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)

        # Create function to get batches
        get_batches_fn = gen_batch_function(os.path.join(DATA_DIR, 'data_road/training'), image_shape)

        output_layer = layers(layer3, layer4, layer7, NUM_CLASSES)
        correct_label = tf.placeholder(dtype=tf.float32, shape=(None, None, None, NUM_CLASSES), name='correct_label')
        learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')
        logits, train_op, cross_entropy_loss = optimize(output_layer, correct_label, learning_rate, NUM_CLASSES)

        # Load the model variables into the model:
        if reload_model and model_exists(model_file):
            saver = load_model(model_file, sess)
            print("Continuing training from model file...")
        else:
            print("Training from scratch...")

        # Train the NN:
        print ("Training the network...")
        sess.run(tf.global_variables_initializer())
        train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, cross_entropy_loss, input_layer, correct_label, keep_prob, learning_rate)

        save_path = save_model(sess, model_file)

        print ("Model saved to file: " + save_path)

if __name__ == '__main__':
    check_gpu()
    check_tensorflow_version()

    run(reload_model=True, model_file="Semantic_seg_trained.ckpt")

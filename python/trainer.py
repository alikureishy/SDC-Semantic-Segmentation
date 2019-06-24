#!/usr/bin/env python3

import os.path
from utils.input import *
from arch.hyperparams import *
from tests import project_tests as tests
from utils.gpu import *
from arch.model import train_nn, save_model, build_model


def run(model_file=None, reload_model=True):
    image_shape = IMAGE_SHAPE_KITI  # KITTI dataset uses 160x576 images
    tests.test_for_kitti_dataset(DATA_DIR)

    assert model_file is not None, "A model file name must be provided"

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Create function to get batches
        get_batches_fn = gen_batch_function(os.path.join(DATA_DIR, 'data_road/training'), image_shape)

        correct_label, cross_entropy_loss, input_layer, keep_prob, learning_rate, train_op = build_model(model_file,
                                                                                                         reload_model,
                                                                                                         sess)

        # Train the NN:
        print ("Training the network...")
        train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, cross_entropy_loss, input_layer, correct_label, keep_prob, learning_rate)

        save_path = save_model(sess, model_file)

        print ("Model saved to file: " + save_path)


if __name__ == '__main__':
    check_gpu()
    check_tensorflow_version()

    run(reload_model=True, model_file="Semantic_seg_trained.ckpt")

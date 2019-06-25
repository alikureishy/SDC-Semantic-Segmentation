#!/usr/bin/env python3

import os.path
import time
from utils.input import *
from arch.hyperparams import *
from tests import project_tests as tests
from utils.gpu import *
from arch.model import train_nn, save_model, build_model, optimize
import argparse

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

        input_layer, keep_prob, correct_label, learning_rate, output_layer, logits = build_model(model_file, reload_model, sess)
        train_op, cross_entropy_loss = optimize(logits, correct_label, learning_rate, NUM_CLASSES)

        # Train the NN:
        print ("Training the network...")
        train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, cross_entropy_loss, input_layer, correct_label, keep_prob, learning_rate)

        save_path = save_model(sess, model_file)

        print ("Model saved to file: " + save_path)


if __name__ == '__main__':
    print("###############################################")
    print("#                    TRAINER                  #")
    print("###############################################")

    current_time_millis = lambda: int(round(time.time() * 1000))

    parser = argparse.ArgumentParser(description='Trainer')
    parser.add_argument('-r', dest='reload', action='store_true', default=False, help='Will load the model from disk if available (default: false).')
    parser.add_argument('-o', dest='model_folder', default=current_time_millis(), type=str, help='Location of model on disk')
    args = parser.parse_args()

    check_gpu()

    model_folder = os.path.join(MODELS_DIR, args.model_folder);
    os.makedirs(model_folder, exist_ok=True);
    model_file = os.path.join(model_folder, MODEL_FILE_PATTERN)
    run(reload_model=args.reload, model_file=model_file)

###############################################################
# parser.add_argument('-i', dest='input', required=True, type=str,
#                     help='Path to image file, image folder, or video file.')
# parser.add_argument('-o', dest='output', type=str,
#                     help='Location of output (Will be treated as the same as input type)')
# parser.add_argument('-c', dest='configs', required=True, nargs='*', type=str, help="Configuration files.")
# parser.add_argument('-s', dest='selector', type=int,
#                     help='Short circuit the pipeline to perform only specified # of operations.')
# parser.add_argument('-x', dest='speed', type=int, default=1,
#                     help='Speed (1 ,2, 3 etc) interpreted as 1x, 2x, 3x etc)')
# parser.add_argument('-r', dest='range', nargs='*', type=int, default=None,
#                     help='Range of frames to process (default: None)')
# parser.add_argument('-d', dest='dry', action='store_true',
#                     help='Dry run. Will not save anything to disk (default: false).')
# parser.add_argument('-p', dest='plot', action='store_true',
#                     help='Plot all illustrations marked as \'ToPlot\' in the config. (default: false).')

# eval:

import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras, sys, time, warnings
from keras.layers import *
import pandas as pd
import argparse
import ntpath
#
import numpy as np
from models_gby import load_model_gby
from datasets_gby import load_dataset
import matplotlib.pyplot as plt
from utils_gby import IoU_ver2,give_color_to_seg_img
import pdb

def argument_parser_eval():
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('-m', '--model', default='fcn_RESNET50_8s', help='choose between \'fcn_VGG16_32s\',\'fcn_VGG16_8s\',\'fcn_RESNET50_32s\', and \'fcn_RESNET50_8s\' networks, with or without \'_crfrnn\' suffix', type=str)
    parser.add_argument('-w', '--weights', default=None, nargs='?', const=None, help='The absolute path of the weights',type=str)
    parser.add_argument('-ds', '--dataset', default='streets', help='The name of train/test sets', type=str)
    parser.add_argument('-vb', '--verbosemode', default=1, help='Specify the verbose mode',type=int)
    parser.add_argument('-is', '--inputsize', default=512, help='Specify the input size N, where N=rows,N=cols. ',type=int)
    return parser.parse_args()

# ===========================
# Main
# ===========================
if __name__ == '__main__':

    # ===============
    # INTRO
    # ===============

    # Parse args:
    # -----------
    args = argument_parser_eval()

    # Import Keras and Tensorflow to develop deep learning FCN models:
    # -----------------------------------------------------------------
    warnings.filterwarnings("ignore")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.70 #0.95
    config.gpu_options.visible_device_list = "2"
    set_session(tf.Session(config=config))

    print("python {}".format(sys.version))
    print("keras version {}".format(keras.__version__)) #; del keras
    print("tensorflow version {}".format(tf.__version__))

    # -----------------------------------------
    # ===============
    # LOAD train data:
    # ===============

    INPUT_SIZE = args.inputsize  # #500 #224 #512 # NOTE: Extract from model

    dataaug_args = None
    ds = load_dataset(args.dataset, INPUT_SIZE, dataaug_args)
    print(ds.X_train.shape, ds.y_train.shape)
    print(ds.X_test.shape, ds.y_test.shape)
    nb_classes = ds.nb_classes

    num_crf_iterations = 10  # at test time

    # ===============
    # LOAD model:
    # ===============

    model_name = args.model
    model_path_name = args.weights

    print('====================================================================================')
    print(model_path_name)
    print('====================================================================================')

    model = load_model_gby(model_name, INPUT_SIZE, nb_classes, num_crf_iterations)

    #loading weights:
    model.load_weights(model_path_name)

    batchsize = 6
    if model.crf_flag:
        batchsize = 1

    # ===============
    # ANALYZE model:
    # ===============

    # Compute IOU:
    # ------------
    print('computing mean IoU for validation set..')
    y_pred = model.predict(ds.X_test,batch_size=batchsize, verbose=1)
    y_predi = np.argmax(y_pred, axis=3)
    y_testi = np.argmax(ds.y_test, axis=3)
    print(y_testi.shape, y_predi.shape)
    IoU_ver2(y_testi, y_predi)

    # Visualize the model performance:
    # --------------------------------
    shape = (INPUT_SIZE, INPUT_SIZE)
    n_classes = nb_classes # 10

    num_examples_to_plot = 4

    fig = plt.figure(figsize=(10, 3*num_examples_to_plot))

    for i in range(num_examples_to_plot):

        img_indx = i*4
        img_is = (ds.X_test[img_indx] + 1) * (255.0 / 2)
        seg = y_predi[img_indx]
        segtest = y_testi[img_indx]

        ax = fig.add_subplot(num_examples_to_plot, 3, 3 * i + 1)
        ax.imshow(img_is / 255.0)
        if i == 0:
            ax.set_title("original")

        ax = fig.add_subplot(num_examples_to_plot, 3, 3 * i + 2)
        ax.imshow(give_color_to_seg_img(seg, n_classes))
        if i == 0:
            ax.set_title("predicted class")

        ax = fig.add_subplot(num_examples_to_plot, 3, 3 * i + 3)
        ax.imshow(give_color_to_seg_img(segtest, n_classes))
        if i == 0:
            ax.set_title("true class")

    plt.savefig('examples_%s.png' % ntpath.basename(model_path_name))

    # clear model: Destroys the current TF graph and creates a new one. Useful to avoid clutter from old models / layers.
    keras.backend.clear_session()

# usage:
# >>python eval_gby.py


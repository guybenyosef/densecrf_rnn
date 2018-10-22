# eval:

import numpy as np
#from keras.callbacks import ModelCheckpoint
#from keras.optimizers import Adam
#from keras.optimizers import SGD
#from keras import backend as K
import matplotlib.pyplot as plt
import pdb
from models_gby import load_model_gby
#fcn_VGG16_32s,fcn_VGG16_32s_crfrnn,fcn_VGG16_8s,fcn_VGG16_8s_crfrnn,fcn_RESNET50_32s,fcn_RESNET50_32s_crfrnn,fcn_RESNET50_8s,fcn_RESNET50_8s_crfrnn
from utils_gby import generate_arrays_from_file,extract_arrays_from_file,IoU,model_predict_gby,getImageArr,getSegmentationArr,IoU_ver2,give_color_to_seg_img

## Import usual libraries
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras, sys, time, warnings
from keras.models import *
from keras.layers import *
import pandas as pd
from sklearn.utils import shuffle
from keras import optimizers
import argparse
import ntpath

# ===========================
# Main
# ===========================
if __name__ == '__main__':

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

    # Parse args:
    # -----------
    # parser = argparse.ArgumentParser()who
    # parser.add_argument('test_data')
    # parser.add_argument('test_data')
    # args = parser.parse_args()

    # Data processing: (for their dataset)
    # -----------------------------------------
    dnns_dir = '/storage/gby/semseg/'
    INPUT_SIZE = 224  # 500 #224
    num_crf_iterations = 10  # at test time
    nb_classes = 12
    dir_img = '/storage/gby/datasets/seg_data_practice/images_prepped_train/'
    dir_seg = '/storage/gby/datasets/seg_data_practice/annotations_prepped_train/'
    images = os.listdir(dir_img)
    images.sort()
    segmentations = os.listdir(dir_seg)
    segmentations.sort()
    #
    X = []
    Y = []
    for im, seg in zip(images, segmentations):
        X.append(getImageArr(dir_img + im, INPUT_SIZE, INPUT_SIZE))
        Y.append(getSegmentationArr(dir_seg + seg, INPUT_SIZE, INPUT_SIZE, nb_classes))
    X, Y = np.array(X), np.array(Y)
    print(X.shape, Y.shape)
    # Split between training and testing data:
    # -----------------------------------------
    train_rate = 0.85
    allow_randomness = False

    if allow_randomness:
        index_train = np.random.choice(X.shape[0], int(X.shape[0] * train_rate), replace=False)
        index_test = list(set(range(X.shape[0])) - set(index_train))
        X, Y = shuffle(X, Y)
        X_train, y_train = X[index_train], Y[index_train]
        X_test, y_test = X[index_test], Y[index_test]
    else:
        index_train = int(X.shape[0] * train_rate)
        X_train, y_train = X[0:index_train], Y[0:index_train]
        X_test, y_test = X[index_train:-1], Y[index_train:-1]

    X, Y = X_test, y_test


    # (for our voc2012 data:)
    # nb_classes = 21
    # print('training data:')
    # [train_imgs,train_labels] = extract_arrays_from_file(args.train_data, args.image_dir, args.label_dir, INPUT_SIZE, nb_classes)
    # #print('validation data:')
    # [val_imgs,val_labels] = extract_arrays_from_file(args.val_data, args.image_dir, args.label_dir, INPUT_SIZE, nb_classes)
    #
    # X_train, y_train = np.array(train_imgs)[:,0,:,:,:], np.array(train_labels)[:,0,:,:,:]
    # X_test, y_test = np.array(val_imgs)[:, 0, :, :, :], np.array(val_labels)[:, 0, :, :, :]
    # #print(X.shape, Y.shape)


    # Constructing model:
    # --------------------
    #names = {'fcn32s_orig','fcn32s','fcn8s'}
    model_names = [
#        'fcn_RESNET50_8s',
#        'fcn_RESNET50_8s',
        'fcn_RESNET50_8s_crfrnn',
        'fcn_RESNET50_8s_crfrnn'
        ]
    # 'fcn_VGG16_8s',
                   # 'fcn_RESNET50_8s',
                   # 'fcn_VGG16_8s',
                   # 'fcn_RESNET50_32s',
                   # 'fcn_VGG16_32s',
                   # 'fcn_VGG16_32s']

    #names = {'fcn8s_Sadeep_crfrnn'}
    #names = {'fcn32s','resnet50fcn32s','fcn8s'}
    #names = {'resnet50fcn32s'}
    model_path_names = [
 #       dnns_dir + 'streets_weights_resnet50fcn8s_1000ep',
 #       dnns_dir + 'streets_weights_resnet50fcn8s_2000ep',
        dnns_dir + 'streets_weights_fcn_RESNET50_8s_crfrnn_2000+1ep',
        dnns_dir + 'streets_weights_fcn_RESNET50_8s_crfrnn_2000+200ep'
        ]

    # dnns_dir+'streets_weights_vgg16fcn8s_100ep',
                        # dnns_dir+'streets_weights_resnet50fcn8s_50ep',
                        # dnns_dir+'streets_weights_vgg16fcn8s_5000ep',
                        # dnns_dir+'streets_weights_resnet50fcn32s_400ep',
                        # dnns_dir+'streets_weights_vgg16fcn32s_100ep',
                        # dnns_dir+'streets_weights_vgg16fcn32s_5000ep']

    #model_path_names = {'/storage/gby/semseg/streets_weights_fcn8s_Sadeep_RNN_50ep'}

    for indx in range(len(model_names)):
        #model_path_name = '/storage/gby/semseg/streets_weights_'+model_name+'_10ep'
      #  model_path_name = '/storage/gby/semseg/streets_weights_' + model_name + '_100ep'
        model_name = model_names[indx]
        model_path_name = model_path_names[indx]

        print('====================================================================================')
        print(model_path_name)
        print('====================================================================================')
        #loading model:
        model = load_model_gby(model_name, INPUT_SIZE, nb_classes, num_crf_iterations)

        #loading weights:
        model.load_weights(model_path_name)

        batchsize = 6
        if model.crf_flag:
            batchsize = 1

        # Compute IOU:
        # ------------
        print('computing mean IoU for validation set..')
        y_pred = model.predict(X,batch_size=batchsize, verbose=1)
        y_predi = np.argmax(y_pred, axis=3)
        y_testi = np.argmax(Y, axis=3)
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
            img_is = (X_test[img_indx] + 1) * (255.0 / 2)
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

        keras.backend.clear_session()
        # clear model: Destroys the current TF graph and creates a new one. Useful to avoid clutter from old models / layers.



# usage:
# >>python eval_gby.py

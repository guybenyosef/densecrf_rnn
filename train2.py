"""
MIT License

"""


import numpy as np
#from keras.callbacks import ModelCheckpoint
#from keras.optimizers import Adam
#from keras.optimizers import SGD
#from keras import backend as K
#import matplotlib.pyplot as plt
import pdb
from models_gby import fcn_32s_orig,fcn_32s,fcn_VGG16_32s_crfrnn,fcn_8s_take2,fcn_VGG16_8s_crfrnn,fcn_RESNET50_32s,fcn_RESNET50_8s,fcn_RESNET50_8s_crfrnn,fcn_8s_Sadeep,fcn_8s_Sadeep_crfrnn
#from crfrnn_model import get_crfrnn_model_def
#from models_gby import fcn_8s_Sadeep_crfrnn
from utils_gby import generate_arrays_from_file,extract_arrays_from_file,IoU,model_predict_gby,getImageArr,getSegmentationArr,IoU_ver2,give_color_to_seg_img,visualize_conv_filters

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
import pickle
from keras.optimizers import Adam
from weighted_categorical_crossentropy import weighted_loss

# Median Frequency Alpha Coefficients
coefficients = {0:0.0237995754847,
                1:0.144286494916,
                2:0.038448897913,
                3:1.33901803472,
                4:1.0,
                5:0.715098627127,
                6:4.20827446939,
                7:1.58754122255,
                8:0.0551054437019,
                9:0.757994265912,
                10:0.218245600783,
                11:0.721125616748
                # 12:6.51048559366,
                # 13:0.125434198729,
                # 14:3.27995580458,
                # 15:3.72813940546,
                # 16:3.76817843552,
                # 17:8.90686657342,
                # 18:2.12162414027,
                # 19:0.
                }
# for python 2.7:
#coefficients = [key for index,key in coefficients.iteritems()]
#python 3:
coefficients = [key for index,key in coefficients.items()]

## location of VGG weights
VGG_Weights_path = "../FacialKeypoint/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"

RES_DIR = "/storage/gby/semseg/"

INPUT_SIZE = 224# #500 #224 #512

# ===========================
# Main
# ===========================
if __name__ == '__main__':

    # Parse args:
    # -----------
    # parser = argparse.ArgumentParser()
    # parser.add_argument('train_data')
    # parser.add_argument('val_data')
    # parser.add_argument('image_dir')
    # parser.add_argument('label_dir')
    # args = parser.parse_args()

    # Import Keras and Tensorflow to develop deep learning FCN models:
    # -----------------------------------------------------------------
    warnings.filterwarnings("ignore")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.70 # default: "0.95"
    config.gpu_options.visible_device_list = "2" # default: "2"
    set_session(tf.Session(config=config))

    print("python {}".format(sys.version))
    print("keras version {}".format(keras.__version__)) #; del keras
    print("tensorflow version {}".format(tf.__version__))

    # Data processing: (for their dataset)
    # -----------------------------------------
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
        index_train = int(X.shape[0] * train_rate) #  NOTE
        X_train, y_train = X[0:index_train], Y[0:index_train]
        X_test, y_test = X[index_train:-1], Y[index_train:-1]  # NOTE -1


    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)


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
    #model = FCN8(nClasses=nb_classes,input_height=224,input_width=224)
    #model = fcn_8s_take2(INPUT_SIZE,nb_classes)
    #model = fcn_VGG16_32s_crfrnn(INPUT_SIZE, nb_classes)
    #model = fcn_VGG16_8s_crfrnn(INPUT_SIZE, nb_classes)
    #model = fcn_32s(INPUT_SIZE, nb_classes)
    #model = fcn_32s_orig(nb_classes)
    #model = fcn_RESNET50_32s(INPUT_SIZE,nb_classes)
    #model = fcn_RESNET50_8s(INPUT_SIZE, nb_classes)
    model = fcn_RESNET50_8s_crfrnn(INPUT_SIZE, nb_classes)
    #model = fcn_8s_Sadeep(nb_classes)
    #model = fcn_8s_Sadeep_crfrnn(nb_classes)

    # if resuming training:
    saved_model_path = '/storage/gby/semseg/streets_weights_fcn8s_Sadeep_500ep' #'crfrnn_keras_model.h5'
    #saved_model_path = '/storage/gby/semseg/voc12_weights'
    #saved_model_path = 'crfrnn_keras_model.h5'
    #model.load_weights(saved_model_path)

    model.summary()

    visualize_filters_flag = False
    layer_name = 'score2' # 'crfrnn' 'score2'
    if visualize_filters_flag:
        visualize_conv_filters(model, INPUT_SIZE, layer_name)


    # Training starts here:
    # ----------------------
    #sgd = optimizers.SGD(lr=1E-2, decay=5 ** (-4), momentum=0.9, nesterov=True)
    #sgd = optimizers.SGD(lr=1e-13, momentum=0.99)
#   adm = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.001)
    #model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
 #   model.compile(loss="categorical_crossentropy", optimizer=adm, metrics=['accuracy'])
    #model.compile(loss="binary_crossentropy", optimizer='sgd', metrics=['accuracy'])
    #model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.0001), metrics=['accuracy'])
    #model.compile(loss="categorical_crossentropy", optimizer='Adadelta', metrics=['accuracy'])

    model.compile(loss=weighted_loss(nb_classes, coefficients),
                  optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.001),
                  metrics=['accuracy'])

    # hist1 = model.fit(X_train, y_train,
    #                   validation_data=(X_test, y_test),
    #                   batch_size=6, epochs=400, verbose=2)

    # for crfrnn:
    hist1 = model.fit(X_train, y_train,
                      validation_data=(X_test, y_test),
                      batch_size=1, epochs=50, verbose=1)

    # save model:
    model.save_weights(RES_DIR + 'voc12_weights')

    save_graphics_mode = False
    print_IoU_flag = False

    # Plot/save the change in loss over epochs:
    # -------------------------------------
    with open('trainHistoryDict', 'wb') as file_pi:
        pickle.dump(hist1.history, file_pi)

    if(save_graphics_mode):
        for key in ['loss', 'val_loss']:
            plt.plot(hist1.history[key], label=key)
        plt.legend()
        #plt.show(block=False)
        plt.savefig('loss_plot.pdf')

    # Compute IOU:
    # ------------
    if(print_IoU_flag):
        print('computing mean IoU for validation set..')
        y_pred = model.predict(X_test)
        y_predi = np.argmax(y_pred, axis=3)
        y_testi = np.argmax(y_test, axis=3)
        print(y_testi.shape, y_predi.shape)
        IoU_ver2(y_testi, y_predi)
        #pdb.set_trace()

    # Visualize the model performance:
    # --------------------------------
    shape = (INPUT_SIZE, INPUT_SIZE)
    n_classes = nb_classes # 10

    if save_graphics_mode:

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

        plt.savefig('examples.png')

    # Predict 1 test exmplae and save:
    #model_predict_gby(model, 'image.jpg', 'predict-final.png', INPUT_SIZE)
    #pdb.set_trace()
    model.save_weights(RES_DIR + 'voc12_weights')


# usage:
# >>python train2.py ./list/train2s.txt ./list/val2s.txt /storage/gby/datasets/pascal_voc12/images_orig/ /storage/gby/datasets/pascal_voc12/labels_orig/
# >>python train2.py ./list/train2.txt ./list/val2.txt /storage/gby/datasets/pascal_voc12/images_orig/ /storage/gby/datasets/pascal_voc12/labels_orig/
# comment: /storage/gby/datasets/pascal_voc12/ is a copy of Cristina's folder



#

"""
MIT License

"""


import numpy as np
#from keras.callbacks import ModelCheckpoint
#from keras.optimizers import Adam
#from keras.optimizers import SGD
#from keras import backend as K
import matplotlib.pyplot as plt
import pdb
from models_gby import fcn_32s_orig,fcn_32s,fcn_8s,fcn_VGG16_32s_crfrnn,fcn_8s_take2,fcn_VGG16_8s_crfrnn # FCN8
from crfrnn_model import get_crfrnn_model_def
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

## location of VGG weights
VGG_Weights_path = "../FacialKeypoint/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"

RES_DIR = "/storage/gby/semseg/"

INPUT_SIZE = 224 #500 #224

# ===========================
# Main
# ===========================
if __name__ == '__main__':

    # Parse args:
    # -----------
    parser = argparse.ArgumentParser()
    parser.add_argument('train_data')
    parser.add_argument('val_data')
    parser.add_argument('image_dir')
    parser.add_argument('label_dir')
    args = parser.parse_args()

    # Import Keras and Tensorflow to develop deep learning FCN models:
    # -----------------------------------------------------------------
    warnings.filterwarnings("ignore")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    config.gpu_options.visible_device_list = "2"
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
    index_train = np.random.choice(X.shape[0], int(X.shape[0] * train_rate), replace=False)
    index_test = list(set(range(X.shape[0])) - set(index_train))

    # X, Y = shuffle(X, Y)
    X_train, y_train = X[index_train], Y[index_train]
    X_test, y_test = X[index_test], Y[index_test]
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
    model = fcn_8s_take2(INPUT_SIZE,nb_classes)
    #model = fcn_VGG16_32s_crfrnn(INPUT_SIZE, nb_classes)
    #model = fcn_VGG16_8s_crfrnn(INPUT_SIZE, nb_classes)
    #model = fcn_32s(INPUT_SIZE, nb_classes)
    #model = fcn_32s_orig(nb_classes)

    # if resuming training:
    #saved_model_path = '/storage/gby/semseg/voc12_weights' #'crfrnn_keras_model.h5'
    #model.load_weights(saved_model_path)

    model.summary()

    # Training starts here:
    # ----------------------
#    sgd = optimizers.SGD(lr=1E-2, decay=5 ** (-4), momentum=0.9, nesterov=True)
#    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.compile(loss="categorical_crossentropy", optimizer='sgd', metrics=['accuracy'])
    #model.compile(loss="binary_crossentropy", optimizer='sgd', metrics=['accuracy'])
    #model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.0001), metrics=['accuracy'])
    #model.compile(loss="categorical_crossentropy", optimizer='Adadelta', metrics=['accuracy'])

    hist1 = model.fit(X_train, y_train,
                      validation_data=(X_test, y_test),
                      batch_size=32, epochs=400, verbose=2)

    # Plot the change in loss over epochs:
    # -------------------------------------
    for key in ['loss', 'val_loss']:
        plt.plot(hist1.history[key], label=key)
    plt.legend()
    #plt.show(block=False)
    plt.savefig('loss_plot.pdf')

    # Compute IOU:
    # ------------
    print('computing mean IoU for validation set..')
    y_pred = model.predict(X_test)
    y_predi = np.argmax(y_pred, axis=3)
    y_testi = np.argmax(y_test, axis=3)
    print(y_testi.shape, y_predi.shape)
    IoU_ver2(y_testi, y_predi)

    # Visualize the model performance:
    # --------------------------------
    shape = (INPUT_SIZE, INPUT_SIZE)
    n_classes = nb_classes # 10

    # for i in range(1):
    #     img_is = (X_test[i] + 1) * (255.0 / 2)
    #     seg = y_predi[i]
    #     segtest = y_testi[i]
    #
    #     fig = plt.figure(figsize=(10, 10))
    #     ax = fig.add_subplot(1, 3, 1)
    #     ax.imshow(img_is / 255.0)
    #     ax.set_title("original")
    #
    #     ax = fig.add_subplot(1, 3, 2)
    #     ax.imshow(give_color_to_seg_img(seg, n_classes))
    #     ax.set_title("predicted class")
    #
    #     ax = fig.add_subplot(1, 3, 3)
    #     ax.imshow(give_color_to_seg_img(segtest, n_classes))
    #     ax.set_title("true class")
    #     #plt.show()
    # plt.savefig('examples.png')

    # Predict 1 test exmplae and save:
    #model_predict_gby(model, 'image.jpg', 'predict-final.png', INPUT_SIZE)
    #pdb.set_trace()
    model.save_weights(RES_DIR + 'voc12_weights')


# usage:
# >>python train2.py ./list/train2s.txt ./list/val2s.txt /storage/gby/datasets/pascal_voc12/images_orig/ /storage/gby/datasets/pascal_voc12/labels_orig/
# >>python train2.py ./list/train2.txt ./list/val2.txt /storage/gby/datasets/pascal_voc12/images_orig/ /storage/gby/datasets/pascal_voc12/labels_orig/
# comment: /storage/gby/datasets/pascal_voc12/ is a copy of Cristina's folder
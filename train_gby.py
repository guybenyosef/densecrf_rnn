"""
MIT License

"""


#from keras.callbacks import ModelCheckpoint
#from keras.optimizers import Adam
#from keras.optimizers import SGD
#from keras import backend as K
#import matplotlib.pyplot as plt
import pdb
from models_gby import load_model_gby
from datatsets_gby import load_dataset
#from models_gby import fcn_8s_Sadeep,fcn_8s_Sadeep_crfrnn
from utils_gby import IoU_ver2,give_color_to_seg_img,visualize_conv_filters

## Import usual libraries
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras, sys, time, warnings
from keras.models import *
from keras.layers import *
import pandas as pd
from keras import optimizers
import argparse
import pickle
from keras.optimizers import Adam
from src.weighted_categorical_crossentropy import weighted_loss

## location of VGG weights
VGG_Weights_path = "../FacialKeypoint/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"

RES_DIR = "/storage/gby/semseg/"

def argument_parser():
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('-m', '--model', default='fcn_RESNET50_8s', help='choose between \'fcn_VGG16_32s\',\'fcn_VGG16_8s\',\'fcn_RESNET50_32s\', and \'fcn_RESNET50_8s\' networks, with or without \'_crfrnn\' suffix', type=str)
    parser.add_argument('-ds', '--dataset', default='streets', help='The name of train/test sets', type=str)
    parser.add_argument('-bs', '--batchsize', default=1, help='Specify the number of batches', type=int)
    parser.add_argument('-is', '--inputsize', default=None, help='Specify the input size N, where N=rows,N=cols. ', type=int)
    parser.add_argument('-w', '--weights', default=None, nargs='?', const=None, help='The absolute path of the weights', type=str)
    parser.add_argument('-e', '--epochs', default=1, const=None, help='Specify the number of epochs to train', type=int)
    parser.add_argument('-vb', '--verbosemode', default=1, help='Specify the verbose mode',type=int)

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
    args = argument_parser()

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

    # ===============
    # LOAD train data:
    # ===============

    INPUT_SIZE = args.inputsize  #500 #224 #512

    ds = load_dataset(args.dataset,INPUT_SIZE)
    print(ds.X_train.shape, ds.y_train.shape)
    print(ds.X_test.shape, ds.y_test.shape)
    nb_classes = ds.nb_classes

    # ===============
    # LOAD model:
    # ===============

    # for training:
    num_crf_iterations = 5

    model = load_model_gby(args.model, INPUT_SIZE, nb_classes, num_crf_iterations)

    # if resuming training:
    if (args.weights is not None) and (os.path.exists(args.weights)):
        model.load_weights(args.weights)

    model.summary()
    print('trining model %s..'% model.name)

    # ===============
    # TRAIN model:
    # ===============
    #sgd = optimizers.SGD(lr=1E-2, decay=5 ** (-4), momentum=0.9, nesterov=True)
    #sgd = optimizers.SGD(lr=1e-13, momentum=0.99)
    #adm = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.001)
    #model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    #model.compile(loss="categorical_crossentropy", optimizer=adm, metrics=['accuracy'])
    #model.compile(loss="binary_crossentropy", optimizer='sgd', metrics=['accuracy'])
    #model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.0001), metrics=['accuracy'])
    #model.compile(loss="categorical_crossentropy", optimizer='Adadelta', metrics=['accuracy'])

    num_epochs = args.epochs
    batch_size = args.batchsize
    verbose_mode = args.verbosemode
    coefficients = ds.weighted_loss_coefficients

    if model.crf_flag:
        model.compile(loss=weighted_loss(nb_classes, coefficients),
                      optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.001),
                      metrics=['accuracy'])
    else:
        model.compile(loss='categorical_crossentropy',
                      optimizer='sgd',
                      metrics=['accuracy'])

    hist1 = model.fit(ds.X_train, ds.y_train,
                      validation_data=(ds.X_test, ds.y_test),
                      batch_size=batch_size, epochs=num_epochs, verbose=verbose_mode)

    # ===============
    # SAVE model:
    # ===============
    model.save_weights(RES_DIR + args.dataset + '_weights_' + model.name + '_' + str(num_epochs) + 'ep')

    # ===============
    # ANALYZE model:
    # ===============
    save_graphics_mode = False
    print_IoU_flag = True
    visualize_filters_flag = False

    # Visualize conv filters:
    # -------------------------------------
    layer_name = 'score2' # 'crfrnn' 'score2'
    if visualize_filters_flag:
        visualize_conv_filters(model, INPUT_SIZE, layer_name)

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
        y_pred = model.predict(ds.X_test)
        y_predi = np.argmax(y_pred, axis=3)
        y_testi = np.argmax(ds.y_test, axis=3)
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

        plt.savefig('examples.png')


# usage:
# >>python train2.py ./list/train2s.txt ./list/val2s.txt /storage/gby/datasets/pascal_voc12/images_orig/ /storage/gby/datasets/pascal_voc12/labels_orig/
# >>python train2.py ./list/train2.txt ./list/val2.txt /storage/gby/datasets/pascal_voc12/images_orig/ /storage/gby/datasets/pascal_voc12/labels_orig/
# comment: /storage/gby/datasets/pascal_voc12/ is a copy of Cristina's folder



#

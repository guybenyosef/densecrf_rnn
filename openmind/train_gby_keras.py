"""
MIT License

"""
import pdb
from models_gby_new import load_model_gby
from datasets_gby_new import load_dataset
from utils_gby_new import IoU_ver2,give_color_to_seg_img,visualize_conv_filters,compute_median_frequency_reweighting,load_segmentations
from src.weighted_categorical_crossentropy import weighted_loss

## Import usual libraries
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import CSVLogger
from keras.optimizers import Adam
from keras.layers import *
import keras, sys, time, warnings
import matplotlib.pyplot as plt
import pandas as pd
from keras import optimizers
import argparse
import ntpath
import pickle
import cv2
from keras.callbacks import ModelCheckpoint
from tensorflow.python import debug as tf_debug

#RES_DIR = "/storage/cfmata/deeplab/crf_rnn/crfasrnn_keras/results/pascal_voc12/"
#RES_DIR = "/storage/cfmata/deeplab/crf_rnn/crfasrnn_keras/results/horse_coarse/"

def argument_parser():
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('-m', '--model', default='fcn_RESNET50_8s', help='choose between \'fcn_VGG16_32s\',\'fcn_VGG16_8s\',\'fcn_RESNET50_32s\', and \'fcn_RESNET50_8s\' networks, with or without \'_crfrnn\' suffix', type=str)
    parser.add_argument('-ds', '--dataset', default='horsecoarse', help='The name of train/test sets', type=str)
    parser.add_argument('-bs', '--batchsize', default=1, help='Specify the number of batches', type=int)
    parser.add_argument('-is', '--inputsize', default=512, help='Specify the input size N, where N=rows,N=cols. ', type=int)
    parser.add_argument('-w', '--weights', default=None, nargs='?', const=None, help='The absolute path of the weights', type=str)
    parser.add_argument('-e', '--epochs', default=1, const=None, help='Specify the number of epochs to train', type=int)
    parser.add_argument('-vb', '--verbosemode', default=1, help='Specify the verbose mode', type=int)
    parser.add_argument('-hf', '--h_flip', default=False, help='Whether to randomly flip the image horizontally for data augmentation', type=bool)
    parser.add_argument('-vf', '--v_flip', default=False, help='Whether to randomly flip the image vertically for data augmentation', type=bool)
    parser.add_argument('-br', '--brightness', type=float, default=None,
                        help='Whether to randomly change the image brightness for data augmentation. Specifies the max bightness change as a factor between 0.0 and 1.0. For example, 0.1 represents a max brightness change of 10%% (+-).')
    parser.add_argument('-ro', '--rotation', type=float, default=None,
                        help='Whether to randomly rotate the image for data augmentation. Specifies the max rotation angle in degrees.')
    parser.add_argument('-se', '--stepsepoch', default=100, help='Specify the number of steps for epoch', type=int)
    parser.add_argument('-g', '--gpu', default="2", help='Select visible gpu device [0-3]', type=str)
    parser.add_argument('-ft', '--finetune_path', default='', help='Path for  finetuning weights', type=str)

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
    config.gpu_options.visible_device_list = args.gpu # default: "2"
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.OFF
    ''' TF Debugger option
    sess = tf.Session(config=config)
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    set_session(sess)
    '''
    set_session(tf.Session(config=config)) # set_session is keras function

    # Set results directory
    if args.dataset == "horsecoarse" or args.dataset == "horsecoarse_small":
        RES_DIR = "/storage/cfmata/deeplab/crf_rnn/crfasrnn_keras/results/horse_coarse/"
    elif args.dataset == "voc2012":
        RES_DIR = "/storage/cfmata/deeplab/crf_rnn/crfasrnn_keras/results/pascal_voc12/"
    else:
        RES_DIR = ""

    # ===============
    # LOAD model:
    # ===============

    INPUT_SIZE = args.inputsize
    nb_classes = 6 # TODO: change this back when variable ordering fixed
    batch_size = args.batchsize

    # for training:
    num_crf_iterations = 5
    model = load_model_gby(args.model, INPUT_SIZE, nb_classes, num_crf_iterations, args.finetune_path, batch_size)

    # if resuming training:
    if (args.weights is not None) and (os.path.exists(args.weights)):
        print("loading weights %s.."% args.weights)
        model.load_weights(args.weights)

    model.summary()
    print('trining model %s..'% model.name)
    
    # ===============
    # LOAD train data:
    # ===============

    #INPUT_SIZE = args.inputsize  #500 #224 #512

    # parameters for data sugmentation:
    #batch_size = args.batchsize
    ds = load_dataset(args.dataset, INPUT_SIZE, args, batch_size) #dataaug_args)
    
    print(ds.X_train.shape, ds.y_train.shape)
    print(ds.X_test.shape, ds.y_test.shape)

    # ===============
    # LOAD sp segment:
    # ===============
    if model.sp_flag:
        segments_train = load_segmentations(ds.segments_dir, ds.train_list, INPUT_SIZE)
        segments_test = load_segmentations(ds.segments_dir, ds.test_list, INPUT_SIZE)
        print("Loading superpixels segmentations:")
        print(segments_train.shape, segments_test.shape)
    else:
        segments_train = np.array([])
        segments_test = np.array([])

    # Pad training/val sets with random imgs to get number divisible by batch_size; can change to using images from next epoch
    #(train_quotient, train_remainder) = divmod(ds.X_train.shape[0], batch_size)
    train_remainder = ds.X_train.shape[0] % batch_size
    print("train remainder ", train_remainder)
    if train_remainder != 0:
        # Choose random indices for extra imgs
        print("train high bound ", ds.X_train.shape[0])
        indices_train = np.random.randint(0,high=ds.X_train.shape[0], size=batch_size-train_remainder)
        print("len extra train ", len(indices_train))
        for i in indices_train:
            ds.X_train = np.concatenate((ds.X_train, [ds.X_train[i]]), axis=0)
            ds.y_train = np.concatenate((ds.y_train, [ds.y_train[i]]), axis=0)
            if model.sp_flag:
                segments_train = np.concatenate((segments_train, [segments_train[i]]), axis=0)
                
    print("new len ", len(ds.X_train), len(segments_train))
    
    #(test_quotient, test_remainder) = divmod(ds.X_test.shape[0], batch_size)
    test_remainder = ds.X_test.shape[0] % batch_size
    if test_remainder != 0:
        indices_test = np.random.randint(0,high=ds.X_test.shape[0], size=batch_size-test_remainder)
        #print("len extra test ", len(indices_test))
        for i in indices_test:
            ds.X_test = np.concatenate((ds.X_test, [ds.X_test[i]]), axis=0)
            ds.y_test = np.concatenate((ds.y_test, [ds.y_test[i]]), axis=0)
            if model.sp_flag:
                segments_test = np.concatenate((segments_test, [segments_test[i]]), axis=0)


    if model.sp_flag:
        ds.X_train = [ds.X_train, segments_train]
        ds.X_test = [ds.X_test, segments_test]


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
    verbose_mode = args.verbosemode
    #   coefficients = ds.weighted_loss_coefficients
    # for weighted_loss_coefficients
    y_traini = np.argmax(ds.y_train, axis=3)
    coefficients = list(compute_median_frequency_reweighting(y_traini))

    # Logger callback for learning curves
    csv_logger = CSVLogger('run/train_log.csv', append=True, separator=',')
    #pdb.set_trace()
    if model.crf_flag: # True:#
        model.compile(loss=weighted_loss(nb_classes, coefficients),
                      optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.001),
                      metrics=['accuracy'])
    else:
        print("using categorical crossentropy")
        model.compile(loss='categorical_crossentropy', # weighted_loss(nb_classes, coefficients), # 'categorical_crossentropy',
                      optimizer='sgd',
                      metrics=['accuracy'])
                      #callbacks=['csv_logger'])


    #pdb.set_trace()
    data_augmentation_flag = False
    if args.h_flip | args.v_flip | (not args.brightness==None) | (not   args.rotation==None):
        data_augmentation_flag = True  # False #

    #checkpoint = ModelCheckpoint(RES_DIR+"voc2012_sadeep_crf.{epoch:02d}-{val_loss:.2f}", save_weights_only=True,verbose=1, period=1)
    checkpoint = ModelCheckpoint(RES_DIR+"horsecoarse_checkpt_crfrnn_start30.{epoch:02d}-{val_loss:.2f}", save_weights_only=True,verbose=1, period=1000)
    cb_list = [checkpoint]
    
    if not data_augmentation_flag:
        # option 1:
        print("NOT using data augmentation..")
        
        hist1 = model.fit(ds.X_train, ds.y_train,
                          validation_data=(ds.X_test, ds.y_test),
                          batch_size=batch_size, epochs=num_epochs, verbose=verbose_mode,#)
                          callbacks=cb_list)
        
        
    else:
        # option 2: use data generator for data augmentation
        print("using data augmentation..")
        hist1 = model.fit_generator(generator=ds.datagen_train,
                          validation_data=(ds.X_test, ds.y_test),
                          steps_per_epoch=args.stepsepoch,
                          use_multiprocessing=True,
                          epochs=num_epochs, verbose=verbose_mode)


    # ===============
    # SAVE model:
    # ===============
    save_by_name = RES_DIR + args.dataset + '_weights_' + model.name + '_is' + str(INPUT_SIZE) + '_ep' + str(num_epochs)
    model.save_weights(save_by_name)
    print("model saved to %s"%save_by_name)

    # ===============
    # ANALYZE model:
    # ===============
    save_graphics_mode = True
    print_IoU_flag = True
    visualize_filters_flag = False

    # Compute IOU:
    # ------------
    if (print_IoU_flag):
        print('computing mean IoU for validation set..')
        y_pred = model.predict(ds.X_test, batch_size=batch_size, verbose=verbose_mode)
        y_predi = np.argmax(y_pred, axis=3)
        y_testi = np.argmax(ds.y_test, axis=3)
        print(y_testi.shape, y_predi.shape)
        IoU_ver2(y_testi, y_predi)
        # pdb.set_trace()
    '''
    # Visualize conv filters:
    # -------------------------------------
    layer_name = 'score_pool7c_upsample_32' # 'crfrnn' 'score2'
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
        plt.savefig('loss_%s.pdf' % ntpath.basename(save_by_name))

    # Visualize the model performance:
    # --------------------------------
    shape = (INPUT_SIZE, INPUT_SIZE)
    n_classes = nb_classes

    if save_graphics_mode:

        num_examples_to_plot = 4

        fig = plt.figure(figsize=(10, 3*num_examples_to_plot))

        for i in range(num_examples_to_plot):

            img_indx = i*2 # i*4
            img_is = ds.X_test[img_indx]
            cv2.normalize(img_is, img_is, 0, 1, cv2.NORM_MINMAX)  # img_is = (ds.X_test[img_indx] + 1) * (255.0 / 2)
            seg = y_predi[img_indx]
            segtest = y_testi[img_indx]

            ax = fig.add_subplot(num_examples_to_plot, 3, 3 * i + 1)
            ax.imshow(img_is) # ax.imshow(img_is / 255.0)
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

        plt.savefig('examples_%s.png' % ntpath.basename(save_by_name))
    '''

# usage:
# >>python train2.py ./list/train2s.txt ./list/val2s.txt /storage/gby/datasets/pascal_voc12/images_orig/ /storage/gby/datasets/pascal_voc12/labels_orig/
# >>python train2.py ./list/train2.txt ./list/val2.txt /storage/gby/datasets/pascal_voc12/images_orig/ /storage/gby/datasets/pascal_voc12/labels_orig/
# comment: /storage/gby/datasets/pascal_voc12/ is a copy of Cristina's folder

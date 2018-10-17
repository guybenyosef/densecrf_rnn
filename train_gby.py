"""
MIT License

"""

# import sys
# sys.path.insert(1, './src')
# from crfrnn_model import get_crfrnn_model_def
# from fcn8_model import get_fcn8_model_def
# from fcn32_model_gby import fcn_32s
# import util
# from os import listdir
# from os.path import isfile, join
# import numpy as np
# from scipy import misc
# from PIL import Image
# from keras import optimizers
# from keras import losses
# import pickle
# import pdb
#
# INPUT_DIR = "/storage/cfmata/deeplab/crf_rnn/crfasrnn_keras/data/pascal_voc12/images_orig/"
# GT_DIR = "/storage/cfmata/deeplab/crf_rnn/crfasrnn_keras/data/pascal_voc12/labels_orig/"
# RES_DIR = "/storage/gby/semseg/"
#
# def prepare_training_data(img_list_path, im_file_name, label_file_name):
#     with open(img_list_path) as f:
#         content = f.readlines()
#     im_list = sorted([x[42:-5] for x in content]) # Slicing specific to pascal voc trainval lists
#
#     # Prepare image and label data
#     inputs, labels = [], []
#     i=0
#     for name in im_list:
#         img_data, img_h, img_w = util.get_preprocessed_image(INPUT_DIR + name + ".jpg")
#         inputs.append(img_data)
#
#         if i % 100 == 0:
#             print("Processed ", i)
#         img_data, img_h, img_w = util.get_preprocessed_label(GT_DIR + name + ".png", 21)
#         labels.append(img_data)
#         i+=1
#     '''
#     # Using pickle
#     im_file = open(im_file_name, 'wb')
#     pickle.dump(inputs,im_file)
#     im_file.close()
#     label_file = open(label_file_name, 'wb')
#     pickle.dump(labels, label_file)
#     label_file.close()
#     '''
#
#     # Using numpy
#     np.save(RES_DIR + "image_data.npy", inputs)
#     np.save(RES_DIR + "label_data.npy", labels)
#
# def train(im_file_name, label_file_name):
#     # Load img and label data
#     '''
#     # Using pickle
#     input_file = open(im_file_name, 'rb')
#     inputs = pickle.load(input_file)
#     label_file = open(label_file_name, 'rb')
#     labels = pickle.load(label_file)
#     '''
#
#     # Using numpy
#     inputs = np.load(RES_DIR + "image_data.npy")
#     labels = np.load(RES_DIR + "label_data.npy")
#     #pdb.set_trace()
#     # Download the model from https://goo.gl/ciEYZi
#     saved_model_path = 'crfrnn_keras_model.h5'
#
#     # Initialize model
#     # fcn8:
#     # model = get_fcn8_model_def(
#     # fcn32
#     model = fcn_32s()
#     # crfasarnn
#     #model = get_crfrnn_model_def()
#     #model.load_weights(saved_model_path)
#
#     # Compile model
#     #adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#     #adam = optimizers.Adam(lr=1e-13, beta_1=0.99, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#     #adam = optimizers.Adam(lr=1e-9, beta_1=0.99, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#     #model.compile(loss='mean_squared_error', optimizer=adam)
#     #model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
#     #model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#     #model.compile(loss=losses.sparse_categorical_crossentropy, optimizer=adam)
#     model.compile(loss='categorical_crossentropy', optimizer='sgd')
#
#     #model.fit(x=inputs, y=labels, batch_size=1)
#     # Start finetuning
#     #for i in range(len(inputs)):
#     for i in range(1000):
#         print("img ", i)
#     #    model.fit(x=inputs[i], y=labels[i], epochs=3, steps_per_epoch=1)
#         model.fit(x=inputs[i], y=labels[i], epochs=1, steps_per_epoch=1)
#
#
#     # Save model weights
#     model.save_weights(RES_DIR + 'voc12_weights')
#
# if __name__ == '__main__':
#     image_fn, label_fn = "image_data.npy", "label_data.npy"
#   #  prepare_training_data("./list/train.txt", image_fn, label_fn)
#     #pdb.set_trace()
#     train(image_fn, label_fn)

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras import backend as K
import matplotlib.pyplot as plt
import argparse
import pdb
from models_gby import fcn_32s,fcn_8s,fcn_VGG16_32s_crfrnn
from utils_gby import generate_arrays_from_file,extract_arrays_from_file,IoU,model_predict_gby

#from fcn8_model import get_fcn8_model_def
#from crfrnn_model import get_crfrnn_model_def

RES_DIR = "/storage/gby/semseg/"

nb_classes = 21

# ===========================
# Main
# ===========================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_data')
    parser.add_argument('val_data')
    parser.add_argument('image_dir')
    parser.add_argument('label_dir')
    args = parser.parse_args()
    nb_data = sum(1 for line in open(args.train_data))

    INPUT_SIZE = None  # None # 496 # 510, 500
    model = fcn_8s(INPUT_SIZE,nb_classes)
    #model = fcn_32s(nb_classes)

    # # crfrnn:
    # INPUT_SIZE = 480  # None # 496 # 510, 500
    # model = fcn_VGG16_32s_crfrnn(INPUT_SIZE,nb_classes)

    # if resuming training:
    #saved_model_path = '/storage/gby/semseg/voc12_weights_180_snap' # voc12_weights_36_snap' #'crfrnn_keras_model.h5'
    #model.load_weights(saved_model_path)

    model.summary()

    # compile 1:
    # model.compile(loss="binary_crossentropy", optimizer='sgd',metrics=['accuracy'])
    #model.compile(loss="categorical_crossentropy", optimizer='sgd', metrics=['accuracy'])
    # compile 2:
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
    # compile 3:
    # sgd = SGD(lr=1E-2, decay=5**(-4), momentum=0.9, nesterov=True)
    # model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


    # ------------------------------
    # Begin training procedure:
    # ------------------------------
    print('lr = {}'.format(K.get_value(model.optimizer.lr)))

    # -- if we use 'fit': --
    # print('training data:')
    # [train_imgs,train_labels] = extract_arrays_from_file(args.train_data, args.image_dir, args.label_dir)
    print('validation data:')
    [val_imgs,val_labels] = extract_arrays_from_file(args.val_data, args.image_dir, args.label_dir, INPUT_SIZE, nb_classes)
    #pdb.set_trace()
    # or simply,
    # train_imgs = np.load(RES_DIR + "image_data.npy")[:,0,:,:]
    # train_labels = np.load(RES_DIR + "label_data.npy")[:,0,:,:]

    for epoch in range(50):

        print('Starting epoch %d ..' % epoch)

        hist1 = model.fit_generator(
            generate_arrays_from_file(args.train_data, args.image_dir, args.label_dir, INPUT_SIZE, nb_classes),
            validation_data=generate_arrays_from_file(args.val_data, args.image_dir, args.label_dir, INPUT_SIZE, nb_classes),
            steps_per_epoch=nb_data,
            validation_steps=nb_data,
            epochs=1,
            verbose=1)    # verbose=1 to show progress bar # verbose=2 to show one line per epoch

        # hist1 = model.fit(
        #     x=train_imgs,
        #     y=train_labels,
        #     #validation_data=[val_imgs,val_labels],
        #     batch_size=1,
        #     epochs=1,
        #     verbose=2)

        # show loss plot:
        # for key in ['loss', 'val_loss']:
        #     plt.plot(hist1.history[key], label=key)
        # plt.legend()
        # plt.show()

        # Compute IOU:
        print('computing mean IoU for validation set..')
        mIoU = 0
        for k in range(len(val_imgs)):
            X_test = val_imgs[k]
            y_test = val_labels[k]
            # pdb.set_trace()
            y_pred = model.predict(X_test)
            y_predi = np.argmax(y_pred, axis=3)
            y_testi = np.argmax(y_test, axis=3)
            # print(y_testi.shape, y_predi.shape)
            mIoU += IoU(y_testi, y_predi,nb_classes)
        mIoU = mIoU/len(val_imgs)

        print("Mean IoU: {:4.3f}".format(mIoU))
        print("_________________")

        # Predict 1 test exmplae and save:
        model_predict_gby(model, 'image.jpg', 'predict-{}.png'.format(epoch), INPUT_SIZE)
        #model.save_weights(RES_DIR + 'voc12_weights-{}'.format(epoch))

    #model.save_weights(RES_DIR + 'voc12_weights')


# usage:
# >>python train_gby.py ./list/train2s.txt ./list/val2s.txt /storage/gby/datasets/pascal_voc12/images_orig/ /storage/gby/datasets/pascal_voc12/labels_orig/
# >>python train_gby.py ./list/train2.txt ./list/val2.txt /storage/gby/datasets/pascal_voc12/images_orig/ /storage/gby/datasets/pascal_voc12/labels_orig/
# comment: /storage/gby/datasets/pascal_voc12/ is a copy of Cristina's folder
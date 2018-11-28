"""
MIT License

Copyright (c) 2017 Sadeep Jayasumana

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

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
from utils_gby import load_image,give_color_to_seg_img
import cv2
import pdb


def argument_parser_eval():
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('-im', '--imagepath', default=None, help='Absolute image path', type=str)
    parser.add_argument('-fl', '--folderpath', default=None, help='Absolute folder path', type=str)
    parser.add_argument('-m', '--model', default='fcn_RESNET50_8s', help='choose between \'fcn_VGG16_32s\',\'fcn_VGG16_8s\',\'fcn_RESNET50_32s\', and \'fcn_RESNET50_8s\' networks, with or without \'_crfrnn\' suffix', type=str)
    parser.add_argument('-w', '--weights', default=None, nargs='?', const=None, help='The absolute path of the weights',type=str)
    parser.add_argument('-nc', '--nbclasses', default=2, help='Number of labels', type=int)
    parser.add_argument('-is', '--inputsize', default=512, help='Specify the input size N, where N=rows,N=cols. ',type=int)
    parser.add_argument('-g', '--gpu', default="2", help='Select visible gpu device [0-3]', type=str)
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

    n_classes = args.nbclasses
    INPUT_SIZE = args.inputsize
    num_crf_iterations = 10  # at test time

    # ===============
    # LOAD model:
    # ===============

    model_name = args.model
    model_path_name = args.weights
    base_img_name = []

    print('====================================================================================')
    print(model_path_name)
    print('====================================================================================')

    finetune_path = ''
    #pdb.set_trace()
    model = load_model_gby(model_name, INPUT_SIZE, n_classes, num_crf_iterations, finetune_path)


    #loading weights:
    model.load_weights(model_path_name)

    # Computing prediction:
    # ------------------------------
    print('computing prediction..')
    if args.folderpath==None:
        img_org = cv2.imread(args.imagepath)
        x = load_image(img_org, INPUT_SIZE)
        base_img_name.append(os.path.splitext(os.path.basename(args.imagepath))[0])
    else:
        X = []
        outdirName = args.folderpath+'out/'
        if not os.path.exists(outdirName):
            os.mkdir(outdirName)

        for filename in os.listdir(args.folderpath):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img_path = args.folderpath+filename
                img_org = cv2.imread(img_path)
                x = load_image(img_org, INPUT_SIZE)
                X.append(x)
                base_img_name.append(outdirName + os.path.splitext(os.path.basename(img_path))[0])
        x = np.array(X)[:, 0, :, :, :]


    y_pred = model.predict(x, batch_size=1, verbose=1)
    y_predi = np.argmax(y_pred, axis=3)

    for ii in range(len(base_img_name)):

        # Visualize the model performance:
        # --------------------------------
        #shape = (INPUT_SIZE, INPUT_SIZE)

        fig = plt.figure(figsize=(10, 5))

        img_is = np.array(x[ii, :, :, :])
        cv2.normalize(img_is, img_is, 0, 1, cv2.NORM_MINMAX)
        seg = y_predi[ii,:,:]#[img_indx]

        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(img_is) # ax.imshow(img_is / 255.0)
        ax.set_title("original")

        ax = fig.add_subplot(1, 2, 2)
        ax.imshow(give_color_to_seg_img(seg, n_classes))
        ax.set_title("predicted segmentation")


        plt.savefig('%s_prediction_%s.png' % (base_img_name[ii],ntpath.basename(model_path_name)))

    # clear model: Destroys the current TF graph and creates a new one. Useful to avoid clutter from old models / layers.
    keras.backend.clear_session()

# def main(saved_model_path,input_file_path):
#     input_file_dir, input_file_basename = os.path.split(input_file_path)
#     input_file_name, input_file_ext = os.path.splitext(input_file_basename)
#     output_file_path = input_file_dir+input_file_name+'_labels.png'
#
#     print('runnnig model %s on image %s, write to %s' % (saved_model_path, input_file_path, output_file_path))
#
#     # Download the model from https://goo.gl/ciEYZi
#     #saved_model_path = 'crfrnn_keras_model.h5'
#
#     #model = get_crfrnn_model_def()
#     model = get_fcn8_model_def()
#     model.load_weights(saved_model_path)
#
#     img_data, img_h, img_w = util.get_preprocessed_image(input_file_path)
#     probs = model.predict(img_data, verbose=False)[0, :, :, :]
#     pdb.set_trace()
#     segmentation = util.get_label_image(probs, img_h, img_w)
#     segmentation.save(output_file_path)


# if __name__ == '__main__':
#     saved_model_path, input_file_path = sys.argv[1],sys.argv[2]
#     main(saved_model_path, input_file_path)


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

import sys
sys.path.insert(1, './src')
#from crfrnn_model import get_crfrnn_model_def
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
from utils_gby import IoU_ver2,give_color_to_seg_img, model_predict_gby, load_segmentations
import pdb
import scipy.misc
from PIL import Image
import matplotlib
#from models_gby import fcn_RESNET50_8s, fcn_RESNET50_8s_crfrnn
import util
from os import listdir
from os.path import isfile, join

def test():
    # Set input and output dirs
    input_dir = "/storage/cfmata/deeplab/crf_rnn/crfasrnn_keras/data/horse_fine/images_orig/"
    output_dir = "/storage/cfmata/deeplab/crf_rnn/crfasrnn_keras/image_results/horse_fine/fcn/"
    input_size = 224
    num_crf_iter = 10

    saved_model_path = 'results/horse_fine/horse_fine_weights.500-0.53'

    #model = get_crfrnn_model_def()
    model = load_model_gby('fcn_RESNET50_8s', input_size, 22, num_crf_iter)
    model.load_weights(saved_model_path)

    im_list = open("lst/horsecoarse_test.txt").readlines()
    im_list = [f[:-1] for f in im_list]
    
    for img in im_list:
        img_data, img_h, img_w = util.get_preprocessed_image(input_dir + img+ ".jpg")
        probs = model.predict(img_data, verbose=False, batch_size=1)[0, :, :, :]
        segmentation = util.get_label_image(probs, img_h, img_w)
        print(output_dir + img)
        segmentation.save(output_dir + img[:-4] + ".png")

if __name__ == '__main__':
    test()

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
from crfrnn_model import get_crfrnn_model_def
from fcn8_model import get_fcn8_model_def
from fcn32_model_gby import fcn_32s
import util
from os import listdir
from os.path import isfile, join
import numpy as np
from scipy import misc
from PIL import Image
from keras import optimizers
from keras import losses
import pickle
import pdb

INPUT_DIR = "/storage/cfmata/deeplab/crf_rnn/crfasrnn_keras/data/pascal_voc12/images_orig/"
GT_DIR = "/storage/cfmata/deeplab/crf_rnn/crfasrnn_keras/data/pascal_voc12/labels_orig/"
RES_DIR = "/storage/gby/semseg/"

def prepare_training_data(img_list_path, im_file_name, label_file_name):
    with open(img_list_path) as f:
        content = f.readlines()
    im_list = sorted([x[42:-5] for x in content]) # Slicing specific to pascal voc trainval lists

    # Prepare image and label data
    inputs, labels = [], []
    i=0
    for name in im_list:
        img_data, img_h, img_w = util.get_preprocessed_image(INPUT_DIR + name + ".jpg")
        inputs.append(img_data)

        if i % 100 == 0:
            print("Processed ", i)
        img_data, img_h, img_w = util.get_preprocessed_label(GT_DIR + name + ".png", 21)
        labels.append(img_data)
        i+=1
    '''
    # Using pickle
    im_file = open(im_file_name, 'wb')
    pickle.dump(inputs,im_file)
    im_file.close()
    label_file = open(label_file_name, 'wb')
    pickle.dump(labels, label_file)
    label_file.close()
    '''

    # Using numpy
    np.save(RES_DIR + "image_data.npy", inputs)
    np.save(RES_DIR + "label_data.npy", labels)
        
def train(im_file_name, label_file_name):
    # Load img and label data
    '''    
    # Using pickle
    input_file = open(im_file_name, 'rb')
    inputs = pickle.load(input_file)
    label_file = open(label_file_name, 'rb')
    labels = pickle.load(label_file)
    '''

    # Using numpy    
    inputs = np.load(RES_DIR + "image_data.npy")
    labels = np.load(RES_DIR + "label_data.npy")
    #pdb.set_trace()
    # Download the model from https://goo.gl/ciEYZi
    saved_model_path = 'crfrnn_keras_model.h5'

    # Initialize model 
    # fcn8:
    # model = get_fcn8_model_def(
    # fcn32
    model = fcn_32s()
    # crfasarnn
    #model = get_crfrnn_model_def()
    #model.load_weights(saved_model_path)

    # Compile model
    #adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    #adam = optimizers.Adam(lr=1e-13, beta_1=0.99, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    #adam = optimizers.Adam(lr=1e-9, beta_1=0.99, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    #model.compile(loss='mean_squared_error', optimizer=adam)
    #model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    #model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    #model.compile(loss=losses.sparse_categorical_crossentropy, optimizer=adam)
    model.compile(loss='categorical_crossentropy', optimizer='sgd')

    #model.fit(x=inputs, y=labels, batch_size=1)
    # Start finetuning
    #for i in range(len(inputs)):
    for i in range(1000):
        print("img ", i)
    #    model.fit(x=inputs[i], y=labels[i], epochs=3, steps_per_epoch=1)
        model.fit(x=inputs[i], y=labels[i], epochs=1, steps_per_epoch=1)


    # Save model weights
    model.save_weights(RES_DIR + 'voc12_weights')

if __name__ == '__main__':
    image_fn, label_fn = "image_data.npy", "label_data.npy"
  #  prepare_training_data("./list/train.txt", image_fn, label_fn)
    #pdb.set_trace()
    train(image_fn, label_fn)

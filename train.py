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
import util
from os import listdir
from os.path import isfile, join
import numpy as np
from scipy import misc
from PIL import Image
from keras import optimizers
import pickle

INPUT_DIR = "/storage/cfmata/deeplab/crf_rnn/crfasrnn_keras/data/pascal_voc12/images_orig/"
GT_DIR = "/storage/cfmata/deeplab/crf_rnn/crfasrnn_keras/data/pascal_voc12/labels_orig/"

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
    np.save("image_data.npy", inputs)
    np.save("label_data.npy", labels)
        
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
    inputs = np.load("image_data.npy")
    labels = np.load("label_data.npy")
    
    # Download the model from https://goo.gl/ciEYZi
    saved_model_path = 'crfrnn_keras_model.h5'

    # Initialize model 
    model = get_crfrnn_model_def()
    model.load_weights(saved_model_path)

    # Compile model
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='mean_squared_error', optimizer=adam)
    
    # Start finetuning
    for i in range(len(inputs)):
        model.fit(x=inputs[i], y=labels[i])

    # Save model weights
    model.save_weights('voc12_weights')

if __name__ == '__main__':
    image_fn, label_fn = "image_data.npy", "label_data.npy"
    prepare_training_data("./list/train.txt", image_fn, label_fn)
    train(image_fn, label_fn)

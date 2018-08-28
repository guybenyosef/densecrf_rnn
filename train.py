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
from keras.callbacks import ModelCheckpoint
import pickle

INPUT_DIR = "/storage/cfmata/deeplab/crf_rnn/crfasrnn_keras/data/pascal_voc12/images_orig/"
GT_DIR = "/storage/cfmata/deeplab/crf_rnn/crfasrnn_keras/data/pascal_voc12/labels_orig/"

def prepare_training_data(img_list_path, im_file_name, label_file_name):
    """ Prepares image data for training crf-as-rnn network.

    @img_list_path (string): path to list containing training image names
    @im_file_name (string): name of file to dump raw image data in
    @label_file_name (string): name of file to dump image label data in
    TODO: add parameter for number of labels
    """
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
    im_file = open(im_file_name + ".p", 'wb')
    pickle.dump(inputs,im_file)
    im_file.close()
    label_file = open(label_file_name + ".p", 'wb')
    pickle.dump(labels, label_file)
    label_file.close()
    '''

    # Using numpy
    np.save(im_file_name + ".npy", inputs)
    np.save(label_file_name + ".npy", labels)
        
def train(im_file_name, label_file_name):
    """ Trains crf-as-rnn network.

    @im_file_name (string): name of file containing raw image data
    @label_file_name (string): name of file containing image label data
    TODO: add parameter for initial weights
    """
    # Load img and label data
    
    # Using pickle
    input_file = open(im_file_name + ".p", 'rb')
    inputs = pickle.load(input_file)
    #label_file = open(label_file_name + ".p", 'rb')
    #labels = pickle.load(label_file)

    # Using numpy
    #inputs = np.load(im_file_name + ".npy")
    labels = np.load(label_file_name + ".npy")

    # Initialize model 
    model = get_crfrnn_model_def()
    #model.load_weights(saved_model_path)

    # Compile model
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='mean_squared_logarithmic_error', optimizer=adam)
    
    # Start finetuning
    for i in range(len(inputs)):
        print("img ", i)
        checkpointer = ModelCheckpoint(filepath='./checkpoint/weights_%d.h5' % i, verbose=1, save_best_only=False, save_weights_only=True, period=5)
        model.fit(x=inputs[i], y=labels[i], epochs=5, steps_per_epoch=1, callbacks=[checkpointer])

    # Save model weights
    model.save_weights('voc12.h5')

if __name__ == '__main__':
    image_fn, label_fn = "image_data", "label_data"
    #prepare_training_data("./list/.txt", image_fn, label_fn)
    train(image_fn, label_fn)

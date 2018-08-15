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

def train():
    # Set input and gt dirs
    input_dir = "/storage/cfmata/deeplab/crf_rnn/crfasrnn_keras/data/pascal_voc12/images_orig/"
    gt_dir = "/storage/cfmata/deeplab/crf_rnn/crfasrnn_keras/data/pascal_voc12/labels_orig/"
    with open("./list/train.txt") as f:
        content = f.readlines()
    im_list = sorted([x[42:-5] for x in content])
    #im_list = sorted([f[41:] for f in listdir(input_dir) if isfile(join(input_dir,f))])

    # Prepare data
    inputs, labels = [], []
    for name in im_list:
        img_data, img_h, img_w = util.get_preprocessed_image(input_dir + name + ".jpg")
        inputs.append(img_data)
        #label_data, img_h, img_w = util.get_preprocessed_image(gt_dir + name + ".png")
        l = Image.open(gt_dir + name + ".png").convert('RGBA')
        labels.append(np.array(l))

    # Download the model from https://goo.gl/ciEYZi
    saved_model_path = 'crfrnn_keras_model.h5'

    # Initialize model
    model = get_crfrnn_model_def()
    model.load_weights(saved_model_path)

    # Compile model
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)

    # Start finetuning
    for i in range(len(inputs)):
        model.fit(inputs[i], labels[i])

    # Save model weights
    model.save_weights('voc12_weights')

if __name__ == '__main__':
    train()

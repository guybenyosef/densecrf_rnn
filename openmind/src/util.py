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

import numpy as np
from PIL import Image

# Pascal VOC color palette for labels
_PALETTE = [0, 0, 0,
            128, 0, 0,
            0, 128, 0,
            128, 128, 0,
            0, 0, 128,
            128, 0, 128,
            0, 128, 128,
            128, 128, 128,
            64, 0, 0,
            192, 0, 0,
            64, 128, 0,
            192, 128, 0,
            64, 0, 128,
            192, 0, 128,
            64, 128, 128,
            192, 128, 128,
            0, 64, 0,
            128, 64, 0,
            0, 192, 0,
            128, 192, 0,
            0, 64, 128,
            128, 64, 128,
            0, 192, 128,
            128, 192, 128,
            64, 64, 0,
            192, 64, 0,
            64, 192, 0,
            192, 192, 0]

_IMAGENET_MEANS = np.array([123.68, 116.779, 103.939], dtype=np.float32)  # RGB mean values
_PASCALVOC_MEANS = np.array([104.008, 116.669, 122.675], dtype=np.float32)

def get_preprocessed_image(file_name):
    """ Reads an image from the disk, pre-processes it by subtracting mean etc. and
    returns a numpy array that's ready to be fed into a Keras model.

    Note: This method assumes 'channels_last' data format in Keras.
    """

    im = np.array(Image.open(file_name)).astype(np.float32)
    assert im.ndim == 3, 'Only RGB images are supported.'
    #im = im - _IMAGENET_MEANS
    im = im - _PASCALVOC_MEANS
    im = im[:, :, ::-1]  # Convert to BGR
    img_h, img_w, img_c = im.shape
    assert img_c == 3, 'Only RGB images are supported.'
    if img_h > 500 or img_w > 500:
        raise ValueError('Please resize your images to be not bigger than 500 x 500.')

    pad_h = 500 - img_h
    pad_w = 500 - img_w
    im = np.pad(im, pad_width=((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
    return np.expand_dims(im.astype(np.float32), 0), img_h, img_w

def get_preprocessed_label(file_name, num_labels):
    """ Reads an image from disk, preprocess by converting to array of same dimensions with a binary layer for each segmentation class.
    """
    img = Image.open(file_name).convert("RGBA")
    img_array = np.array(img)[:,:,0] # Take only a single layer

    # Pad to make 500 x 500
    img_h, img_w = np.shape(img_array)
    pad_h = 500 - img_h
    pad_w = 500 - img_w
    padded_im = np.pad(img_array, pad_width = ((0,pad_h), (0,pad_w)), mode="constant", constant_values=0)

    res = np.zeros((500, 500, num_labels))
    for i in range(500):
        for j in range(500):
            k = padded_im[i,j]
            res[i,j,k] = 1

    '''
    # Check the resulting array
    for i in range(num_labels):
        print(i)
        print(np.shape(res[:,:,i]))
        f = res[:,:,i].flatten()
        print(f)
        s = 0
        for elt in f:
            if elt != 0:
                s+=1
        print("num nonzero: ", s)
    '''
    
    res = res.reshape((-1, 500, 500, num_labels)) # Need to have 4 dimensions
    return res, img_h, img_w

def get_label_image(probs, img_h, img_w):
    """ Returns the label image (PNG with Pascal VOC colormap) given the probabilities.

    Note: This method assumes 'channels_last' data format.
    """

    labels = probs.argmax(axis=2).astype('uint8')[:img_h, :img_w]
    print(labels)
    label_im = Image.fromarray(labels, 'P')
    label_im.putpalette(_PALETTE)
    return label_im

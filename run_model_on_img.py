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
import util
import os
import pdb

def main(saved_model_path,input_file_path):
    input_file_dir, input_file_basename = os.path.split(input_file_path)
    input_file_name, input_file_ext = os.path.splitext(input_file_basename)
    output_file_path = input_file_dir+input_file_name+'_labels.png'

    print('runnnig model %s on image %s, write to %s' % (saved_model_path, input_file_path, output_file_path))

    # Download the model from https://goo.gl/ciEYZi
    #saved_model_path = 'crfrnn_keras_model.h5'

    #model = get_crfrnn_model_def()
    model = get_fcn8_model_def()
    model.load_weights(saved_model_path)

    img_data, img_h, img_w = util.get_preprocessed_image(input_file_path)
    probs = model.predict(img_data, verbose=False)[0, :, :, :]
    pdb.set_trace()
    segmentation = util.get_label_image(probs, img_h, img_w)
    segmentation.save(output_file_path)


if __name__ == '__main__':
    saved_model_path, input_file_path = sys.argv[1],sys.argv[2]
    main(saved_model_path, input_file_path)


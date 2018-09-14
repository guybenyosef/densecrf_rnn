# my own keras implementation for fcn8

import numpy as np
import pdb

from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, Add, Cropping2D
from keras.initializers import Constant
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras import backend as K
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import argparse
import copy
import os

import util

#from fcn8_model import get_fcn8_model_def
#from crfrnn_model import get_crfrnn_model_def

RES_DIR = "/storage/gby/semseg/"

nb_classes = 21
INPUT_SIZE = None# None # 496 # 510, 500

# ------------------
# Extract files:
# ------------------
def getImageArr(path, width, height):
    img_org = Image.open(path)
    img = np.float32(img_org.resize((width, height))) / 127.5 - 1
    return img

def getSegmentationArr(path, width, height):
    seg_labels = np.zeros((height, width, nb_classes))
    img_org = Image.open(path)
    img = img_org.resize((width, height))
    img = img[:, :, 0]

    for c in range(nb_classes):
        seg_labels[:, :, c] = (img == c).astype(int)
    ##seg_labels = np.reshape(seg_labels, ( width*height,nClasses  ))
    return seg_labels

def getImageLabelsPairs(dir_img,dir_lbls, input_width, input_height, output_width, output_height):
    images = os.listdir(dir_img)
    images.sort()
    segmentations = os.listdir(dir_lbls)
    segmentations.sort()

    X = []
    Y = []
    for im, seg in zip(images, segmentations):
        X.append(getImageArr(dir_img + im, input_width, input_height))
        Y.append(getSegmentationArr(dir_lbls + seg, nb_classes, output_width, output_height))

    X, Y = np.array(X), np.array(Y)
    print(X.shape, Y.shape)
    return [X,Y]

def load_image(path):
    img_org = Image.open(path)
    w, h = img_org.size
    if INPUT_SIZE == None:
        img = img_org.resize(((w // 32) * 32, (h // 32) * 32))
    else:
        # if the input size is fixed:
        img = img_org.resize((INPUT_SIZE,INPUT_SIZE))
    img = np.array(img, dtype=np.float32)
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    return x

def load_label(path):
    img_org = Image.open(path)
    w, h = img_org.size
    if INPUT_SIZE == None:
        img = img_org.resize(((w//32)*32, (h//32)*32))
    else:
        # if the input size is fixed:
        img = img_org.resize((INPUT_SIZE, INPUT_SIZE))
    img = np.array(img, dtype=np.uint8)
    img[img==255] = 0
    y = np.zeros((1, img.shape[0], img.shape[1], nb_classes), dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            y[0, i, j, img[i][j]] = 1
    return y

def generate_arrays_from_file(path, image_dir, label_dir):
    while 1:
        f = open(path)
        for line in f:
            filename = line.rstrip('\n')
            path_image = os.path.join(image_dir, filename+'.jpg')
            path_label = os.path.join(label_dir, filename+'.png')
            x = load_image(path_image)
            y = load_label(path_label)
            yield (x, y)
        f.close()

def extract_arrays_from_file(path, image_dir, label_dir):
    X = []
    Y = []
    f = open(path)
    content = f.readlines()
    line_count = len(content)
    print('extracting %d image and label files..' % line_count)
    for ii in range(line_count):
        line = content[ii]
        filename = line.rstrip('\n')
        #print(filename)
        path_image = os.path.join(image_dir, filename+'.jpg')
        path_label = os.path.join(label_dir, filename+'.png')
        x = load_image(path_image)
        y = load_label(path_label)
        #pdb.set_trace()
        X.append(x)
        Y.append(y)
    f.close()
 #   X, Y = np.array(X), np.array(Y)
    print('done!')
    return [X,Y]

# -----------------------
# Initiate model weights
# -----------------------

# Bilinear interpolation (reference: https://github.com/warmspringwinds/tf-image-segmentation/blob/master/tf_image_segmentation/utils/upsampling.py)
def bilinear_upsample_weights(factor, number_of_classes):
    filter_size = factor*2 - factor%2
    factor = (filter_size + 1) // 2
    if filter_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:filter_size, :filter_size]
    upsample_kernel = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weights = np.zeros((filter_size, filter_size, number_of_classes, number_of_classes),
                       dtype=np.float32)
    for i in range(number_of_classes):
        weights[:, :, i, i] = upsample_kernel
    return weights

# -----------------------
# Model design
# -----------------------

def fcn_32s():
    inputs = Input(shape=(None, None, 3))
    vgg16 = VGG16(weights='imagenet', include_top=False, input_tensor=inputs)
    x = Conv2D(filters=nb_classes,
               kernel_size=(1, 1))(vgg16.output)
    x = Conv2DTranspose(filters=nb_classes,
                        kernel_size=(64, 64),
                        strides=(32, 32),
                        padding='same',
                        activation='sigmoid',
                        kernel_initializer=Constant(bilinear_upsample_weights(32, nb_classes)))(x)
    model = Model(inputs=inputs, outputs=x)
    for layer in model.layers[:15]:
        layer.trainable = False
    return model

def fcn_8s():
    """ Returns Keras FCN-8 model definition.

      """
    fcn32_flag = False

    inputs = Input(shape=(INPUT_SIZE, INPUT_SIZE, 3))
    # Start from VGG16 layers
    vgg16 = VGG16(weights='imagenet', include_top=False, input_tensor=inputs)

    # Skip connections from pool3, 256 channels
    vgg16_upto_intermediate_layer_pool3 = Model(inputs=vgg16.input, outputs=vgg16.get_layer('block3_pool').output)
    score_pool3 = vgg16_upto_intermediate_layer_pool3.output
    # 1x1 conv layer to reduce number of channels to nb_classes:
    score_pool3c = Conv2D(filters=nb_classes,kernel_size=(1, 1),name="score_pool3c")(score_pool3)

    # Skip connections from pool4, 512 channels
    vgg16_upto_intermediate_layer_pool4 = Model(inputs=vgg16.input, outputs=vgg16.get_layer('block4_pool').output)
    score_pool4 = vgg16_upto_intermediate_layer_pool4.output
    # 1x1 conv layer to reduce number of channels to nb_classes:
    score_pool4c = Conv2D(filters=nb_classes, kernel_size=(1, 1))(score_pool4)

    # score from the top vgg16 layer:
    score7 = vgg16.output
    score7c = Conv2D(filters=nb_classes,kernel_size=(1, 1))(score7)
    score7c_upsample = Conv2DTranspose(filters=nb_classes,
                                       kernel_size=(4, 4),
                                       strides=(2, 2),
                                       padding='same',
                                       activation = None,
                                       kernel_initializer = Constant(bilinear_upsample_weights(2, nb_classes)),
                                       name="score_pool7c_upsample")(score7c)

    # Fuse scores
    score_7_4 = Add()([score7c_upsample, score_pool4c])
    # upsample:
    score_7_4_up = Conv2DTranspose(filters=nb_classes,
                                      kernel_size=(4, 4),
                                      strides=(2, 2),
                                      padding='same',
                                      activation= None,
                                      kernel_initializer=Constant(bilinear_upsample_weights(2, nb_classes)),
                                      name="score_7_4_up")(score_7_4)

    # Fuse scores
    score_7_4_3 = Add()([score_7_4_up, score_pool3c])
    # upsample:
    score_7_4_3_up = Conv2DTranspose(filters=nb_classes,
                                   kernel_size=(16, 16),
                                   strides=(8, 8),
                                   padding='same',
                                   activation='sigmoid',
                                   kernel_initializer=Constant(bilinear_upsample_weights(8, nb_classes)),
                                   name="score_7_4_3_up")(score_7_4_3)

    # # -- There's another way to match the tensor sizes from earlier layers, using a Cropping2D layer --
    # # e.g., for fcn-16, we can crop layer 'score_pool4c' to get the same size as layer 'score_7c'
    # score_pool4c_cropped = Cropping2D((5+3, 5+3))(score_pool4c)
    # # fuse layers,
    # score_7_4_cropped = Add()([score7c, score_pool4c_cropped])
    # # then upsample to input size:
    # x = Conv2DTranspose(filters=nb_classes,
    #                     kernel_size=(64, 64),
    #                     strides=(32+2,32+2),
    #                     padding='same',
    #                     activation='sigmoid',
    #                     kernel_initializer=Constant(bilinear_upsample_weights(32, nb_classes)))(score_7_4_cropped)

    # Creating the model:
    model = Model(inputs=inputs, outputs=score_7_4_3_up)
    # # -- and this is fcn-32: --
    if(fcn32_flag):
        score7c_upsample_32 = Conv2DTranspose(filters=nb_classes,
                                                     kernel_size=(64, 64),
                                                     strides=(32, 32),
                                                     padding='same',
                                                     activation='sigmoid',
                                                     kernel_initializer=Constant(bilinear_upsample_weights(32, nb_classes)),
                                                     name="score_pool7c_upsample_32")(score7c)
        model = Model(inputs=inputs, outputs=score7c_upsample_32)


    # Fixing weighs in lower layers
    for layer in model.layers[:15]:  # sometimes I use it, sometimes not.
        layer.trainable = False
    return model

def model_predict(model, input_path, output_path):
    img_org = Image.open(input_path)
    w, h = img_org.size
    img = img_org.resize(((w//32)*32, (h//32)*32))
    img = np.array(img, dtype=np.float32)
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    pdb.set_trace()
    pred = model.predict(x)
    pred = pred[0].argmax(axis=-1).astype(np.uint8)
    img = Image.fromarray(pred, mode='P')
    img = img.resize((w, h))
    palette_im = Image.open('palette.png')
    img.palette = copy.copy(palette_im.palette)
    img.save(output_path)

def model_predict_gby(model, input_path, output_path):
    img_org = Image.open(input_path)
    ww, hh = img_org.size
    if INPUT_SIZE==None:
        img = img_org.resize(((ww//32)*32, (hh//32)*32))
    else:
        # if the input size is fixed:
        img = img_org.resize((INPUT_SIZE, INPUT_SIZE))
    img = np.array(img, dtype=np.float32)
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    probs = model.predict(x)
    segmentation = util.get_label_image(probs[0,:,:,:], hh, ww)
    segmentation.save(output_path)

def IoU(Yi, y_predi):
    ## mean Intersection over Union
    ## Mean IoU = TP/(FN + TP + FP)

    IoUs = []
    Nclass = int(np.max(Yi)) + 1
    for c in range(nb_classes):
        TP = np.sum((Yi == c) & (y_predi == c))
        FP = np.sum((Yi != c) & (y_predi == c))
        FN = np.sum((Yi == c) & (y_predi != c))
        IoU = TP / float(TP + FP + FN)
        #print("class {:02.0f}: #TP={:6.0f}, #FP={:6.0f}, #FN={:5.0f}, IoU={:4.3f}".format(c, TP, FP, FN, IoU))
        IoUs.append(IoU)
    #pdb.set_trace()
    mIoU = np.nanmean(IoUs)
    # print("_________________")
    # print("Mean IoU: {:4.3f}".format(mIoU))
    return mIoU

    #>>IoU(y_testi, y_predi)

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

    model = fcn_8s()
    model.summary()
    #pdb.set_trace()
    #model = get_fcn8_model_def()
    #model = get_crfrnn_model_def()
    #saved_model_path = 'crfrnn_keras_model.h5'
    #model.load_weights(saved_model_path)

    # compile 1:
    model.compile(loss="binary_crossentropy", optimizer='sgd',metrics=['accuracy'])
    #model.compile(loss="categorical_crossentropy", optimizer='sgd', metrics=['accuracy'])
    # compile 2:
    # model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
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
    [val_imgs,val_labels] = extract_arrays_from_file(args.val_data, args.image_dir, args.label_dir)
    #pdb.set_trace()
    # or simply,
    # train_imgs = np.load(RES_DIR + "image_data.npy")[:,0,:,:]
    # train_labels = np.load(RES_DIR + "label_data.npy")[:,0,:,:]

    for epoch in range(50):

        hist1 = model.fit_generator(
            generate_arrays_from_file(args.train_data, args.image_dir, args.label_dir),
            validation_data=generate_arrays_from_file(args.val_data, args.image_dir, args.label_dir),
            steps_per_epoch=nb_data,
            validation_steps=nb_data,
            epochs=1)

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
            X_test = val_imgs[0]
            y_test = val_labels[0]
            # pdb.set_trace()
            y_pred = model.predict(X_test)
            y_predi = np.argmax(y_pred, axis=3)
            y_testi = np.argmax(y_test, axis=3)
            # print(y_testi.shape, y_predi.shape)
            mIoU += IoU(y_testi, y_predi)
        mIoU = mIoU/len(val_imgs)

        print("_________________")
        print("Mean IoU: {:4.3f}".format(mIoU))

        # Predict 1 test exmplae and save:
        model_predict_gby(model, 'image.jpg', 'predict-{}.png'.format(epoch))
        model.save_weights(RES_DIR + 'voc12_weights-{}'.format(epoch))

    model.save_weights(RES_DIR + 'voc12_weights')



# usage:
# >>python src/fcn8_model_gby.py ./list/train2.txt ./list/val2.txt /storage/gby/datasets/pascal_voc12/images_orig/ /storage/gby/datasets/pascal_voc12/labels_orig/
# comment: /storage/gby/datasets/pascal_voc12/ is a copy of Cristina's folder
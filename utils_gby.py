# my own keras implementation for fcn8

import pdb
from keras.applications.vgg16 import preprocess_input, decode_predictions
#from keras.applications.resnet50 import preprocess_input, decode_predictions
from PIL import Image
import numpy as np
import copy
import os
import sys
sys.path.insert(1, './src')
import util
import seaborn as sns
## seaborn has white grid by default so I will get rid of this.
sns.set_style("whitegrid", {'axes.grid' : False})

# ------------------
# Extract files:
# ------------------
def getImageArr(path, width, height):
    img_org = Image.open(path)
    img = np.float32(np.array(img_org.resize((width, height)))) / 127.5 - 1
    return img

def getSegmentationArr(path, width, height,nb_classes):
    seg_labels = np.zeros((height, width, nb_classes))
    img_org = Image.open(path)
    img = np.array(img_org.resize((width, height)))
    #img = img[:, :, 0]

    for c in range(nb_classes):
        seg_labels[:, :, c] = (img == c).astype(int)
    ##seg_labels = np.reshape(seg_labels, ( width*height,nClasses  ))
    return seg_labels

def getImageLabelsPairs(dir_img,dir_lbls, input_width, input_height, output_width, output_height,nb_classes):
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

def load_image(path,INPUT_SIZE):
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

def load_label(path,INPUT_SIZE,nb_classes):
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

def generate_arrays_from_file(path, image_dir, label_dir, INPUT_SIZE,nb_classes):
    while 1:
        f = open(path)
        for line in f:
            filename = line.rstrip('\n')
            path_image = os.path.join(image_dir, filename+'.jpg')
            path_label = os.path.join(label_dir, filename+'.png')
            x = load_image(path_image,INPUT_SIZE)
            y = load_label(path_label,INPUT_SIZE,nb_classes)
            yield (x, y)
        f.close()

def extract_arrays_from_file(path, image_dir, label_dir, INPUT_SIZE,nb_classes):
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
        x = load_image(path_image, INPUT_SIZE)
        y = load_label(path_label, INPUT_SIZE, nb_classes)
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
# Predict
# -----------------------
def model_predict(model, input_path, output_path):
    img_org = Image.open(input_path)
    w, h = img_org.size
    img = img_org.resize(((w//32)*32, (h//32)*32))
    img = np.array(img, dtype=np.float32)
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    pred = model.predict(x)
    pred = pred[0].argmax(axis=-1).astype(np.uint8)
    img = Image.fromarray(pred, mode='P')
    img = img.resize((w, h))
    palette_im = Image.open('palette.png')
    img.palette = copy.copy(palette_im.palette)
    img.save(output_path)

def model_predict_gby(model, input_path, output_path,INPUT_SIZE):
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
    #pdb.set_trace()
    print(probs)
    segmentation = util.get_label_image(probs[0,:,:,:], hh, ww)
    segmentation.save(output_path)

# -----------------------
# Evaluation
# -----------------------
def IoU(Yi, y_predi,nb_classes):
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

def IoU_ver2(Yi,y_predi):
    ## mean Intersection over Union
    ## Mean IoU = TP/(FN + TP + FP)

    IoUs = []
    Nclass = int(np.max(Yi)) + 1
    for c in range(Nclass):
        TP = np.sum( (Yi == c)&(y_predi==c) )
        FP = np.sum( (Yi != c)&(y_predi==c) )
        FN = np.sum( (Yi == c)&(y_predi != c))
        IoU = TP/float(TP + FP + FN)
        print("class {:02.0f}: #TP={:6.0f}, #FP={:6.0f}, #FN={:5.0f}, IoU={:4.3f}".format(c,TP,FP,FN,IoU))
        IoUs.append(IoU)
    mIoU = np.mean(IoUs)
    print("_________________")
    print("Mean IoU: {:4.3f}".format(mIoU))


def give_color_to_seg_img(seg, n_classes):
    '''
    seg : (input_width,input_height,3)
    '''

    if len(seg.shape) == 3:
        seg = seg[:, :, 0]
    seg_img = np.zeros((seg.shape[0], seg.shape[1], 3)).astype('float')
    colors = sns.color_palette("hls", n_classes)

    for c in range(n_classes):
        segc = (seg == c)
        seg_img[:, :, 0] += (segc * (colors[c][0]))
        seg_img[:, :, 1] += (segc * (colors[c][1]))
        seg_img[:, :, 2] += (segc * (colors[c][2]))

    return (seg_img)

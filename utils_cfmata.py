# utils

import pdb
from keras import backend as K
#from keras.applications.vgg16 import preprocess_input, decode_predictions
# [DEP] from scipy.misc import imsave
from keras.applications.resnet50 import preprocess_input, decode_predictions
from PIL import Image
import scipy.io
from scipy import misc
from skimage.future import graph
import numpy as np
import copy
import os, sys
import time
sys.path.insert(1, './src')
import util
import seaborn as sns
## seaborn has white grid by default so I will get rid of this.
sns.set_style("whitegrid", {'axes.grid' : False})
from os import listdir
from os.path import isfile, join

# ------------------
# Generate segmentations:
# ------------------

def create_RAG(image_path, compactness,num_segments,threshold):
    img_org = Image.open(image_path)
    img = np.array(img_org, dtype=np.float32)
    labels1 = segmentation.slic(img_org, compactness=compactness, n_segments=num_segments)
    out1 = color.label2rgb(labels1, img, kind='avg')

    g = graph.rag_mean_color(img, labels1)
    labels2 = graph.cut_threshold(labels1, g, threshold)
    out2 = color.label2rgb(labels2, img, kind='avg')
    #misc.imsave('./RAG_other/'+name, out2)
    return out2

def load_RAG_segmentations(dataset, INPUT_SIZE):
    img_names, folder_array = [], []
    if dataset == 'horsecoarse':
        RAG_path = './data/horse_coarse_parts/RAG/'
        img_names = [f for f in listdir(RAG_path) if isfile(join(RAG_path, f))]
    for name in img_names:
        folder_array.append(load_image(RAG_path+name,INPUT_SIZE)[0,:,:])
    folder_array = np.array(folder_array)
    return folder_array[:325], folder_array[325:]

def load_segmentations(dataset, INPUT_SIZE):
    img_names, folder_array = [], []
    sp_seg_path = 'data/horse_coarse_parts/sp_seg/'
    if dataset == 'horsecoarse':
        with open('lst/'+dataset+'_train.txt', 'r') as f:
            img_names.extend([line.strip() for line in f])
        with open('lst/'+dataset+'_test.txt', 'r') as f:
            img_names.extend([line.strip() for line in f])
    for name in img_names:
        name = img_names[0]
        data = np.load(sp_seg_path+name + "_sp.npy")
        folder_array.append(data)
    folder_array = np.array(folder_array)
    return folder_array[:325], folder_array[325:]
    
def load_mat_segmentations(dataset, INPUT_SIZE):
    img_names, folder_array = [], []
    sp_seg_path = '/storage/cfmata/deeplab/datasets/superpixels/'
    if dataset == 'horsecoarse':
        with open('lst/'+dataset+'_train.txt', 'r') as f:
            img_names.extend([line.strip() for line in f])
        with open('lst/'+dataset+'_test.txt', 'r') as f:
            img_names.extend([line.strip() for line in f])
    for name in img_names:
        name = img_names[0]
        mat = scipy.io.loadmat(sp_seg_path+name+'_sp.mat')
        folder_array.append(mat)
    folder_array = np.array(folder_array)
    return folder_array[:325], folder_array[325:]
    
# ------------------
# Extract files:
# ------------------
def dir2list(names,listname):
    #names = os.listdir(dirname)
    f = open(listname, 'w')
    for name in names:
        f.write(name.split(".")[0] + '\n')
    f.close()

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
    #print(path)
    #print(np.unique(img))
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
# For loss functions:
# -----------------------
def compute_median_frequency_reweighting(Yi):
    Nclass = int(np.max(Yi)) + 1
    count_labels = np.zeros(Nclass) #[0] * len(Nclass)
    for c in range(Nclass):
        count_labels[c] = np.sum(Yi == c)
        #print("class {:02.0f}: #pixels={:6.0f}".format(c,count_labels[c]))

    median_count_labels = np.median(count_labels)  # equivalent to median freq
    #print(median_count_labels)

    median_frequency_coef = median_count_labels/count_labels

    print("median_frequency_reweighting:")
    print(median_frequency_coef)

    return median_frequency_coef

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

# -----------------------
# Visualization
# -----------------------
# from: https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def visualize_conv_filters(model, INPUT_SIZE, layer_name):

    kept_filters = []
    for filter_index in range(10):
        # we only scan through the first 200 filters,
        # but there are actually 512 of them
        print('Processing filter %d' % filter_index)
        start_time = time.time()

        # dimensions of the generated pictures for each filter.
        if(INPUT_SIZE == None):
            img_width = 128
            img_height = 128
        else:
            img_width = INPUT_SIZE
            img_height = INPUT_SIZE

        # this is the placeholder for the input images
        input_img = model.input

        # get the symbolic outputs of each "key" layer (we gave them unique names).
        layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])

        # build a loss function that maximizes the activation
        # of the nth filter of the layer considered
        layer_output = layer_dict[layer_name].output
        if K.image_data_format() == 'channels_first':
            loss = K.mean(layer_output[:, filter_index, :, :])
        else:
            loss = K.mean(layer_output[:, :, :, filter_index])

        # compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, input_img)[0]

        # normalization trick: we normalize the gradient
        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])

        # step size for gradient ascent
        step = 1.

        # we start from a gray image with some noise
        if K.image_data_format() == 'channels_first':
            input_img_data = np.random.random((1, 3, img_width, img_height))
        else:
            input_img_data = np.random.random((1, img_width, img_height, 3))
        input_img_data = (input_img_data - 0.5) * 20 + 128

        # run gradient ascent for 20 steps
        for i in range(20):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * step

            print('Current loss value:', loss_value)
            if loss_value <= 0.:
                # some filters get stuck to 0, we can skip them
                break

        # decode the resulting input image
        if loss_value > 0:
            img = deprocess_image(input_img_data[0])
            kept_filters.append((img, loss_value))
        end_time = time.time()
        print('Filter %d processed in %ds' % (filter_index, end_time - start_time))

    # pdb.set_trace()
    # imsave('%s_filter_%d.png' % (layer_name, filter_index), img)

    # we will stich the best 64 filters on a 8 x 8 grid.
    n = 2

    # the filters that have the highest loss are assumed to be better-looking.
    # we will only keep the top 64 filters.
    kept_filters.sort(key=lambda x: x[1], reverse=True)
    kept_filters = kept_filters[:n * n]

    # build a black picture with enough space for
    # our 8 x 8 filters of size 128 x 128, with a 5px margin in between
    margin = 5
    width = n * img_width + (n - 1) * margin
    height = n * img_height + (n - 1) * margin
    stitched_filters = np.zeros((width, height, 3))

    # fill the picture with our saved filters
    for i in range(n):
        for j in range(n):
            img, loss = kept_filters[i * n + j]
            width_margin = (img_width + margin) * i
            height_margin = (img_height + margin) * j
            stitched_filters[
            width_margin: width_margin + img_width,
            height_margin: height_margin + img_height, :] = img

    # save the result to disk
   # imsave('%s_%s_stitched_filters_%dx%d.png' % (model.name,layer_name,n, n), stitched_filters)

# utils

import pdb
from keras import backend as K

# NOTE:   VGG16 or RESNET50
from keras.applications.vgg16 import preprocess_input, decode_predictions
#from keras.applications.resnet50 import preprocess_input, decode_predictions

# [DEP] from scipy.misc import imsave
from PIL import Image
#from scipy import misc
#from skimage.future import graph
import numpy as np
import copy
import os, time, sys
import random
import cv2
import src.util
#import seaborn as sns
## seaborn has white grid by default so I will get rid of this.
#sns.set_style("whitegrid", {'axes.grid' : False})

# ------------------
# Extract files:
# ------------------
def dir2list(names,listname):
    #names = os.listdir(dirname)
    f = open(listname, 'w')
    for name in names:
        f.write(name.split(".")[0] + '\n')
    f.close()

# def getImageArr(path, width, height):
#     img_org = Image.open(path)
#     img = np.float32(np.array(img_org.resize((width, height)))) / 127.5 - 1
#     return img
#
# def getSegmentationArr(path, width, height,nb_classes):
#     seg_labels = np.zeros((height, width, nb_classes))
#     img_org = Image.open(path)
#     img = np.array(img_org.resize((width, height)))
#     #img = img[:, :, 0]
#
#     for c in range(nb_classes):
#         seg_labels[:, :, c] = (img == c).astype(int)
#     ##seg_labels = np.reshape(seg_labels, ( width*height,nClasses  ))
#     return seg_labels

# def getImageLabelsPairs(dir_img,dir_lbls, input_width, input_height, output_width, output_height,nb_classes):
#     images = os.listdir(dir_img)
#     images.sort()
#     segmentations = os.listdir(dir_lbls)
#     segmentations.sort()
#
#     X = []
#     Y = []
#     for im, seg in zip(images, segmentations):
#         X.append(getImageArr(dir_img + im, input_width, input_height))
#         Y.append(getSegmentationArr(dir_lbls + seg, nb_classes, output_width, output_height))
#
#     X, Y = np.array(X), np.array(Y)
#     print(X.shape, Y.shape)
#     return [X,Y]
#
def load_image(img_org,INPUT_SIZE):
    #img_org = Image.open(path)
    #w, h = img_org.size
    h, w, c = img_org.shape
    if INPUT_SIZE == None:
        #img = img_org.resize(((w // 32) * 32, (h // 32) * 32))
        img = cv2.resize(img_org, ((w // 32) * 32, (h // 32) * 32))
    else:
        # if the input size is fixed:
        #img = img_org.resize((INPUT_SIZE,INPUT_SIZE))
        img = cv2.resize(img_org, (INPUT_SIZE, INPUT_SIZE))
    img = np.array(img, dtype=np.float32)
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    return x

def load_label(img_org,INPUT_SIZE,nb_classes):
    #img_org = Image.open(path)
    #w, h = img_org.size
    h, w, c = img_org.shape
    if INPUT_SIZE == None:
        #img = img_org.resize(((w//32)*32, (h//32)*32))
        img = cv2.resize(img_org,((w // 32) * 32, (h // 32) * 32))
    else:
        # if the input size is fixed:
        #img = img_org.resize((INPUT_SIZE, INPUT_SIZE))
        img = cv2.resize(img_org,(INPUT_SIZE, INPUT_SIZE))
    img = np.array(img, dtype=np.uint8)
    img[img==255] = 0
    y = np.zeros((1, img.shape[0], img.shape[1], nb_classes), dtype=np.float32)
    #print(path)
    #print(np.unique(img))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            y[0, i, j, img[i][j]] = 1
    return y

def generate_arrays_from_file(path, image_dir, label_dir, INPUT_SIZE,nb_classes, dataaug_args):
    batch_size = dataaug_args.batchsize
    sample_counter = 0
    X = []
    Y = []
    while 1:
        f = open(path)
        for line in f:
            filename = line.rstrip('\n')
            path_image = os.path.join(image_dir, filename+'.jpg')
            img_org = cv2.imread(path_image)
            path_label = os.path.join(label_dir, filename+'.png')
            lbl_org = cv2.imread(path_label)

            if dataaug_args is None:
                img, lbl = img_org, lbl_org
            else:
                img, lbl = data_augmentation(dataaug_args, img_org, lbl_org)

            x = load_image(img,INPUT_SIZE)
            y = load_label(lbl,INPUT_SIZE,nb_classes)

            X.append(x)
            Y.append(y)

            sample_counter += 1
            #print(sample_counter)

            if sample_counter == batch_size:
                X, Y = np.array(X)[:, 0, :, :, :], np.array(Y)[:, 0, :, :, :]
                yield (X, Y)
                sample_counter = 0
                X = []
                Y = []
    #        yield (x, y)
        f.close()

def extract_arrays_from_file(path, image_dir, label_dir, INPUT_SIZE, nb_classes, dataaug_args):
    X = []
    Y = []
    f = open(path)
    content = f.readlines()
    line_count = len(content)
    print('extracting %d image and label files..' % line_count)
    for ii in range(line_count):
        line = content[ii]
        filename = line.rstrip('\n')
        # load image and label:
        path_image = os.path.join(image_dir, filename+'.jpg')
        img_org = cv2.imread(path_image)
        path_label = os.path.join(label_dir, filename+'.png')
        lbl_org = cv2.imread(path_label)
        #
        #data_augmentation(dataaug_args, img_org, lbl_org)
        #
        x = load_image(img_org, INPUT_SIZE)
        y = load_label(lbl_org, INPUT_SIZE, nb_classes)

        X.append(x)
        Y.append(y)
    f.close()
 #   X, Y = np.array(X), np.array(Y)
    print('done!')
    return [X,Y]


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


def load_segmentations(dirpath,list_of_images,INPUT_SIZE):
    folder_array = []
    #img_names = dir2list(dirpath)
    f = open(list_of_images,'r')
    content = f.readlines()
    for indx in range(len(content)):
        line = content[indx]
        filename = line.rstrip('\n')
        # npy version:
        segs = np.load(dirpath + filename + "_sp.npy")
        #print(dirpath+filename+"_sp.npy")
        #cv2.resize(segs, dsize=(INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_CUBIC)
        #if INPUT_SIZE!=512:
        #    segs = np.resize(segs, [INPUT_SIZE, INPUT_SIZE])
            #print("segs ", segs)
        folder_array.append(segs)
        # jpg version
        #segs = load_image(cv2.imread(dirpath + filename + '.jpg'), INPUT_SIZE)
        #folder_array.append(segs[0, :, :])

    folder_array = np.array(folder_array)
    return folder_array

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
# Data augmentation
# -----------------------
def data_augmentation(args, input_image, output_image):
    # Data augmentation
    #input_image, output_image = utils.random_crop(input_image, output_image, args.crop_height, args.crop_width)

    if args.h_flip and random.randint(0,1):
        input_image = cv2.flip(input_image, 1)
        output_image = cv2.flip(output_image, 1)
    if args.v_flip and random.randint(0,1):
        input_image = cv2.flip(input_image, 0)
        output_image = cv2.flip(output_image, 0)
    if args.brightness:
        factor = 1.0 + random.uniform(-1.0*args.brightness, args.brightness)
        table = np.array([((i / 255.0) * factor) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        input_image = cv2.LUT(input_image, table)
    if args.rotation:
        angle = random.uniform(-1*args.rotation, args.rotation)
    if args.rotation:
        M = cv2.getRotationMatrix2D((input_image.shape[1]//2, input_image.shape[0]//2), angle, 1.0)
        input_image = cv2.warpAffine(input_image, M, (input_image.shape[1], input_image.shape[0]), flags=cv2.INTER_NEAREST)
        output_image = cv2.warpAffine(output_image, M, (output_image.shape[1], output_image.shape[0]), flags=cv2.INTER_NEAREST)

    return input_image, output_image

# -----------------------
# Predict
# -----------------------
# def model_predict(model, input_path, output_path):
#     img_org = Image.open(input_path)
#     w, h = img_org.size
#     img = img_org.resize(((w//32)*32, (h//32)*32))
#     img = np.array(img, dtype=np.float32)
#     x = np.expand_dims(img, axis=0)
#     x = preprocess_input(x)
#     pred = model.predict(x)
#     pred = pred[0].argmax(axis=-1).astype(np.uint8)
#     img = Image.fromarray(pred, mode='P')
#     img = img.resize((w, h))
#     palette_im = Image.open('palette.png')
#     img.palette = copy.copy(palette_im.palette)
#     img.save(output_path)
#
# def model_predict_gby(model, input_path, output_path,INPUT_SIZE):
#     img_org = Image.open(input_path)
#     ww, hh = img_org.size
#     if INPUT_SIZE==None:
#         img = img_org.resize(((ww//32)*32, (hh//32)*32))
#     else:
#         # if the input size is fixed:
#         img = img_org.resize((INPUT_SIZE, INPUT_SIZE))
#     img = np.array(img, dtype=np.float32)
#     x = np.expand_dims(img, axis=0)
#     x = preprocess_input(x)
#     probs = model.predict(x)
#     #pdb.set_trace()
#     print(probs)
#     segmentation = util.get_label_image(probs[0,:,:,:], hh, ww)
#     segmentation.save(output_path)

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

    # setting the background color to be black
    colors[0] = (0, 0, 0)

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


import os
import numpy as np
#from utils_gby import generate_arrays_from_file,extract_arrays_from_file,IoU,model_predict_gby,getImageArr,getSegmentationArr,dir2list
from utils_gby import dir2list,extract_arrays_from_file,generate_arrays_from_file
from sklearn.utils import shuffle
#from random import shuffle

class datas(object):
  def __init__(self, X_train, y_train, X_test, y_test, input_size, nb_classes):
     self.X_train = X_train
     self.y_train = y_train
     self.X_test = X_test
     self.y_test = y_test
     self.nb_classes = nb_classes
     self.input_size = input_size


# ===================
# split methods:
# ===================
def create_train_test_lists(dirname,train_list_name,test_list_name,train_rate):
    names = os.listdir(dirname)
    #names.sort()
    shuffle(names)
    train_ind = int(len(names) * train_rate)
    dir2list(names[:train_ind], train_list_name)
    dir2list(names[train_ind:], test_list_name)

# def split_train_test(image_dir,label_dir, train_rate, allow_randomness, INPUT_SIZE, nb_classes):
#
#     images = os.listdir(image_dir)
#     images.sort()
#     segmentations = os.listdir(label_dir)
#     segmentations.sort()
#     #
#     X = []
#     Y = []
#     for im, seg in zip(images, segmentations):
#         X.append(getImageArr(image_dir + im, INPUT_SIZE, INPUT_SIZE))
#         Y.append(getSegmentationArr(label_dir + seg, INPUT_SIZE, INPUT_SIZE, nb_classes))
#     X, Y = np.array(X), np.array(Y)
#     print(X.shape, Y.shape)
#     # Split between training and testing data:
#     # -----------------------------------------
#
#     if allow_randomness:
#         index_train = np.random.choice(X.shape[0], int(X.shape[0] * train_rate), replace=False)
#         index_test = list(set(range(X.shape[0])) - set(index_train))
#         X, Y = shuffle(X, Y)
#         X_train, y_train = X[index_train], Y[index_train]
#         X_test, y_test = X[index_test], Y[index_test]
#     else:
#         index_train = int(X.shape[0] * train_rate)  # NOTE
#         X_train, y_train = X[0:index_train], Y[0:index_train]
#         X_test, y_test = X[index_train:-1], Y[index_train:-1]  # NOTE -1
#
#     return datas(X_train, y_train, X_test, y_test, INPUT_SIZE, nb_classes)


def split_from_list(train_data, val_data, image_dir, label_dir, INPUT_SIZE, nb_classes, dataaug_args):

    print('training data:')
    [train_imgs, train_labels] = extract_arrays_from_file(train_data, image_dir, label_dir, INPUT_SIZE, nb_classes, dataaug_args)
    print('validation data:')
    [val_imgs, val_labels] = extract_arrays_from_file(val_data, image_dir, label_dir, INPUT_SIZE, nb_classes, dataaug_args)
    #train_imgs, train_labels = val_imgs, val_labels
    #
    X_train, y_train = np.array(train_imgs)[:, 0, :, :, :], np.array(train_labels)[:, 0, :, :, :]
    X_test, y_test = np.array(val_imgs)[:, 0, :, :, :], np.array(val_labels)[:, 0, :, :, :]

    return datas(X_train, y_train, X_test, y_test, INPUT_SIZE, nb_classes)

# ===================
# dataset types:
# ===================

def streets(INPUT_SIZE, dataaug_args):

    nb_classes = 11+1

    train_data = 'lst/streets_train.txt'  # to fix
    val_data = 'lst/streets_val.txt'  # to fix
    image_dir = '/storage/gby/datasets/streets/all_imgs/'
    label_dir = '/storage/gby/datasets/streets/all_labels/'
    segments_dir = ''

    #train_rate = 0.85
    #allow_randomness = False
    #ds = split_train_test(image_dir,label_dir, train_rate, allow_randomness, INPUT_SIZE, nb_classes)
    ds = split_from_list(train_data, val_data, image_dir, label_dir, INPUT_SIZE, nb_classes, dataaug_args)

    ds.segments_dir = segments_dir
    ds.train_list = train_data
    ds.test_list = val_data
    ds.datagen_train = generate_arrays_from_file(train_data, image_dir, label_dir, INPUT_SIZE, nb_classes, dataaug_args)
    ds.datagen_test = generate_arrays_from_file(val_data, image_dir, label_dir, INPUT_SIZE, nb_classes, None)

    return ds


def voc2012(INPUT_SIZE, dataaug_args):

    nb_classes = 20+1

    train_data = 'lst/voc2012_train.txt'
    val_data = 'lst/voc2012_val.txt'
    image_dir = '/storage/gby/datasets/pascal_voc12/images_orig/'
    label_dir = '/storage/gby/datasets/pascal_voc12/labels_orig/'
    segments_dir = ''

    ds = split_from_list(train_data, val_data, image_dir, label_dir, INPUT_SIZE, nb_classes, dataaug_args)

    ds.segments_dir = segments_dir
    ds.train_list = train_data
    ds.test_list = val_data
    ds.datagen_train = generate_arrays_from_file(train_data, image_dir, label_dir, INPUT_SIZE, nb_classes, dataaug_args)
    ds.datagen_test = generate_arrays_from_file(val_data, image_dir, label_dir, INPUT_SIZE, nb_classes, None)

    return ds

def horsecoarse(INPUT_SIZE,dataaug_args):

    nb_classes = 5+1
    #horse: head, tail, torso, upper legs, lower legs

    train_data = 'lst/horsecoarse_train.txt'
    val_data = 'lst/horsecoarse_test.txt'
    #image_dir = "/storage/gby/datasets/horse_coarse_parts/images_orig/"
    image_dir = '/storage/gby/datasets/pascal_voc12/images_orig/'
    label_dir = '/storage/gby/datasets/horse_coarse_parts/labels_orig/'
    segments_dir = '/storage/gby/datasets/horse_coarse_parts/sp_seg/'

    ds = split_from_list(train_data, val_data, image_dir, label_dir, INPUT_SIZE, nb_classes, dataaug_args)

    ds.segments_dir = segments_dir
    ds.train_list = train_data
    ds.test_list = val_data
    ds.datagen_train = generate_arrays_from_file(train_data, image_dir, label_dir, INPUT_SIZE,nb_classes, dataaug_args)
    ds.datagen_test = generate_arrays_from_file(val_data, image_dir, label_dir, INPUT_SIZE, nb_classes, None)

    return ds

def horsefine(INPUT_SIZE, dataaug_args):

    nb_classes = 21+1

    train_data = 'lst/horsecoarse_train.txt'
    val_data = 'lst/horsecoarse_test.txt'
    #image_dir = "/storage/gby/datasets/horse_coarse_parts/images_orig/"
    image_dir = '/storage/gby/datasets/pascal_voc12/images_orig/'
    label_dir = '/storage/gby/datasets/horse_fine_parts/labels_orig/'
    segments_dir = '/storage/gby/datasets/horse_coarse_parts/sp_seg/'

    ds = split_from_list(train_data, val_data, image_dir, label_dir, INPUT_SIZE, nb_classes, dataaug_args)

    ds.segments_dir = segments_dir
    ds.train_list = train_data
    ds.test_list = val_data
    ds.datagen_train = generate_arrays_from_file(train_data, image_dir, label_dir, INPUT_SIZE,nb_classes, dataaug_args)
    ds.datagen_test = generate_arrays_from_file(val_data, image_dir, label_dir, INPUT_SIZE, nb_classes, None)

    return ds

def personcoarse(INPUT_SIZE, dataaug_args):

    nb_classes = 6+1
    #person: head, torso, upper/lower arms and legs

    train_data = 'lst/personcoarse_train.txt'
    val_data = 'lst/personcoarse_test.txt'
    #image_dir = "/storage/gby/datasets/horse_coarse_parts/images_orig/"
    image_dir = '/storage/gby/datasets/pascal_voc12/images_orig/'
    label_dir = '/storage/gby/datasets/person_coarse_parts/labels_orig/'
    segments_dir = ''

    ds = split_from_list(train_data, val_data, image_dir, label_dir, INPUT_SIZE, nb_classes, dataaug_args)

    ds.segments_dir = segments_dir
    ds.train_list = train_data
    ds.test_list = val_data
    ds.datagen_train = generate_arrays_from_file(train_data, image_dir, label_dir, INPUT_SIZE,nb_classes, dataaug_args)
    ds.datagen_test = generate_arrays_from_file(val_data, image_dir, label_dir, INPUT_SIZE, nb_classes, None)

    return ds

def personfine(INPUT_SIZE, dataaug_args):

    nb_classes = 24+1

    train_data = 'lst/personcoarse_train.txt'
    val_data = 'lst/personcoarse_test.txt'
    #image_dir = "/storage/gby/datasets/horse_coarse_parts/images_orig/"
    image_dir = '/storage/gby/datasets/pascal_voc12/images_orig/'
    label_dir = '/storage/gby/datasets/person_fine_parts/labels_orig/'
    segments_dir = ''

    ds = split_from_list(train_data, val_data, image_dir, label_dir, INPUT_SIZE, nb_classes, dataaug_args)

    ds.segments_dir = segments_dir
    ds.train_list = train_data
    ds.test_list = val_data
    ds.datagen_train = generate_arrays_from_file(train_data, image_dir, label_dir, INPUT_SIZE,nb_classes, dataaug_args)
    ds.datagen_test = generate_arrays_from_file(val_data, image_dir, label_dir, INPUT_SIZE, nb_classes, None)

    return ds

# to debug:
def horsecoarsedbg(INPUT_SIZE,dataaug_args):

    nb_classes = 5+1
    #horse: head, tail, torso, upper legs, lower legs

    train_data = 'lst/horsecoarse_train_dbg.txt'
    val_data = 'lst/horsecoarse_test_dbg.txt'
    #image_dir = "/storage/gby/datasets/horse_coarse_parts/images_orig/"
    image_dir = '/storage/gby/datasets/pascal_voc12/images_orig/'
    label_dir = '/storage/gby/datasets/horse_coarse_parts/labels_orig/'
    segments_dir = '/storage/gby/datasets/horse_coarse_parts/sp_seg/'

    ds = split_from_list(train_data, val_data, image_dir, label_dir, INPUT_SIZE, nb_classes, dataaug_args)

    ds.segments_dir = segments_dir
    ds.train_list = train_data
    ds.test_list = val_data
    ds.datagen_train = generate_arrays_from_file(train_data, image_dir, label_dir, INPUT_SIZE,nb_classes, dataaug_args)
    ds.datagen_test = generate_arrays_from_file(val_data, image_dir, label_dir, INPUT_SIZE, nb_classes, None)

    return ds




# ===================
# load:
# ===================

def load_dataset(ds_name,INPUT_SIZE, dataaug_args):

    print('loading dataset : %s..'% ds_name)

    if ds_name == 'streets':
        ds = streets(INPUT_SIZE, dataaug_args)

    elif ds_name == 'voc2012':
        ds = voc2012(INPUT_SIZE, dataaug_args)

    elif ds_name == 'horsecoarse':
        ds = horsecoarse(INPUT_SIZE, dataaug_args)

    elif ds_name == 'horsefine':
        ds = horsefine(INPUT_SIZE, dataaug_args)

    elif ds_name == 'personcoarse':
        ds = personcoarse(INPUT_SIZE, dataaug_args)

    elif ds_name == 'personfine':
        ds = personfine(INPUT_SIZE, dataaug_args)

    elif ds_name == 'horsecoarsedbg':
        ds = horsecoarsedbg(INPUT_SIZE, dataaug_args)

    else:
        print('ERROR: dataset name does not exist..')
        return

    return ds


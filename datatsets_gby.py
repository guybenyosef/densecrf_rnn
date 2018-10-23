
import os
import numpy as np
from utils_gby import generate_arrays_from_file,extract_arrays_from_file,IoU,model_predict_gby,getImageArr,getSegmentationArr,dir2list
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

def split_train_test(image_dir,label_dir, train_rate, allow_randomness, INPUT_SIZE, nb_classes):

    images = os.listdir(image_dir)
    images.sort()
    segmentations = os.listdir(label_dir)
    segmentations.sort()
    #
    X = []
    Y = []
    for im, seg in zip(images, segmentations):
        X.append(getImageArr(image_dir + im, INPUT_SIZE, INPUT_SIZE))
        Y.append(getSegmentationArr(label_dir + seg, INPUT_SIZE, INPUT_SIZE, nb_classes))
    X, Y = np.array(X), np.array(Y)
    print(X.shape, Y.shape)
    # Split between training and testing data:
    # -----------------------------------------

    if allow_randomness:
        index_train = np.random.choice(X.shape[0], int(X.shape[0] * train_rate), replace=False)
        index_test = list(set(range(X.shape[0])) - set(index_train))
        X, Y = shuffle(X, Y)
        X_train, y_train = X[index_train], Y[index_train]
        X_test, y_test = X[index_test], Y[index_test]
    else:
        index_train = int(X.shape[0] * train_rate)  # NOTE
        X_train, y_train = X[0:index_train], Y[0:index_train]
        X_test, y_test = X[index_train:-1], Y[index_train:-1]  # NOTE -1

    return datas(X_train, y_train, X_test, y_test, INPUT_SIZE, nb_classes)


def split_from_list(train_data, val_data, image_dir, label_dir, INPUT_SIZE, nb_classes):

    print('training data:')
    [train_imgs, train_labels] = extract_arrays_from_file(train_data, image_dir, label_dir, INPUT_SIZE, nb_classes)
    print('validation data:')
    [val_imgs, val_labels] = extract_arrays_from_file(val_data, image_dir, label_dir, INPUT_SIZE, nb_classes)
    #
    X_train, y_train = np.array(train_imgs)[:, 0, :, :, :], np.array(train_labels)[:, 0, :, :, :]
    X_test, y_test = np.array(val_imgs)[:, 0, :, :, :], np.array(val_labels)[:, 0, :, :, :]

    return datas(X_train, y_train, X_test, y_test, INPUT_SIZE, nb_classes)

# ===================
# dataset types:
# ===================

def streets(INPUT_SIZE):

    nb_classes = 11+1

    image_dir = '/storage/gby/datasets/streets/images_prepped_train/'
    label_dir = '/storage/gby/datasets/streets/annotations_prepped_train/'

    train_rate = 0.85
    allow_randomness = False
    ds = split_train_test(image_dir,label_dir, train_rate, allow_randomness, INPUT_SIZE, nb_classes)

    # Median Frequency Alpha Coefficients
    coefficients = {0: 0.0237995754847,
                    1: 0.144286494916,
                    2: 0.038448897913,
                    3: 1.33901803472,
                    4: 1.0,
                    5: 0.715098627127,
                    6: 4.20827446939,
                    7: 1.58754122255,
                    8: 0.0551054437019,
                    9: 0.757994265912,
                    10: 0.218245600783,
                    11: 0.721125616748
    #                12: 6.51048559366,
    #                13: 0.125434198729,
    #                14: 3.27995580458,
    #                15: 3.72813940546,
    #                16: 3.76817843552,
    #                17: 8.90686657342,
    #                18: 2.12162414027,
    #                19: 0.
                    }

    # for python 2.7:
    # coefficients = [key for index,key in coefficients.iteritems()]
    # python 3:
    coefficients = [key for index, key in coefficients.items()]

    ds.weighted_loss_coefficients = coefficients

    return ds


def voc2012(INPUT_SIZE):

    nb_classes = 20+1

    train_data = 'lst/voc2012_train2.txt'
    val_data = 'lst/voc2012_val2.txt'
    image_dir = '/storage/gby/datasets/pascal_voc12/images_orig/'
    label_dir = '/storage/gby/datasets/pascal_voc12/labels_orig/'

    ds = split_from_list(train_data, val_data, image_dir, label_dir, INPUT_SIZE, nb_classes)

    # Median Frequency Alpha Coefficients
    # coefficients = {0: 0.0237995754847,
    #                 1: 0.144286494916,
    #                 2: 0.038448897913,
    #                 3: 1.33901803472,
    #                 4: 1.0,
    #                 5: 0.715098627127,
    #                 6: 4.20827446939,
    #                 7: 1.58754122255,
    #                 8: 0.0551054437019,
    #                 9: 0.757994265912,
    #                 10: 0.218245600783,
    #                 11: 0.721125616748,
    #                 12: 0.237995754847,
    #                 13: 0.144286494916,
    #                 14: 0.038448897913,
    #                 15: 1.33901803472,
    #                 16: 1.0,
    #                 17: 0.715098627127,
    #                 18: 4.20827446939,
    #                 19: 1.58754122255,
    #                 20: 0.0551054437019,
    #                 }

    coefficients = {
             0: 3.772978369033286,
             1: 1.89431801672104,
             2: 1.8857878994401478,
             3: 1.7128794137720464,
             4: 1.606830920261937,
             5: 1.388255758943193,
             6: 1.2817089685863616,
             7: 1.2690787118706317,
             8: 1.252809975220461,
             9: 1.2178819913716592,
             10: 1.0,
             11: 0.9904626431598804,
             12: 0.8460681114917294,
             13: 0.8186828295823247,
             14: 0.7920569266234229,
             15: 0.7181024326358014,
             16: 0.6584863672800914,
             17: 0.6538706954119894,
             18: 0.4237074630711668,
             19: 0.23850667393139946,
             20: 0.01571903401130907
    }

    # for python 2.7:
    # coefficients = [key for index,key in coefficients.iteritems()]
    # python 3:
    coefficients = [key for index, key in coefficients.items()]

    ds.weighted_loss_coefficients = coefficients

    return ds

def horsecoarse(INPUT_SIZE):

    nb_classes = 5+1

    train_data = 'lst/horsecoarse_train.txt'
    val_data = 'lst/horsecoarse_test.txt'
    #image_dir = "/storage/gby/datasets/horse_coarse_parts/images_orig/"
    image_dir = '/storage/gby/datasets/pascal_voc12/images_orig/'
    label_dir = "/storage/gby/datasets/horse_coarse_parts/labels_orig/"

    return split_from_list(train_data, val_data, image_dir, label_dir, INPUT_SIZE, nb_classes)

def horsefine(INPUT_SIZE):

    nb_classes = 21+1

    train_data = 'lst/horsecoarse_train.txt'
    val_data = 'lst/horsecoarse_test.txt'
    #image_dir = "/storage/gby/datasets/horse_coarse_parts/images_orig/"
    image_dir = '/storage/gby/datasets/pascal_voc12/images_orig/'
    label_dir = '/storage/gby/datasets/horse_fine_parts/labels_orig/'

    return split_from_list(train_data, val_data, image_dir, label_dir, INPUT_SIZE, nb_classes)

# ===================
# load:
# ===================

def load_dataset(ds_name,INPUT_SIZE):

    print('loading dataset : %s..'% ds_name)

    if ds_name == 'streets':
        ds = streets(INPUT_SIZE)

    elif ds_name == 'voc2012':
        ds = voc2012(INPUT_SIZE)

    elif ds_name == 'horsecoarse':
        ds = horsecoarse(INPUT_SIZE)

    elif ds_name == 'horsefine':
        ds = horsefine(INPUT_SIZE)

    else:
        print('ERROR: dataset name does not exist..')
        return

    return ds


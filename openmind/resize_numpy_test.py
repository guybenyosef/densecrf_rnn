import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
from os import listdir
from os.path import isfile, join

if __name__ == "__main__":
    sp_path = './data/pascal_voc12/sp_seg_img'
    onlyfiles = [f for f in listdir(sp_path) if isfile(join(sp_path, f))]
    print("only files")
    print(onlyfiles)

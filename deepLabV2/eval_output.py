import numpy as np
from sklearn.metrics import jaccard_similarity_score
import os
from os import listdir
from os.path import isfile, join
import cv2

def load_image(img_org,INPUT_SIZE):
    h, w, c = img_org.shape
    img = cv2.resize(img_org, (INPUT_SIZE, INPUT_SIZE))
    img = np.array(img, dtype=np.float32)
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    return x

def load_label(img_org,INPUT_SIZE,nb_classes):
    img = cv2.resize(img_org,(INPUT_SIZE, INPUT_SIZE))
    img = np.array(img, dtype=np.uint8)
    img[img==255] = 0
    y = np.zeros((1, img.shape[0], img.shape[1], nb_classes), dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            y[0, i, j, img[i][j]] = 1
    return y

def get_IOU(prediction_dir, gt_dir):
    input_size = 321
    nb_classes = 21
    all_imgs= [f for f in listdir(prediction_dir) if isfile(join(prediction_dir, f))]
    predictions, gt = [], []
    for im_name in all_imgs:
        pred_img_org = cv2.imread(os.path.join(prediction_dir, im_name))
        #print(pred_img_org)
        gt_img_org = cv2.imread(gt_dir + im_name[:-9] + ".png")
        #gt_img_org = cv2.imread(os.path.join(gt_dir, im_name))
        #print(gt_dir + im_name[:-9] + ".png")
        predictions.append(load_label(pred_img_org, input_size, nb_classes))
        gt.append(load_label(gt_img_org, input_size, nb_classes))

    iou = jaccard_similarity_score(np.array(gt).flatten(), np.array(predictions).flatten())
    print(iou)

if __name__ == "__main__":
    get_IOU("output/prediction/", "/om2/user/cfmata/voc12/train/VOCdevkit/VOC2012/SegmentationClass_1D")
    #get_IOU("output/deeplab_resnet_ckpt_prediction_test/", "output/deeplab_resnet_ckpt_prediction_test/") #debug

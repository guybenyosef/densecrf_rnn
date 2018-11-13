from skimage import data, io, segmentation, color
from skimage.segmentation import find_boundaries
from skimage.future import graph
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from scipy import misc
from os import listdir
from os.path import isfile, join
import json
import copy

def test_rag_code():
    img_org = Image.open("2008_000008.jpg")
    img = np.array(img_org, dtype=np.float32)

    labels1 = segmentation.slic(img_org, compactness=40, n_segments=600)
    out1 = color.label2rgb(labels1, img, kind='avg')

    g = graph.rag_mean_color(img, labels1)
    labels2 = graph.cut_threshold(labels1, g, 10)
    out2 = color.label2rgb(labels2, img, kind='avg')
    print(out2)
    misc.imsave('outfile.jpg', out2)
    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True,
                           figsize=(6, 8))

    ax[0].imshow(out1)
    ax[1].imshow(out2)

    for a in ax:
        a.axis('off')

    plt.tight_layout()
    #plt.savefig('test.png')

def generate_sp_and_bd_seg(dataset, INPUT_SIZE):
    # Set image paths
    image_path = 'images_orig/'
    save_path_sp = 'sp_seg_test/'
    save_path_bd = 'bd_seg_test/'
    img_names = []
    # Load image names from txt files
    if dataset == 'horsecoarse':
        with open('../../lst/'+dataset+'_train.txt', 'r') as f:
            img_names.extend([line.strip() for line in f])
        with open('../../lst/'+dataset+'_test.txt', 'r') as f:
            img_names.extend([line.strip() for line in f])
    # Generate and save superpixels and boundaries
    for name in img_names:
        '''
        # Superpixels
        img_org = Image.open(image_path+name+'.jpg')
        img = np.array(img_org.resize((INPUT_SIZE, INPUT_SIZE)))
        labels = segmentation.slic(img, compactness = 40, n_segments = 600)
        np.save(save_path_sp+name+"_sp.npy", labels)
        '''
        # Boundaries
        data = np.load("sp_seg/"+name+"_sp.npy")
        num_segments = max(np.unique(data))
        # Take care of 0th superpixel
        bd_dict = {}
        labeled = data.copy()
        labeled[labeled != 0] = 1
        labeled = 1 - labeled
        ind = zip(*np.where(find_boundaries(labeled,mode="inner").astype(np.uint8) == 1))
        bd_dict[str(0)] = [str(t) for t in ind]
        for i in range(1,num_segments+1):
            labeled = data.copy()
            labeled[labeled != i] = 0
            ind = zip(*np.where(find_boundaries(labeled, mode="inner").astype(np.uint8) == 1))
            bd_dict[str(i)] = [str(t) for t in ind]
        with open(save_path_bd+name+"_bd.json", "w") as f:
            json.dump(bd_dict, f)
        
if __name__ == "__main__":
    generate_sp_and_bd_seg('horsecoarse', 512)
    '''
    im_names = [f for f in listdir("./images_orig/") if isfile(join("./images_orig/", f))]
    for name in im_names:
        img_org = Image.open("./images_orig/"+name)
        img = np.array(img_org, dtype=np.float32)
        labels1 = segmentation.slic(img_org, compactness=30, n_segments=200)
        out1 = color.label2rgb(labels1, img, kind='avg')

        g = graph.rag_mean_color(img, labels1)
        labels2 = graph.cut_threshold(labels1, g, 10)
        out2 = color.label2rgb(labels2, img, kind='avg')
        misc.imsave('./RAG_other/'+name, out2)
    '''
# ./RAG: compactness 40, n_segments 600, cut_threshold 10
# ./RAG_other: compactness 30, n_segments 200, cut_threshold 10

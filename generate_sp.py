from skimage import data, io, segmentation, color
from skimage.future import graph
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from scipy import misc
from os import listdir
from os.path import isfile, join

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

def generate_sp_seg(dataset, INPUT_SIZE):
    image_path = 'images_orig/'
    save_path = 'sp_seg/'
    img_names = []
    if dataset == 'horsecoarse':
        with open('../../lst/'+dataset+'_train.txt', 'r') as f:
            img_names.extend([line.strip() for line in f])
        with open('../../lst/'+dataset+'_test.txt', 'r') as f:
            img_names.extend([line.strip() for line in f])
    for name in img_names:
        img_org = Image.open(image_path+name+'.jpg')
        img = np.array(img_org.resize((INPUT_SIZE, INPUT_SIZE)))
        labels = segmentation.slic(img, compactness = 40, n_segments = 600)
        np.save(save_path+name+"_sp.npy", labels)
        
if __name__ == "__main__":
    generate_sp_seg('horsecoarse', 512)
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

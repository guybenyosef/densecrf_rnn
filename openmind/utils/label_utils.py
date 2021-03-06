from PIL import Image
import numpy as np
import tensorflow as tf

# colour map
label_colours = [(0,0,0)
                # 0=background
                ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
                # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
                ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128), (0, 192, 128), (128, 64, 128), (128, 192, 128), (0, 64, 192)]
                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor

def decode_labels(mask, num_images=1, num_classes=21):
    """Decode batch of segmentation masks.
    
    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).
    
    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    n, h, w, c = mask.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
      img = Image.new('RGB', (len(mask[i, 0]), len(mask[i]))) # Size is given as a (width, height)-tuple.
      pixels = img.load()
      for j_, j in enumerate(mask[i, :, :, 0]):
          for k_, k in enumerate(j):
              if k < num_classes:
                  pixels[k_,j_] = label_colours[k]
      outputs[i] = np.array(img)
    return outputs

def prepare_label(input_batch, new_size, num_classes, one_hot=True):
    """Resize masks and perform one-hot encoding.

    Args:
      input_batch: input tensor of shape [batch_size H W 1].
      new_size: a tensor with new height and width.
      num_classes: number of classes to predict (including background).
      one_hot: whether perform one-hot encoding.

    Returns:
      Outputs a tensor of shape [batch_size h w 21]
      with last dimension comprised of 0's and 1's only.
    """
    with tf.name_scope('label_encode'):
        input_batch = tf.image.resize_nearest_neighbor(input_batch, new_size) # as labels are integer numbers, need to use NN interp.
        input_batch = tf.squeeze(input_batch, squeeze_dims=[3]) # reducing the channel dimension.
        if one_hot:
            input_batch = tf.one_hot(input_batch, depth=num_classes)
    return input_batch

def inv_preprocess(imgs, num_images, img_mean):
    """Inverse preprocessing of the batch of images.
       Add the mean vector and convert from BGR to RGB.
       
    Args:
      imgs: batch of input images.
      num_images: number of images to apply the inverse transformations on.
      img_mean: vector of mean colour values.
  
    Returns:
      The batch of the size num_images with the same spatial dimensions as the input.
    """
    n, h, w, c = imgs.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, c), dtype=np.uint8)
    for i in range(num_images):
        outputs[i] = (imgs[i] + img_mean)[:, :, ::-1].astype(np.uint8)
    return outputs

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

def get_class_coefficients(dataset):
    if dataset == 'pascal_voc12':
        return [0.01460247, 1.25147725, 2.88479363, 1.20348121, 1.65261654, 1.67514772,
                0.62338799, 0.7729363,  0.42038501, 0.98557268, 1.31867536, 0.85313332,
                0.67227604, 1.21317965, 1.        , 0.24263748, 1.80877607, 1.3082213,
                0.79664027, 0.72543945, 1.27823374]
    elif dataset == 'horse_fine':
        return [3.78973381e-03, 9.64408576e-02, 1.88294674e+00, 9.92517805e-01,
                7.09033026e-01, 1.02420400e+00, 3.17414917e+00, 2.45586737e+00,
                1.00759586e+00, 5.36619703e-01, 1.27326380e-01, 1.04601344e-01,
                3.31558138e+00, 1.12646887e+00, 7.59358415e-01, 1.08926514e+00,
                4.42533957e+00, 3.23007101e+00, 1.14992130e+00, 5.92241178e-01,
                3.99196119e-01, 3.33183389e-02]
    elif dataset == 'person_fine':
        return [0.00662793, 0.16033985, 2.92858541, 3.42624528, 3.03382611, 3.21656659,
                4.4105997 , 4.04033713, 1.88111505, 2.53033728, 0.19551068, 0.06375701,
                0.86649008, 0.64958644, 0.34305739, 1.11480713, 0.6617285,  0.34152296,
                1.11532255, 0.96739083, 0.436357  , 2.97533242, 1.       ,  0.4131923,
                3.326273  ]


from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, Add, Cropping2D, MaxPooling2D, Activation, Dropout, ZeroPadding2D, UpSampling2D, concatenate
from keras.layers.normalization import BatchNormalization
from keras.initializers import Constant
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from utils_gby_new import bilinear_upsample_weights
import sys
sys.path.insert(1, './src')
from crfrnn_layer import CrfRnnLayer #, CrfRnnLayerSPIOAT
from crfrnn_layer_all import CrfRnnLayerAll, CrfRnnLayerSP, CrfRnnLayerSPIO, CrfRnnLayerSPAT

# saved_model_path = '/storage/gby/semseg/streets_weights_resnet50fcn8s_2000ep'
# saved_model_path = '/storage/gby/semseg/voc2012_weights_fcn_RESNET50_8s_500ep'
# #
# saved_model_path = '/storage/gby/semseg/personfine_weights_fcn_RESNET50_8s_500ep'
# saved_model_path = '/storage/cfmata/deeplab/crf_rnn/crfasrnn_keras/results/person_fine/person_fine_weights.200-0.54'
# #
# saved_model_path = '/storage/gby/semseg/horsecoarse_weights_fcn_RESNET50_8s_100ep'
# saved_model_path = '/storage/cfmata/deeplab/crf_rnn/crfasrnn_keras/results/horse_coarse/horse_coarse_weights.5000-0.39'
# #
# saved_model_path = '/storage/gby/semseg/horsefine_weights_fcn_RESNET50_8s_500ep'
# #
# #saved_model_path = ''


# -----------------------
# Model design
# -----------------------

def fcn_32s_orig(nb_classes):
    inputs = Input(shape=(None, None, 3))
    vgg16 = VGG16(weights='imagenet', include_top=False, input_tensor=inputs)
    x = Conv2D(filters=nb_classes,
               kernel_size=(1, 1))(vgg16.output)
    x = Conv2DTranspose(filters=nb_classes,
                        kernel_size=(64, 64),
                        strides=(32, 32),
                        padding='same',
                        activation='sigmoid',
                        kernel_initializer=Constant(bilinear_upsample_weights(32, nb_classes)))(x)
    model = Model(inputs=inputs, outputs=x)
    for layer in model.layers[:15]:
        layer.trainable = False
    return model


def fcn_VGG16_32s(INPUT_SIZE,nb_classes):
    """ Returns Keras FCN-32 model definition.

      """
    # Input and output layers for FCN:
    inputs = Input(shape=(INPUT_SIZE, INPUT_SIZE, 3))
    # Start from VGG16 layers
    vgg16 = VGG16(weights='imagenet', include_top=False, input_tensor=inputs)
    # score from the top vgg16 layer:
    score7 = vgg16.output
    #score7 = Dropout(0.5)(score7)  # (optional)

    score7c = Conv2D(filters=nb_classes,kernel_size=(1, 1), name='score7c')(score7)
    #
    score7c_upsample_32 = Conv2DTranspose(filters=nb_classes,
                                          kernel_size=(64, 64),
                                          strides=(32, 32),
                                          padding='same',
                                          activation=None,
                                          kernel_initializer=Constant(bilinear_upsample_weights(32, nb_classes)),
                                          name="score_pool7c_upsample_32")(score7c)

    fcn_output = (Activation('softmax'))(score7c_upsample_32)
    #fcn_output = score7c_upsample_32

    model = Model(inputs=inputs, output=fcn_output, name='fcn_VGG16_32s')

    # Fixing weighs in lower layers
    # for layer in model.layers[:15]:  # sometimes I use it, sometimes not.
    #     layer.trainable = False

    return model

def fcn_VGG16_32s_crfrnn(INPUT_SIZE,nb_classes,num_crf_iterations):
    """ Returns Keras FCN-32 + CRFRNN layer model definition.

      """
    fcn = fcn_VGG16_32s(INPUT_SIZE,nb_classes)
    saved_model_path = '/storage/gby/semseg/voc12_weights_fcn32_200ep'
    fcn.load_weights(saved_model_path)

    inputs = fcn.layers[0].output

    fcn_score = fcn.get_layer('score_pool7c_upsample_32').output
    # used to be: fcn.output
    #fcn_score = fcn.output

    # Adding the crfrnn layer:
    height, weight = INPUT_SIZE, INPUT_SIZE
    crfrnn_output = CrfRnnLayer(image_dims=(height, weight),
                             num_classes=nb_classes,
                             theta_alpha=160.,
                             theta_beta=90.,
                             theta_gamma=3.,
                             num_iterations=num_crf_iterations,   # 10 at test time, 5 at train time
                             name='crfrnn')([fcn_score, inputs])

    model = Model(inputs=inputs, output=crfrnn_output, name='fcn_VGG16_32s_crfrnn')

    return model


def fcn_VGG16_8s(INPUT_SIZE,nb_classes):    # previous name: fcn8s_take2
    """ Returns Keras FCN-8 model definition.

      """
    fcn32_flag = False

    inputs = Input(shape=(INPUT_SIZE, INPUT_SIZE, 3))
    # Start from VGG16 layers
    vgg16 = VGG16(weights='imagenet', include_top=False, input_tensor=inputs)

    # Skip connections from pool3, 256 channels
    vgg16_upto_intermediate_layer_pool3 = Model(inputs=vgg16.input, outputs=vgg16.get_layer('block3_pool').output)
    score_pool3 = vgg16_upto_intermediate_layer_pool3.output
    # 1x1 conv layer to reduce number of channels to nb_classes:
    score_pool3c = Conv2D(filters=nb_classes,kernel_size=(1, 1),name="score_pool3c")(score_pool3)

    # Skip connections from pool4, 512 channels
    vgg16_upto_intermediate_layer_pool4 = Model(inputs=vgg16.input, outputs=vgg16.get_layer('block4_pool').output)
    score_pool4 = vgg16_upto_intermediate_layer_pool4.output
    # 1x1 conv layer to reduce number of channels to nb_classes:
    score_pool4c = Conv2D(filters=nb_classes, kernel_size=(1, 1))(score_pool4)

    # score from the top vgg16 layer:
    score_pool5 = vgg16.output
    #n = 4096
    score6c = Conv2D(filters=4096, kernel_size=(7, 7), padding='same', name="conv6")(score_pool5)
    score7c = Conv2D(filters=4096, kernel_size=(1, 1), padding='same', name="conv7")(score6c)

    #score7c = Conv2D(filters=nb_classes,kernel_size=(1, 1))(score6c)
    score7c_upsample = Conv2DTranspose(filters=nb_classes,
                                       kernel_size=(4, 4),
                                       strides=(2, 2),
                                       padding='same',
                                       activation = None,
                                       kernel_initializer = Constant(bilinear_upsample_weights(2, nb_classes)),
                                       name="score_pool7c_upsample")(score7c)

    # Fuse scores
    score_7_4 = Add()([score7c_upsample, score_pool4c])
    # upsample:
    score_7_4_up = Conv2DTranspose(filters=nb_classes,
                                      kernel_size=(4, 4),
                                      strides=(2, 2),
                                      padding='same',
                                      activation= None,
                                      kernel_initializer=Constant(bilinear_upsample_weights(2, nb_classes)),
                                      name="score_7_4_up")(score_7_4)

    # Fuse scores
    score_7_4_3 = Add()([score_7_4_up, score_pool3c])
    # upsample:
    score_7_4_3_up = Conv2DTranspose(filters=nb_classes,
                                   kernel_size=(16, 16),
                                   strides=(8, 8),
                                   padding='same',
                                   activation=None,
                                   kernel_initializer=Constant(bilinear_upsample_weights(8, nb_classes)),
                                   name="score_7_4_3_up")(score_7_4_3)

    # Batch Normalization: (optional)
    #score_7_4_3_up = BatchNormalization()(score_7_4_3_up)

    output = (Activation('softmax'))(score_7_4_3_up)

    # # -- There's another way to match the tensor sizes from earlier layers, using a Cropping2D layer --
    # # e.g., for fcn-16, we can crop layer 'score_pool4c' to get the same size as layer 'score_7c'
    # score_pool4c_cropped = Cropping2D((5+3, 5+3))(score_pool4c)
    # # fuse layers,
    # score_7_4_cropped = Add()([score7c, score_pool4c_cropped])
    # # then upsample to input size:
    # x = Conv2DTranspose(filters=nb_classes,
    #                     kernel_size=(64, 64),
    #                     strides=(32+2,32+2),
    #                     padding='same',
    #                     activation='sigmoid',
    #                     kernel_initializer=Constant(bilinear_upsample_weights(32, nb_classes)))(score_7_4_cropped)

    # Creating the model:
    model = Model(inputs=inputs, outputs=output, name='fcn_VGG16_8s')

    # Fixing weighs in lower layers
    # for layer in model.layers[:15]:  # sometimes I use it, sometimes not.
    #     layer.trainable = False

    return model


def fcn_VGG16_8s_crfrnn(INPUT_SIZE,nb_classes,num_crf_iterations, batch_size):
    """ Returns Keras FCN-8 + CRFRNN layer model definition.

    """
    fcn = fcn_VGG16_8s(INPUT_SIZE,nb_classes)
    saved_model_path = '/storage/cfmata/deeplab/crf_rnn/crfasrnn_keras/results/results/pascal_voc12/voc2012_sadeep_start0.80-1.18'
    fcn.load_weights(saved_model_path)

    inputs = fcn.layers[0].output
    # Add plenty of zero padding
    #inputs = ZeroPadding2D(padding=(100, 100))(inputs)

    fcn_score = fcn.get_layer('score_7_4_3_up').output
    # used to be: fcn.output
    #fcn_score = fcn.output

    # Adding the crfrnn layer:
    height, weight = INPUT_SIZE, INPUT_SIZE
    crfrnn_output = CrfRnnLayer(image_dims=(height, weight),
                             num_classes=nb_classes,
                             theta_alpha=160.,
                             theta_beta=90.,   #3.
                             theta_gamma=3.,
                             num_iterations=num_crf_iterations, # 10 in test time, 5 in train time
                             batch_size = batch_size,
                             name='crfrnn')([fcn_score, inputs])

    model = Model(inputs=inputs, output=crfrnn_output, name='fcn_VGG16_8s_crfrnn')

    # # Fixing weighs in lower layers (optional)
    # for layer in model.layers[:28]:  # 15,21,29 (overall 30 layers)
    #      layer.trainable = True

    return model


def fcn_8s_Sadeep(nb_classes):
    """ Returns Keras CRF-RNN model definition.

    Currently, only 500 x 500 images are supported. However, one can get this to
    work with different image sizes by adjusting the parameters of the Cropping2D layers
    below.
    """
    channels, height, weight = 3, 500, 500

    # Input
    input_shape = (height, weight, 3)
    img_input = Input(shape=input_shape)

    # Add plenty of zero padding
    x = ZeroPadding2D(padding=(100, 100))(img_input)

    # VGG-16 convolution block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='valid', name='conv1_1')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

    # VGG-16 convolution block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2', padding='same')(x)

    # VGG-16 convolution block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3', padding='same')(x)
    pool3 = x

    # VGG-16 convolution block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4', padding='same')(x)
    pool4 = x

    # VGG-16 convolution block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool5', padding='same')(x)

    # Fully-connected layers converted to convolution layers
    x = Conv2D(4096, (7, 7), activation='relu', padding='valid', name='fc6')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(4096, (1, 1), activation='relu', padding='valid', name='fc7')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(nb_classes, (1, 1), padding='valid', name='score-fr')(x)

    # Deconvolution
    score2 = Conv2DTranspose(nb_classes, (4, 4), strides=2, name='score2')(x)

    # Skip connections from pool4
    score_pool4 = Conv2D(nb_classes, (1, 1), name='score-pool4')(pool4)
    score_pool4c = Cropping2D((5, 5))(score_pool4)
    score_fused = Add()([score2, score_pool4c])
    score4 = Conv2DTranspose(nb_classes, (4, 4), strides=2, name='score4', use_bias=False)(score_fused)

    # Skip connections from pool3
    score_pool3 = Conv2D(nb_classes, (1, 1), name='score-pool3')(pool3)
    score_pool3c = Cropping2D((9, 9))(score_pool3)

    # Fuse things together
    score_final = Add()([score4, score_pool3c])

    # Final up-sampling and cropping
    upsample = Conv2DTranspose(nb_classes, (16, 16), strides=8, name='upsample', use_bias=False)(score_final)
    upscore = Cropping2D(((31, 37), (31, 37)),name='upscore')(upsample)

    # Batch Normalization: (optional)
    # upscore = BatchNormalization()(upscore)

    output = (Activation('softmax'))(upscore)

    # Build the model
    model = Model(img_input, output, name='fcn_8s_Sadeep')

    return model


def fcn_8s_Sadeep_crfrnn(nb_classes,num_crf_iterations, batch_size):#, batch_sizes_train, batch_sizes_val, batch_sizes_total):
    """ Returns Keras FCN-8 + CRFRNN layer model definition.

      """
    INPUT_SIZE = 500

    fcn = fcn_8s_Sadeep(nb_classes)
    #saved_model_path = '/storage/gby/semseg/streets_weights_fcn8s_Sadeep_500ep'
    #saved_model_path = '/storage/cfmata/deeplab/crf_rnn/crfasrnn_keras/results/pascal_voc12/voc2012_sadeep_start0.80-1.18'
    saved_model_path = '/storage/cfmata/deeplab/crf_rnn/crfasrnn_keras/results/horse_coarse/horsecoarse_fcn_8s_Sadeep_is500_ep30_iou.215'
    fcn.load_weights(saved_model_path)

    inputs = fcn.layers[0].output
    #seg_input = fcn.layers[0].output
    # Add plenty of zero padding
    #inputs = ZeroPadding2D(padding=(100, 100))(inputs)

    fcn_score = fcn.get_layer('upscore').output
    # used to be: fcn.output
    #fcn_score = fcn.output

    # Adding the crfrnn layer:
    height, weight = INPUT_SIZE, INPUT_SIZE
    crfrnn_output = CrfRnnLayer(image_dims=(height, weight),
                                num_classes=nb_classes,
                                theta_alpha=160.,
                                theta_beta=90.,  #3.
                                theta_gamma=3.,
                                batch_size = batch_size,
                                #batch_sizes_train = batch_sizes_train,
                                #batch_sizes_val = batch_sizes_val,
                                #batch_sizes_total = batch_sizes_total,
                                num_iterations=num_crf_iterations, # 10 in test time, 5 in train time
                                name='crfrnn')([fcn_score, inputs])

    # crfrnn_output = CrfRnnLayerSP(image_dims=(height, weight),
    #                      num_classes=nb_classes,
    #                      theta_alpha=160.,
    #                      theta_beta=3.,
    #                      theta_gamma=3.,
    #                      num_iterations=0, #5
    #                      bil_rate = 0.5, #add for the segmentation
    #                      theta_alpha_seg = 30, #add for the segmentation
    #                      name='crfrnn')([fcn_score, inputs, seg_input]) #set num_iterations to 0 if we do not want crf


    model = Model(inputs=inputs, output=crfrnn_output, name='fcn_8s_Sadeep_crfrnn')

    # # Fixing weighs in lower layers (optional)
    # for layer in model.layers[:29]:  # 15,21,29 (overall 30 layers)
    #      layer.trainable = True #False

    return model

def fcn_RESNET50_32s(INPUT_SIZE,nb_classes):
    """ Returns Keras FCN-32 + based on ResNet50 model definition.

      """

    # Input and output layers for FCN:
    inputs = Input(shape=(INPUT_SIZE, INPUT_SIZE, 3))
    # Start from ResNet50 layers
    resnet50 = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)

    # score from the top resnet50 layer:
    act49 = resnet50.output  # equivalent to: resnet50.get_layer('activation_49').output
    act49 = Dropout(0.5)(act49) # (optional)
    # add classifier:
    pred32 = Conv2D(filters=nb_classes,kernel_size=(1, 1), name='pred_32')(act49)
    # add upsampler:
    score_pred32_upsample = Conv2DTranspose(filters=nb_classes,
                                          kernel_size=(64, 64),
                                          strides=(32, 32),
                                          padding='same',
                                          activation=None,
                                          kernel_initializer=Constant(bilinear_upsample_weights(32, nb_classes)),
                                          name="score_pred32_upsample")(pred32)

    output = (Activation('softmax'))(score_pred32_upsample)

    model = Model(inputs=inputs, outputs=output, name='fcn_RESNET50_32s')

    # fine-tune
    train_layers = ['pred_32',
                    'score_pred32_upsample'

                    'bn5c_branch2c',
                    'res5c_branch2c',
                    'bn5c_branch2b',
                    'res5c_branch2b',
                    'bn5c_branch2a',
                    'res5c_branch2a',

                    'bn5b_branch2c',
                    'res5b_branch2c',
                    'bn5b_branch2b',
                    'res5b_branch2b',
                    'bn5b_branch2a',
                    'res5b_branch2a',

                    'bn5a_branch2c',
                    'res5a_branch2c',
                    'bn5a_branch2b',
                    'res5a_branch2b',
                    'bn5a_branch2a',
                    'res5a_branch2a']

    # for l in model.layers:
    #     if l.name in train_layers:
    #         l.trainable = True
    #     else:
    #         l.trainable = False

    return model


def fcn_RESNET50_32s_crfrnn(INPUT_SIZE,nb_classes,num_crf_iterations):
    """ Returns Keras FCN-8 + based on ResNet50 model definition.

    """
    fcn = fcn_RESNET50_32s(INPUT_SIZE, nb_classes)
    saved_model_path = '/storage/gby/semseg/streets_weights_resnet50fcn32s_5000ep'
    fcn.load_weights(saved_model_path)

    inputs = fcn.layers[0].output

    #fcn_score = fcn.output
    fcn_score = fcn.get_layer('score_pred32_upsample').output

    # Adding the crfrnn layer:
    height, weight = INPUT_SIZE, INPUT_SIZE
    crfrnn_output = CrfRnnLayer(image_dims=(height, weight),
                                num_classes=nb_classes,
                                theta_alpha=160.,
                                theta_beta=90.,
                                theta_gamma=3.,
                                num_iterations=num_crf_iterations,  # 10 for test, 5 for train
                                name='crfrnn')([fcn_score, inputs])

    model = Model(inputs=inputs, outputs=crfrnn_output, name='fcn_RESNET50_32s_crfrnn')

    # Fixing weighs in lower layers (optional)
    # for layer in model.layers[:-1]:  # 15,21,29 (overall 30 layers)
    #     layer.trainable = True
    return model



def fcn_RESNET50_8s(INPUT_SIZE,nb_classes):
    """ Returns Keras FCN-8 + based on ResNet50 model definition.

      """

    # Input and output layers for FCN:
    inputs = Input(shape=(INPUT_SIZE, INPUT_SIZE, 3))
    # Start from ResNet50 layers
    resnet50 = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)

    act22 = resnet50.get_layer('activation_22').output
    act22 = Dropout(0.5)(act22)
    # add classifier:
    pred8 = Conv2D(filters=nb_classes, kernel_size=(1, 1), name='pred_8')(act22)
    # add upsampler:
    score_pred8_upsample = Conv2DTranspose(filters=nb_classes,
                                            kernel_size=(64, 64),
                                            strides=(8, 8),
                                            padding='same',
                                            activation=None,
                                            kernel_initializer=Constant(bilinear_upsample_weights(8, nb_classes)),
                                            name="score_pred8_upsample")(pred8)

    act40 = resnet50.get_layer('activation_40').output
    act40 = Dropout(0.5)(act40)
    # add classifier:
    pred16 = Conv2D(filters=nb_classes, kernel_size=(1, 1), name='pred_16')(act40)
    # add upsampler:
    score_pred16_upsample = Conv2DTranspose(filters=nb_classes,
                                            kernel_size=(64, 64),
                                            strides=(16, 16),
                                            padding='same',
                                            activation=None,
                                            kernel_initializer=Constant(bilinear_upsample_weights(16, nb_classes)),
                                            name="score_pred16_upsample")(pred16)

    # score from the top resnet50 layer:
    act49 = resnet50.output  # equivalent to: resnet50.get_layer('activation_49').output
    act49 = Dropout(0.5)(act49)
    # add classifier:
    pred32 = Conv2D(filters=nb_classes,kernel_size=(1, 1), name='pred_32')(act49)
    # add upsampler:
    score_pred32_upsample = Conv2DTranspose(filters=nb_classes,
                                          kernel_size=(64, 64),
                                          strides=(32, 32),
                                          padding='same',
                                          activation=None,
                                          kernel_initializer=Constant(bilinear_upsample_weights(32, nb_classes)),
                                          name="score_pred32_upsample")(pred32)


    # Fuse scores
    score_pred16_pred32 = Add()([score_pred32_upsample, score_pred16_upsample])
    score_pred8_pred16_pred32 = Add(name='add_pred8_pred16_pred32')([score_pred16_pred32, score_pred8_upsample])

    #score_pred8_pred16_pred32 = BatchNormalization()(score_pred8_pred16_pred32)

    output = (Activation('softmax'))(score_pred8_pred16_pred32)

    model = Model(inputs=inputs, outputs=output, name='fcn_RESNET50_8s')

    # fine-tune
    train_layers = ['pred_32',
                    'score_pred32_upsample'

                    'bn5c_branch2c',
                    'res5c_branch2c',
                    'bn5c_branch2b',
                    'res5c_branch2b',
                    'bn5c_branch2a',
                    'res5c_branch2a',

                    'bn5b_branch2c',
                    'res5b_branch2c',
                    'bn5b_branch2b',
                    'res5b_branch2b',
                    'bn5b_branch2a',
                    'res5b_branch2a',

                    'bn5a_branch2c',
                    'res5a_branch2c',
                    'bn5a_branch2b',
                    'res5a_branch2b',
                    'bn5a_branch2a',
                    'res5a_branch2a']

    # for l in model.layers:
    #     if l.name in train_layers:
    #         l.trainable = True
    #     else:
    #         l.trainable = False

    return model


def fcn_RESNET50_8s_crfrnn(INPUT_SIZE,nb_classes,num_crf_iterations,finetune_path, batch_size):#, batch_sizes_train, batch_sizes_val, batch_sizes_total):
    """ Returns Keras FCN-8 + CRFRNNlayer, based on ResNet50 model definition.

    """
    fcn = fcn_RESNET50_8s(INPUT_SIZE, nb_classes)

    if not finetune_path=='':
        fcn.load_weights(finetune_path)

    inputs = fcn.layers[0].output

    #fcn_score = fcn.output
    fcn_score = fcn.get_layer('add_pred8_pred16_pred32').output

    # Adding the crfrnn layer:
    height, weight = INPUT_SIZE, INPUT_SIZE
    crfrnn_output = CrfRnnLayer(image_dims=(height, weight),
                                num_classes=nb_classes,
                                theta_alpha=160.,
                                theta_beta=90.,
                                theta_gamma=3.,
                                batch_size=batch_size,
                                num_iterations=num_crf_iterations,  # 10 for test, 5 for train
                                name='crfrnn')([fcn_score, inputs])

    model = Model(inputs=inputs, outputs=crfrnn_output, name='fcn_RESNET50_8s_crfrnn')

    # Fixing weighs in lower layers (optional)
    for layer in model.layers[:160]:
        layer.trainable = False
    
    return model


def fcn_RESNET50_8s_crfrnnSP(INPUT_SIZE,nb_classes,num_crf_iterations,finetune_path, batch_size):
    """ Returns Keras FCN-8 + CRFRNNlayer with SP, based on ResNet50 model definition.

    """
    fcn = fcn_RESNET50_8s(INPUT_SIZE, nb_classes)

    if not finetune_path=='':
        fcn.load_weights(finetune_path)

    # two inputs:
    img_input = fcn.layers[0].output
    #seg_input = fcn.layers[0].output
    seg_input = Input(shape=(INPUT_SIZE, INPUT_SIZE))

    #fcn_score = fcn.output
    fcn_score = fcn.get_layer('add_pred8_pred16_pred32').output

    # Adding the crfrnn layer:
    height, weight = INPUT_SIZE, INPUT_SIZE
    crfrnn_output = CrfRnnLayerSP(image_dims=(height, weight),
                                num_classes=nb_classes,
                                theta_alpha=160.,
                                theta_beta=90.,
                                theta_gamma=3.,
                                batch_size=batch_size,
                                num_iterations=num_crf_iterations,  # 10 for test, 5 for train
                                name='crfrnn')([fcn_score, img_input, seg_input])

    model = Model(inputs=[img_input, seg_input], outputs=crfrnn_output, name='fcn_RESNET50_8s_crfrnnSP')


    # Fixing weighs in lower layers (optional)
    for layer in model.layers[:160]: #[:181]:
        layer.trainable = False

    return model


def fcn_RESNET50_8s_crfrnnSPIO(INPUT_SIZE,nb_classes,num_crf_iterations,finetune_path, batch_size):
    """ Returns Keras FCN-8 + CRFRNNlayer with SP term and Inside/outside term, based on ResNet50 model definition.

    """
    fcn = fcn_RESNET50_8s(INPUT_SIZE, nb_classes)

    if not finetune_path=='':
        fcn.load_weights(finetune_path)

    # two inputs:
    img_input = fcn.layers[0].output
    seg_input = Input(shape=(INPUT_SIZE, INPUT_SIZE))

    #fcn_score = fcn.output
    fcn_score = fcn.get_layer('add_pred8_pred16_pred32').output

    # Adding the crfrnn layer:
    height, weight = INPUT_SIZE, INPUT_SIZE
    crfrnn_output = CrfRnnLayerSPIO(image_dims=(height, weight),
                                    num_classes=nb_classes,
                                    theta_alpha=160.,
                                    theta_beta=90.,
                                    theta_gamma=3.,
                                    batch_size=batch_size,
                                    num_iterations=num_crf_iterations,  # 10 for test, 5 for train
                                    name='crfrnn')([fcn_score, img_input, seg_input])

    model = Model(inputs=[img_input, seg_input], outputs=crfrnn_output, name='fcn_RESNET50_8s_crfrnnSPIO')

    # Fixing weighs in lower layers (optional)
    #for layer in model.layers[:-1]: #[:181]:  # 15,21,29 (overall 30 layers) feezing until layer pred 8 (182)
    #    layer.trainable = False

    return model


def fcn_RESNET50_8s_crfrnnSPAT(INPUT_SIZE,nb_classes,num_crf_iterations,finetune_path, batch_size):
    """ Returns Keras FCN-8 + CRFRNNlayer with SP term and attachment term, based on ResNet50 model definition.

    """
    fcn = fcn_RESNET50_8s(INPUT_SIZE, nb_classes)

    if not finetune_path=='':
        fcn.load_weights(finetune_path)

    # two inputs:
    img_input = fcn.layers[0].output
    seg_input = Input(shape=(INPUT_SIZE, INPUT_SIZE))

    #fcn_score = fcn.output
    fcn_score = fcn.get_layer('add_pred8_pred16_pred32').output

    # Adding the crfrnn layer:
    height, weight = INPUT_SIZE, INPUT_SIZE
    crfrnn_output = CrfRnnLayerSPAT(image_dims=(height, weight),
                                num_classes=nb_classes,
                                theta_alpha=160.,
                                theta_beta=90.,
                                theta_gamma=3.,
                                batch_size=batch_size,
                                num_iterations=num_crf_iterations,  # 10 for test, 5 for train
                                name='crfrnn')([fcn_score, img_input, seg_input])

    model = Model(inputs=[img_input, seg_input], outputs=crfrnn_output, name='fcn_RESNET50_8s_crfrnnSPAT')

    # Fixing weighs in lower layers (optional)
    #for layer in model.layers[:-1]: #[:181]:  # 15,21,29 (overall 30 layers) feezing until layer pred 8 (182)
    #    layer.trainable = False

    return model


def fcn_RESNET50_8s_crfrnnSPIOAT(INPUT_SIZE,nb_classes,num_crf_iterations,finetune_path, batch_size):
    """ Returns Keras FCN-8 + CRFRNNlayer with SP term and Inside/outside term, based on ResNet50 model definition.

    """
    fcn = fcn_RESNET50_8s(INPUT_SIZE, nb_classes)

    if not finetune_path=='':
        fcn.load_weights(finetune_path)

    # two inputs:
    img_input = fcn.layers[0].output
    seg_input = Input(shape=(INPUT_SIZE, INPUT_SIZE))

    #fcn_score = fcn.output
    fcn_score = fcn.get_layer('add_pred8_pred16_pred32').output

    # Adding the crfrnn layer:
    height, weight = INPUT_SIZE, INPUT_SIZE
    crfrnn_output = CrfRnnLayerAll(image_dims=(height, weight),
                                num_classes=nb_classes,
                                theta_alpha=160.,
                                theta_beta=90.,
                                theta_gamma=3.,
                                   batch_size=batch_size,
                                num_iterations=num_crf_iterations,  # 10 for test, 5 for train
                                name='crfrnn')([fcn_score, img_input, seg_input])

    model = Model(inputs=[img_input, seg_input], outputs=crfrnn_output, name='fcn_RESNET50_8s_crfrnnSPIOAT')

    # Fixing weighs in lower layers (optional)
    #for layer in model.layers[:-1]: #181]:  # 15,21,29 (overall 30 layers) feezing until layer pred 8 (182)
    #    layer.trainable = False

    return model


# def UNET(INPUT_SIZE=(256, 256, 1),nb_classes):
#     """ Returns Keras UNET implmentation, based on:
#     https://github.com/zhixuhao/unet/blob/master/model.py
#
#        """
#     inputs = Input(INPUT_SIZE)
#     #   inputs = Input(shape=(INPUT_SIZE, INPUT_SIZE, 3))
#     conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
#     conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
#     conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
#     conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
#     conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
#     conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
#     pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
#     conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
#     conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
#     drop4 = Dropout(0.5)(conv4)
#     pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
#
#     conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
#     conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
#     drop5 = Dropout(0.5)(conv5)
#
#     up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
#         UpSampling2D(size=(2, 2))(drop5))
#     merge6 = concatenate([drop4, up6], axis=3)
#     conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
#     conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
#
#     up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
#         UpSampling2D(size=(2, 2))(conv6))
#     merge7 = concatenate([conv3, up7], axis=3)
#     conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
#     conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
#
#     up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
#         UpSampling2D(size=(2, 2))(conv7))
#     merge8 = concatenate([conv2, up8], axis=3)
#     conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
#     conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
#
#     up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
#         UpSampling2D(size=(2, 2))(conv8))
#     merge9 = concatenate([conv1, up9], axis=3)
#     conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
#     conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
#     conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
#
#     #conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
#     output = (Activation('softmax'))(conv9)
#
#     model = Model(inputs=inputs, outputs=output, name='UNET')
#
#     #model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
#
#     # model.summary()
#
#     # if (pretrained_weights):
#     #     model.load_weights(pretrained_weights)
#
#     return model



def load_model_gby(model_name, INPUT_SIZE, nb_classes, num_crf_iterations, finetune_path, batch_size):#, batch_sizes_train, batch_sizes_val, batch_sizes_total):

    print('loading network type: %s..'% model_name)

    if model_name == 'fcn_8s_Sadeep':
        model = fcn_8s_Sadeep(nb_classes)
        model.crf_flag = False
        model.sp_flag = False

    elif model_name == 'fcn_8s_Sadeep_crfrnn':
        model = fcn_8s_Sadeep_crfrnn(nb_classes, num_crf_iterations, batch_size)#, batch_sizes_train, batch_sizes_val, batch_sizes_total)
        model.crf_flag = True
        model.sp_flag = False

    elif model_name == 'fcn_VGG16_32s':
        model = fcn_VGG16_32s(INPUT_SIZE, nb_classes)
        model.crf_flag = False
        model.sp_flag = False

    elif model_name == 'fcn_VGG16_32s_crfrnn':
        model = fcn_VGG16_32s_crfrnn(INPUT_SIZE, nb_classes)
        model.crf_flag = True
        model.sp_flag = False

    elif model_name == 'fcn_VGG16_8s':
        model = fcn_VGG16_8s(INPUT_SIZE, nb_classes)
        #model = fcn_8s_Sadeep(INPUT_SIZE)
        model.crf_flag = False
        model.sp_flag = False

    elif model_name == 'fcn_VGG16_8s_crfrnn':
        model = fcn_VGG16_8s_crfrnn(INPUT_SIZE, nb_classes, num_crf_iterations)
        model.crf_flag = True
        model.sp_flag = False

    elif model_name == 'fcn_RESNET50_32s':
        model = fcn_RESNET50_32s(INPUT_SIZE, nb_classes)
        model.crf_flag = False
        model.sp_flag = False

    elif model_name == 'fcn_RESNET50_32s_crfrnn':
        model = fcn_RESNET50_32s_crfrnn(INPUT_SIZE, nb_classes, num_crf_iterations)
        model.crf_flag = True
        model.sp_flag = False

    elif model_name == 'fcn_RESNET50_8s':
        model = fcn_RESNET50_8s(INPUT_SIZE, nb_classes)
        model.crf_flag = False
        model.sp_flag = False

    elif model_name == 'fcn_RESNET50_8s_crfrnn':
        model = fcn_RESNET50_8s_crfrnn(INPUT_SIZE, nb_classes, num_crf_iterations, finetune_path, batch_size)#, batch_sizes_train, batch_sizes_val, batch_sizes_total)
        model.crf_flag = True
        model.sp_flag = False

    elif model_name == 'fcn_RESNET50_8s_crfrnnSP':
        model = fcn_RESNET50_8s_crfrnnSP(INPUT_SIZE, nb_classes, num_crf_iterations, finetune_path, batch_size)
        model.crf_flag = True
        model.sp_flag = True

    elif model_name == 'fcn_RESNET50_8s_crfrnnSPIO':
        model = fcn_RESNET50_8s_crfrnnSPIO(INPUT_SIZE, nb_classes, num_crf_iterations, finetune_path, batch_size)
        model.crf_flag = True
        model.sp_flag = True

    elif model_name == 'fcn_RESNET50_8s_crfrnnSPAT':
        model = fcn_RESNET50_8s_crfrnnSPAT(INPUT_SIZE, nb_classes, num_crf_iterations, finetune_path, batch_size)
        model.crf_flag = True
        model.sp_flag = True

    elif model_name == 'fcn_RESNET50_8s_crfrnnSPIOAT':
        model = fcn_RESNET50_8s_crfrnnSPIOAT(INPUT_SIZE, nb_classes, num_crf_iterations, finetune_path, batch_size)
        model.crf_flag = True
        model.sp_flag = True

    else:
        print('ERROR: model name does not exist..')
        return


    return model


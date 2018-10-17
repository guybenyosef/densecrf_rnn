
from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, Add, Cropping2D, MaxPooling2D, Activation, Dropout, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.initializers import Constant
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from utils_gby import bilinear_upsample_weights
import sys
sys.path.insert(1, './src')
from crfrnn_layer import CrfRnnLayer
from crfrnn_layer_sp import CrfRnnLayerSP

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


def fcn_32s(INPUT_SIZE,nb_classes):
    """ Returns Keras FCN-32 model definition.

      """
    # Input and output layers for FCN:
    inputs = Input(shape=(INPUT_SIZE, INPUT_SIZE, 3))
    # Start from VGG16 layers
    vgg16 = VGG16(weights='imagenet', include_top=False, input_tensor=inputs)
    # score from the top vgg16 layer:
    score7 = vgg16.output
    score7c = Conv2D(filters=nb_classes,kernel_size=(1, 1))(score7)
    #
    score7c_upsample_32 = Conv2DTranspose(filters=nb_classes,
                                          kernel_size=(64, 64),
                                          strides=(32, 32),
                                          padding='same',
                                          activation=None,
                                          kernel_initializer=Constant(bilinear_upsample_weights(32, nb_classes)),
                                          name="score_pool7c_upsample_32")(score7c)

    fcn_output = (Activation('softmax'))(score7c_upsample_32)

    model = Model(inputs=inputs, output=fcn_output, name='fcn32s_net')

    # Fixing weighs in lower layers
    for layer in model.layers[:15]:  # sometimes I use it, sometimes not.
        layer.trainable = False
    return model


def fcn_8s_take2(INPUT_SIZE,nb_classes):
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
    model = Model(inputs=inputs, outputs=output, name='fcn8s_net')

    # Fixing weighs in lower layers
    for layer in model.layers[:15]:  # sometimes I use it, sometimes not.
        layer.trainable = False
    return model


def fcn_VGG16_32s_crfrnn(INPUT_SIZE,nb_classes):
    """ Returns Keras FCN-32 + CRFRNN layer model definition.

      """
    fcn = fcn_32s(INPUT_SIZE,nb_classes)
    saved_model_path = '/storage/gby/semseg/voc12_weights_fcn32_200ep'
    fcn.load_weights(saved_model_path)

    inputs = fcn.layers[0].output
    fcn_score = fcn.output

    # Adding the crfrnn layer:
    height, weight = INPUT_SIZE, INPUT_SIZE
    crfrnn_output = CrfRnnLayer(image_dims=(height, weight),
                             num_classes=nb_classes,
                             theta_alpha=160.,
                             theta_beta=3.,
                             theta_gamma=3.,
                             num_iterations=10,
                             name='crfrnn')([fcn_score, inputs])

    model = Model(inputs=inputs, output=crfrnn_output, name='fcn32_crfrnn_net')
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


def fcn_VGG16_8s_crfrnn(INPUT_SIZE,nb_classes):
    """ Returns Keras FCN-8 + CRFRNN layer model definition.

      """
    fcn = fcn_8s_take2(INPUT_SIZE,nb_classes)
    saved_model_path = '/storage/gby/semseg/streets_weights_fcn8s_5000ep'
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
                             theta_beta=3.,
                             theta_gamma=3.,
                             num_iterations=5, # 10 in test time, 5 in train time
                             name='crfrnn')([fcn_score, inputs])

    model = Model(inputs=inputs, output=crfrnn_output, name='fcn8_crfrnn_net')

    # # Fixing weighs in lower layers (optional)
    for layer in model.layers[:29]:  # 15,21,29 (overall 30 layers)
         layer.trainable = True #False

    return model

def fcn_8s_Sadeep_crfrnn(nb_classes):
    """ Returns Keras FCN-8 + CRFRNN layer model definition.

      """
    INPUT_SIZE = 500

    fcn = fcn_8s_Sadeep(nb_classes)
    saved_model_path = '/storage/gby/semseg/streets_weights_fcn8s_Sadeep_500ep'
    fcn.load_weights(saved_model_path)

    inputs = fcn.layers[0].output
    seg_input = fcn.layers[0].output
    # Add plenty of zero padding
    #inputs = ZeroPadding2D(padding=(100, 100))(inputs)

    fcn_score = fcn.get_layer('upscore').output
    # used to be: fcn.output
    #fcn_score = fcn.output

    # Adding the crfrnn layer:
    height, weight = INPUT_SIZE, INPUT_SIZE
    # crfrnn_output = CrfRnnLayer(image_dims=(height, weight),
    #                          num_classes=nb_classes,
    #                          theta_alpha=160.,
    #                          theta_beta=3.,
    #                          theta_gamma=3.   ,
    #                          num_iterations=0, # 10 in test time, 5 in train time
    #                          name='crfrnn')([fcn_score, inputs])

    crfrnn_output = CrfRnnLayerSP(image_dims=(height, weight),
                         num_classes=nb_classes,
                         theta_alpha=160.,
                         theta_beta=3.,
                         theta_gamma=3.,
                         num_iterations=0, #5
                         bil_rate = 0.5, #add for the segmentation
                         theta_alpha_seg = 30, #add for the segmentation
                         name='crfrnn')([fcn_score, inputs, seg_input]) #set num_iterations to 0 if we do not want crf

    model = Model(inputs=inputs, output=crfrnn_output, name='fcn8_Sadeep_crfrnn_net')

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

    model = Model(inputs=inputs, outputs=output, name='fcn32_resnet50')

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
    score_pred8_pred16_pred32 = Add()([score_pred16_pred32, score_pred8_upsample])

    output = (Activation('softmax'))(score_pred8_pred16_pred32)

    model = Model(inputs=inputs, outputs=output, name='fcn8_resnet50')

    return model

def fcn_RESNET50_8s_crfrnn(INPUT_SIZE,nb_classes):
    """ Returns Keras FCN-8 + based on ResNet50 model definition.

    """
    fcn = fcn_RESNET50_8s(INPUT_SIZE, nb_classes)
    saved_model_path = '/storage/gby/semseg/streets_weights_resnet50fcn8s_50ep'
    fcn.load_weights(saved_model_path)

    inputs = fcn.layers[0].output
    fcn_score = fcn.output

    # Adding the crfrnn layer:mv
    height, weight = INPUT_SIZE, INPUT_SIZE
    crfrnn_output = CrfRnnLayer(image_dims=(height, weight),
                                num_classes=nb_classes,
                                theta_alpha=160.,
                                theta_beta=3.,
                                theta_gamma=3.,
                                num_iterations=10,  # 10
                                name='crfrnn')([fcn_score, inputs])

    model = Model(inputs=inputs, output=crfrnn_output, name='fcn8_Resnet50_crfrnn_net')

    # Fixing weighs in lower layers (optional)
    # for layer in model.layers[:29]:  # 15,21,29 (overall 30 layers)
    #     layer.trainable = False

    return model

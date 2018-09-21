
from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, Add, Cropping2D, MaxPooling2D, Activation
from keras.initializers import Constant
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from utils_gby import bilinear_upsample_weights
import sys
sys.path.insert(1, './src')
from crfrnn_layer import CrfRnnLayer

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


def fcn_8s(INPUT_SIZE,nb_classes):
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
    score7 = vgg16.output
    score7c = Conv2D(filters=nb_classes,kernel_size=(1, 1))(score7)
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
                                   activation='sigmoid',
                                   kernel_initializer=Constant(bilinear_upsample_weights(8, nb_classes)),
                                   name="score_7_4_3_up")(score_7_4_3)

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
    model = Model(inputs=inputs, outputs=score_7_4_3_up)
    # # -- and this is fcn-32: --
    if(fcn32_flag):
        score7c_upsample_32 = Conv2DTranspose(filters=nb_classes,
                                                     kernel_size=(64, 64),
                                                     strides=(32, 32),
                                                     padding='same',
                                                     activation='sigmoid',
                                                     kernel_initializer=Constant(bilinear_upsample_weights(32, nb_classes)),
                                                     name="score_pool7c_upsample_32")(score7c)
        model = Model(inputs=inputs, outputs=score7c_upsample_32)


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


def fcn_VGG16_8s_crfrnn(INPUT_SIZE,nb_classes):
    """ Returns Keras FCN-8 + CRFRNN layer model definition.

      """
    fcn = fcn_8s_take2(INPUT_SIZE,nb_classes)
    saved_model_path = '/storage/gby/semseg/voc12_weights_fcn8_200ep'
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

    model = Model(inputs=inputs, output=crfrnn_output, name='fcn8_crfrnn_net')
    return model


def fcn_RESNET50_32s(INPUT_SIZE,nb_classes):

    """ Returns Keras FCN-32 + based on ResNet50 model definition. TODO

      """
    height, weight = INPUT_SIZE, INPUT_SIZE

    # Input and output layers for FCN:
    inputs = Input(shape=(INPUT_SIZE, INPUT_SIZE, 3))
    # Start from ResNet50 layers
    resnet50 = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)
    # score from the top resnet50 layer:
    score7 = resnet50.output
    score7c = Conv2D(filters=nb_classes,kernel_size=(1, 1))(score7)
    #
    score7c_upsample_32 = Conv2DTranspose(filters=nb_classes,
                                          kernel_size=(64, 64),
                                          strides=(32, 32),
                                          padding='same',
                                          activation='sigmoid',
                                          kernel_initializer=Constant(bilinear_upsample_weights(32, nb_classes)),
                                          name="score_pool7c_upsample_32")(score7c)

    model = Model(inputs=inputs, outputs=score7c_upsample_32, name='fcn32_resnet50')

    return model

"""
# yet another Keras implementation for fcn8
def FCN8(nClasses, input_height=224, input_width=224):
    ## input_height and width must be devisible by 32 because maxpooling with filter size = (2,2) is operated 5 times,
    ## which makes the input_height and width 2^5 = 32 times smaller
    assert input_height % 32 == 0
    assert input_width % 32 == 0
    IMAGE_ORDERING = "channels_last"

    img_input = Input(shape=(input_height, input_width, 3))  ## Assume 224,224,3

    ## Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=IMAGE_ORDERING)(
        img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING)(x)
    f1 = x

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING)(x)
    f2 = x

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format=IMAGE_ORDERING)(x)
    pool3 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format=IMAGE_ORDERING)(x)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format=IMAGE_ORDERING)(
        x)  ## (None, 14, 14, 512)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format=IMAGE_ORDERING)(pool4)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format=IMAGE_ORDERING)(x)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format=IMAGE_ORDERING)(
        x)  ## (None, 7, 7, 512)

    # x = Flatten(name='flatten')(x)
    # x = Dense(4096, activation='relu', name='fc1')(x)
    # <--> o = ( Conv2D( 4096 , ( 7 , 7 ) , activation='relu' , padding='same', data_format=IMAGE_ORDERING))(o)
    # assuming that the input_height = input_width = 224 as in VGG data

    # x = Dense(4096, activation='relu', name='fc2')(x)
    # <--> o = ( Conv2D( 4096 , ( 1 , 1 ) , activation='relu' , padding='same', data_format=IMAGE_ORDERING))(o)
    # assuming that the input_height = input_width = 224 as in VGG data

    # x = Dense(1000 , activation='softmax', name='predictions')(x)
    # <--> o = ( Conv2D( nClasses ,  ( 1 , 1 ) ,kernel_initializer='he_normal' , data_format=IMAGE_ORDERING))(o)
    # assuming that the input_height = input_width = 224 as in VGG data

    VGG_Weights_path = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    vgg = Model(img_input, pool5)
    vgg.load_weights(VGG_Weights_path)  ## loading VGG weights for the encoder parts of FCN8

    n = 4096
    o = (Conv2D(n, (7, 7), activation='relu', padding='same', name="conv6", data_format=IMAGE_ORDERING))(pool5)
    conv7 = (Conv2D(n, (1, 1), activation='relu', padding='same', name="conv7", data_format=IMAGE_ORDERING))(o)

    ## 4 times upsamping for pool4 layer
    conv7_4 = Conv2DTranspose(nClasses, kernel_size=(4, 4), strides=(4, 4), use_bias=False, data_format=IMAGE_ORDERING)(
        conv7)
    ## (None, 224, 224, 10)
    ## 2 times upsampling for pool411
    pool411 = (
        Conv2D(nClasses, (1, 1), activation='relu', padding='same', name="pool4_11", data_format=IMAGE_ORDERING))(pool4)
    pool411_2 = (
        Conv2DTranspose(nClasses, kernel_size=(2, 2), strides=(2, 2), use_bias=False, data_format=IMAGE_ORDERING))(
        pool411)

    pool311 = (
        Conv2D(nClasses, (1, 1), activation='relu', padding='same', name="pool3_11", data_format=IMAGE_ORDERING))(pool3)

    o = Add(name="add")([pool411_2, pool311, conv7_4])
    o = Conv2DTranspose(nClasses, kernel_size=(8, 8), strides=(8, 8), use_bias=False, data_format=IMAGE_ORDERING)(o)
    o = (Activation('softmax'))(o)

    model = Model(img_input, o)

    return model
"""
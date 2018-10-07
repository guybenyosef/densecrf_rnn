# here is my own implementatino of the crfrnn Keras layer


from keras import backend as K
from keras.engine.topology import Layer

# The permutohedral filtering is implemented in C++/Caffe
import high_dim_filter_loader
custom_module = high_dim_filter_loader.custom_module
import pdb

def _diagonal_initializer(shape):
    return np.eye(shape[0], shape[1], dtype=np.float32)


def _potts_model_initializer(shape):
    return -1 * _diagonal_initializer(shape)


class CrfRnnLayer_GBY(Layer):

    def __init__(self, num_classes, num_of_iterations, **kwargs):
        # params
        self.num_classes = num_classes
        self.num_of_iterations = num_of_iterations


        super(CrfRnnLayer_GBY, self).__init__(**kwargs)

    # build(input_shape):
    # this is where you will define your weights.
    # This method must set self.built = True at the end, which can be done by calling super([Layer], self).build().
    def build(self, input_shape):

        # Create a trainable weight variable for this layer.
        # here we need the CRF trainable parameters, namely the Gaussian kernels, and the label compatability function.
        self.spatial_kernel = self.add_weight(name='spatial_kernel_weights',
                                      shape=(self.num_classes, self.num_classes),
                                      initializer='uniform',
                                      trainable=True)

        self.bilateral_kernel = self.add_weight(name='bilateral_kernel_weights',
                                      shape=(self.num_classes, self.num_classes),
                                      initializer='uniform',
                                      trainable=True)

        # Compatibility matrix
        self.compatibility_matrix = self.add_weight(name='compatibility_matrix',
                                                    shape=(self.num_classes, self.num_classes),
                                                    initializer=_potts_model_initializer,
                                                    trainable=True)

        super(CrfRnnLayer_GBY, self).build(input_shape)  # Be sure to call this at the end

    # call(x):
    # this is where the layer's logic lives.
    # Unless you want your layer to support masking, you only have to care about the first argument passed to call: the input tensor.
    def call(self, x):

        unaries = tf.transpose(inputs[0][0, :, :, :], perm=(2, 0, 1))  # the fcn_scores
        rgb = tf.transpose(inputs[1][0, :, :, :], perm=(2, 0, 1))  # the raw rgb
        # pdb.set_trace()
        c, h, w = self.num_classes, self.image_dims[0], self.image_dims[1]

        self.Mui = compute_mui(rgb)
        self.F = compute_F(rgb)

        # Normalize Q:
        Z[i] = sum(exp(U(i,l)))
        Q  = unaries

        # Iterate:
        for ii in range (self.num_of_iterations)
            # message passing with Gaussian filters:
            # compute filter responses:

            # Spatial filtering (a.k.a the 'smoothness kernel')
            spatial_out = custom_module.high_dim_filter(softmax_out, rgb, bilateral=False,
                                                        theta_gamma=self.theta_gamma)
            spatial_out = spatial_out / spatial_norm_vals

            # Bilateral filtering (a.k.a the 'appearance kernel')
            bilateral_out = custom_module.high_dim_filter(softmax_out, rgb, bilateral=True,
                                                          theta_alpha=self.theta_alpha,
                                                          theta_beta=self.theta_beta)
            bilateral_out = bilateral_out / bilateral_norm_vals

            Q = Q*self.bilateral_kernel(F,F)
            # weighting filter outputs:
            Q = K.dot(self.spatial_kernel, Q)
            # Compatability transform:
            Q = Q * self.Mui
            # Adding unary potentials:
            Q = U - Q
            # Normalizing:
            Q = (1/Z)*exp(Q)

        return Q





    # compute_output_shape(input_shape):
    # in case your layer modifies the shape of its input, you should specify here the shape transformation logic.
    # This allows Keras to do automatic shape inference.
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
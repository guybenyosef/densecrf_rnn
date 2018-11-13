"""
MIT License
Copyright (c) 2017 Sadeep Jayasumana
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
# To print tensor values
# for i in range(1):
#     my_tensor = tf.Print(my_tensor, [my_tensor[i]], message="q_values first 500 ", summarize=500)
import numpy as np
import tensorflow as tf
from keras.engine.topology import Layer
import high_dim_filter_loader
import random
import time

custom_module = high_dim_filter_loader.custom_module

def _diagonal_initializer(shape):
    return np.eye(shape[0], shape[1], dtype=np.float32)


def _potts_model_initializer(shape):
    return -1 * _diagonal_initializer(shape)

def _sp_high_weight_initializer(shape):
    return [1]

def _sp_low_weight_initializer(shape):
    return np.ones(shape, dtype=np.float32)

class CrfRnnLayerSP(Layer):
    """ Implements the CRF-RNN layer described in:
    Conditional Random Fields as Recurrent Neural Networks,
    S. Zheng, S. Jayasumana, B. Romera-Paredes, V. Vineet, Z. Su, D. Du, C. Huang and P. Torr,
    ICCV 2015
    Based on: https://github.com/liyin2015/superpixel_crfasrnn.git
    """

    def __init__(self, image_dims, num_classes,
                 theta_alpha, theta_beta, theta_gamma,
                 num_iterations, bil_rate=0.5, theta_alpha_seg=None, **kwargs):  # add theta_alpha_seg
        self.image_dims = image_dims
        self.num_classes = num_classes
        self.theta_alpha = theta_alpha
        self.theta_alpha_seg = theta_alpha_seg  # to add sp-pairwise
        self.theta_beta = theta_beta
        self.theta_gamma = theta_gamma
        self.num_iterations = num_iterations
        self.spatial_ker_weights = None
        self.bilateral_ker_weights = None
        self.compatibility_matrix = None
        self.spatial_norm_vals = None  # to modularize
        self.bilateral_norm_vals = None
        self.bilateral_outs = []
        self.bil_rate = bil_rate  # ratio of wegiths
        super(CrfRnnLayerSP, self).__init__(**kwargs)

    def build(self, input_shape):
        # Weights of the spatial kernel
        self.spatial_ker_weights = self.add_weight(name='spatial_ker_weights',
                                                   shape=(self.num_classes, self.num_classes),
                                                   initializer=_diagonal_initializer,
                                                   trainable=True)

        # Weights of the bilateral kernel
        self.bilateral_ker_weights = self.add_weight(name='bilateral_ker_weights',
                                                     shape=(self.num_classes, self.num_classes),
                                                     initializer=_diagonal_initializer,
                                                     trainable=True)

        # Compatibility matrix
        self.compatibility_matrix = self.add_weight(name='compatibility_matrix',
                                                    shape=(self.num_classes, self.num_classes),
                                                    initializer=_potts_model_initializer,
                                                    trainable=True)

        super(CrfRnnLayerSP, self).build(input_shape)

    def filtering_norming(self, imgs):
        c, h, w = self.num_classes, self.image_dims[0], self.image_dims[1]
        all_ones = np.ones((c, h, w), dtype=np.float32)
        # Prepare filter normalization coefficients, they are tensors
        self.spatial_norm_vals = custom_module.high_dim_filter(all_ones, imgs[0], bilateral=False,
                                                               theta_gamma=self.theta_gamma)
        self.bilateral_norm_vals = [custom_module.high_dim_filter(all_ones, imgs[0], bilateral=True,
                                                                  theta_alpha=self.theta_alpha,
                                                                  theta_beta=self.theta_beta)]  # add original image
        for i in range(1, len(imgs)):
            theta_alpha_seg = self.theta_alpha_seg if self.theta_alpha_seg is not None else self.theta_alpha
            self.bilateral_norm_vals.append(custom_module.high_dim_filter(all_ones, imgs[i], bilateral=True,
                                                                          theta_alpha=theta_alpha_seg,
                                                                          theta_beta=self.theta_beta))  # add segmented image

    def bilateral_filtering(self, softmax_out, imgs):
        #         bilateral_outs = []
        self.bilateral_outs = []
        self.bilateral_outs.append(custom_module.high_dim_filter(softmax_out, imgs[0], bilateral=True,
                                                                 theta_alpha=self.theta_alpha,
                                                                 theta_beta=self.theta_beta))
        if len(imgs) > 1:  # we have segmentations
            for i in range(1, len(imgs)):
                theta_alpha_seg = self.theta_alpha_seg if self.theta_alpha_seg is not None else self.theta_alpha
                self.bilateral_outs.append(custom_module.high_dim_filter(softmax_out, imgs[i], bilateral=True,
                                                                         theta_alpha=theta_alpha_seg,
                                                                         theta_beta=self.theta_beta))

        self.bilateral_outs = [out / norm for (out, norm) in zip(self.bilateral_outs, self.bilateral_norm_vals)]

    def call(self, inputs):
        start_time = time.time()
        unaries = tf.transpose(inputs[0][0, :, :, :], perm=(2, 0, 1))
        rgb = tf.transpose(inputs[1][0, :, :, :], perm=(2, 0, 1))
        segs = [tf.transpose(inputs[i][0, :, :, :], perm=(2, 0, 1)) for i in range(2, len(inputs))]

        c, h, w = self.num_classes, self.image_dims[0], self.image_dims[1]

        # Prepare filter normalization coefficients, they are tensors
        self.filtering_norming([rgb] + segs)
        q_values = unaries

        for i in range(self.num_iterations):
            softmax_out = tf.nn.softmax(q_values, 0)

            # Spatial filtering
            spatial_out = custom_module.high_dim_filter(softmax_out, rgb, bilateral=False,
                                                        theta_gamma=self.theta_gamma)
            spatial_out = spatial_out / self.spatial_norm_vals

            # Bilateral filtering
            self.bilateral_filtering(softmax_out, [rgb] + segs)

            # Weighting filter outputs

            message_passing = tf.matmul(self.spatial_ker_weights,
                                        tf.reshape(spatial_out, (c, -1)))
            ratios = [1.0] + [self.bil_rate] * len(segs)
            message_passing += tf.add_n([tf.matmul(self.bilateral_ker_weights * ratios[i],
                                                   tf.reshape(self.bilateral_outs[i], (c, -1))) for i in
                                         range(len(segs) + 1)])

            # Compatibility transform
            pairwise = tf.matmul(self.compatibility_matrix, message_passing)

            # Adding unary potentials
            pairwise = tf.reshape(pairwise, (c, h, w))
            q_values = unaries - pairwise
        elapsed_time = time.time() - start_time
        print(elapsed_time)

        return tf.transpose(tf.reshape(q_values, (1, c, h, w)), perm=(0, 2, 3, 1))

    def compute_output_shape(self, input_shape):
        return input_shape


class CrfRnnLayerSPIO(Layer):
    """ Implements the CRF-RNN layer described in:

    Conditional Random Fields as Recurrent Neural Networks,
    S. Zheng, S. Jayasumana, B. Romera-Paredes, V. Vineet, Z. Su, D. Du, C. Huang and P. Torr,
    ICCV 2015
    """

    def __init__(self, image_dims, num_classes,
                 theta_alpha, theta_beta, theta_gamma,
                 num_iterations, **kwargs):
        self.image_dims = image_dims
        self.num_classes = num_classes
        self.theta_alpha = theta_alpha
        self.theta_beta = theta_beta
        self.theta_gamma = theta_gamma
        self.num_iterations = num_iterations
        self.spatial_ker_weights = None
        self.bilateral_ker_weights = None
        self.compatibility_matrix = None
        super(CrfRnnLayerSPIO, self).__init__(**kwargs)

    def build(self, input_shape):
        # Weights of the spatial kernel
        self.spatial_ker_weights = self.add_weight(name='spatial_ker_weights',
                                                   shape=(self.num_classes, self.num_classes),
                                                   initializer=_diagonal_initializer,
                                                   trainable=True)

        # Weights of the bilateral kernel
        self.bilateral_ker_weights = self.add_weight(name='bilateral_ker_weights',
                                                     shape=(self.num_classes, self.num_classes),
                                                     initializer=_diagonal_initializer,
                                                     trainable=True)

        # Weights of the superpixel term
        self.superpixel_low_weights = self.add_weight(name='superpixel_low_weights',
                                                      shape=(self.num_classes),
                                                      initializer=_sp_low_weight_initializer,
                                                      trainable=True)

        self.superpixel_high_weight = self.add_weight(name='superpixel_high_weight',
                                                      shape=(1),
                                                      initializer=_sp_high_weight_initializer,
                                                      trainable=True)
        # Compatibility matrix
        self.compatibility_matrix = self.add_weight(name='compatibility_matrix',
                                                    shape=(self.num_classes, self.num_classes),
                                                    initializer=_potts_model_initializer,
                                                    trainable=True)

        super(CrfRnnLayerSPIO, self).build(input_shape)

    def call(self, inputs):
        unaries = tf.transpose(inputs[0][0, :, :, :], perm=(2, 0, 1)) # the fcn_scores
        rgb = tf.transpose(inputs[1][0, :, :, :], perm=(2, 0, 1)) # the raw rgb
        superpixel_cliques = tf.transpose(inputs[2][0,:,:])
        # h = num_rows, w = num_cols
        c, h, w = self.num_classes, self.image_dims[0], self.image_dims[1]
        all_ones = np.ones((c, h, w), dtype=np.float32)
        y, indx = tf.unique(superpixel_cliques)

        # Prepare filter normalization coefficients
        spatial_norm_vals = custom_module.high_dim_filter(all_ones, rgb, bilateral=False,
                                                          theta_gamma=self.theta_gamma)
        bilateral_norm_vals = custom_module.high_dim_filter(all_ones, rgb, bilateral=True,
                                                            theta_alpha=self.theta_alpha,
                                                            theta_beta=self.theta_beta)

        q_values = unaries

        # for i in range(1):
        #     my_tensor = tf.Print(my_tensor, [my_tensor[i]], message="q_values first 500 ", summarize=500)
        for i in range(self.num_iterations):
            softmax_out = tf.nn.softmax(q_values, 0)
            softmax_out = tf.Print(softmax_out, [softmax_out[0]], message="softmax out", summarize=5)
            # Spatial filtering
            spatial_out = custom_module.high_dim_filter(softmax_out, rgb, bilateral=False,
                                                        theta_gamma=self.theta_gamma)
            spatial_out = spatial_out / spatial_norm_vals

            # Bilateral filtering
            bilateral_out = custom_module.high_dim_filter(softmax_out, rgb, bilateral=True,
                                                          theta_alpha=self.theta_alpha,
                                                          theta_beta=self.theta_beta)
            bilateral_out = bilateral_out / bilateral_norm_vals

            # compute superpixel tensor:
            sp_map = superpixel_cliques
            # replicate the sp_map m times and have the shape of [rows,cols,m], where m in the number of labels
            extended_sp_map = tf.stack([sp_map] * c)

            prod_tensor = tf.zeros(shape=(c,h,w))
            #idxs = tf.range(tf.shape(y)[0])
            #ridxs = tf.random_shuffle(idxs)[:5]
            #rinput = tf.gather(y, ridxs)
            
            for sp_indx in random.sample(range(1,400), 5):
                # This will put True where sp index is sp_indx, False otherwise:
                cond_sp_indx = tf.equal(extended_sp_map,sp_indx)
                #cond_sp_indx = tf.Print(cond_sp_indx, [cond_sp_indx], message="cond sp indx ", summarize=5)
                # put 1 in q_vals if doesn't belong to sp_indx:
                #q_values = tf.Print(q_values, [q_values], message="q values ", summarize=5)
                A = tf.multiply(tf.to_float(cond_sp_indx), softmax_out) + tf.to_float(tf.logical_not(cond_sp_indx))
                #A = tf.Print(A, [A], message="A ", summarize=5)
                # compute the product for each label:
                B = tf.reduce_prod(A, [1, 2])
                #B = tf.Print(B, [B], message="B ", summarize=5)
                # Create a tensor where each cell contains the product for its superpiel sp_indx and its label l:
                C = tf.stack([B]*(h*w))
                C = tf.reshape(tf.transpose(C), (c, h, w))
                C = tf.multiply(tf.to_float(cond_sp_indx), C)
                #C = tf.Print(C, [C], message="C ", summarize=5)
                # add this to the overall product tensor; each cell contains the 'product' for its update rule:
                prod_tensor += C

            # the actual product: we need to divide it by the current q_vals
            first_term = tf.divide(tf.to_float(prod_tensor),q_values)
            superpixel_low_weights_duplicated = tf.transpose(tf.stack([self.superpixel_low_weights] * (h * w)))
            first_term_resp = tf.multiply(superpixel_low_weights_duplicated, tf.reshape(first_term, (c, -1)))
            first_term_resp_back = tf.reshape(first_term_resp, (c, h, w))
            superpixel_update = first_term_resp_back + self.superpixel_high_weight * (tf.ones(shape=(c,h,w)) - first_term)

            superpixel_update = tf.Print(superpixel_update, [superpixel_update[0]], message="sp first 5 ", summarize=5)

            # Weighting filter outputs
            message_passing = (tf.matmul(self.spatial_ker_weights,
                                         tf.reshape(spatial_out, (c, -1))) +
                               tf.matmul(self.bilateral_ker_weights,
                                         tf.reshape(bilateral_out, (c, -1)))
                               )

            # Compatibility transform
            pairwise = tf.matmul(self.compatibility_matrix, message_passing)

            # Adding unary potentials
            pairwise = tf.reshape(pairwise, (c, h, w))

            q_values = unaries - pairwise - superpixel_update
            #for i in range(1):
            #    q_values = tf.Print(q_values, [q_values[i]], message="q_values first 5 ", summarize=5)

        return tf.transpose(tf.reshape(q_values, (1, c, h, w)), perm=(0, 2, 3, 1))

    def compute_output_shape(self, input_shape):
        return input_shape

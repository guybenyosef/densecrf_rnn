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

import numpy as np
import tensorflow as tf
from keras.engine.topology import Layer
import high_dim_filter_loader
custom_module = high_dim_filter_loader.custom_module
import time
import random
import pdb

def _diagonal_initializer(shape):
    return np.eye(shape[0], shape[1], dtype=np.float32)


def _potts_model_initializer(shape):
    return -1 * _diagonal_initializer(shape)

def _sp_high_weight_initializer(shape):
    return [1]

def _sp_low_weight_initializer(shape):
    return np.ones(shape, dtype=np.float32)


class CrfRnnLayer(Layer):
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
        super(CrfRnnLayer, self).__init__(**kwargs)

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

        super(CrfRnnLayer, self).build(input_shape)

    def call(self, inputs):

        unaries = tf.transpose(inputs[0][0, :, :, :], perm=(2, 0, 1)) # the fcn_scores
        rgb = tf.transpose(inputs[1][0, :, :, :], perm=(2, 0, 1)) # the raw rgb
        #pdb.set_trace()
        c, h, w = self.num_classes, self.image_dims[0], self.image_dims[1]
        all_ones = np.ones((c, h, w), dtype=np.float32)

        # Prepare filter normalization coefficients
        spatial_norm_vals = custom_module.high_dim_filter(all_ones, rgb, bilateral=False,
                                                          theta_gamma=self.theta_gamma)
        bilateral_norm_vals = custom_module.high_dim_filter(all_ones, rgb, bilateral=True,
                                                            theta_alpha=self.theta_alpha,
                                                            theta_beta=self.theta_beta)
        q_values = unaries
        # for i in range(1):
        #     q_values = tf.Print(q_values, [q_values[i]], message="unaries first 500 ", summarize=500)

        for i in range(self.num_iterations):
            softmax_out = tf.nn.softmax(q_values, 0)

            # Spatial filtering
            spatial_out = custom_module.high_dim_filter(softmax_out, rgb, bilateral=False,
                                                        theta_gamma=self.theta_gamma)
            spatial_out = spatial_out / spatial_norm_vals

            # Bilateral filtering
            bilateral_out = custom_module.high_dim_filter(softmax_out, rgb, bilateral=True,
                                                          theta_alpha=self.theta_alpha,
                                                          theta_beta=self.theta_beta)
            bilateral_out = bilateral_out / bilateral_norm_vals

            # Weighting filter outputs
            message_passing = (tf.matmul(self.spatial_ker_weights,
                                         tf.reshape(spatial_out, (c, -1))) +
                               tf.matmul(self.bilateral_ker_weights,
                                         tf.reshape(bilateral_out, (c, -1))))

            # Compatibility transform
            pairwise = tf.matmul(self.compatibility_matrix, message_passing)

            # Adding unary potentials
            pairwise = tf.reshape(pairwise, (c, h, w))
            q_values = unaries - pairwise
            #pdb.set_trace()
            # for i in range(1):
            #     q_values = tf.Print(q_values, [q_values[i]], message="q_values first 500 ", summarize=500)
            # pdb.set_trace()

        return tf.transpose(tf.reshape(q_values, (1, c, h, w)), perm=(0, 2, 3, 1))

    def compute_output_shape(self, input_shape):
        return input_shape



class CrfRnnLayerSP(Layer):
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

        # Weights of the superpixel term
        self.superpixel_low_weights = self.add_weight(name='superpixel_low_weights',
                                                      shape=(self.num_classes),
                                                      initializer=_sp_low_weight_initializer,
                                       #              shape=(self.num_classes, self.num_classes),
                                   #                   initializer=_diagonal_initializer,
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

        super(CrfRnnLayerSP, self).build(input_shape)

    def call(self, inputs):

        unaries = tf.transpose(inputs[0][0, :, :, :], perm=(2, 0, 1)) # the fcn_scores
        rgb = tf.transpose(inputs[1][0, :, :, :], perm=(2, 0, 1)) # the raw rgb
        #segs = [tf.transpose(inputs[i][0, :, :, :], perm=(2, 0, 1)) for i in range(2, len(inputs))]
        superpixel_cliques = tf.transpose(inputs[2][0,:,:])  # perm=(0,1)

        c, h, w = self.num_classes, self.image_dims[0], self.image_dims[1]

        all_ones = np.ones((c, h, w), dtype=np.float32)

        # Prepare filter normalization coefficients
        spatial_norm_vals = custom_module.high_dim_filter(all_ones, rgb, bilateral=False,
                                                          theta_gamma=self.theta_gamma)
        bilateral_norm_vals = custom_module.high_dim_filter(all_ones, rgb, bilateral=True,
                                                            theta_alpha=self.theta_alpha,
                                                            theta_beta=self.theta_beta)

        q_values = unaries

#        pdb.set_trace()
#         my_tensor = q_values
#          for i in range(1):
#              my_tensor = tf.Print(my_tensor, [my_tensor[i]], message="q_values first 500 ", summarize=500)
#         #pdb.set_trace()
        superpixel_update = tf.get_variable("superpixel_update", [c, h, w], dtype=tf.float32, initializer=tf.zeros_initializer)
        for i in range(self.num_iterations):
            #pdb.set_trace()
            softmax_out = tf.nn.softmax(q_values, 0)

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
            # ---------------------------
            #sp_map = segs[0][0,:,:]
            sp_map = superpixel_cliques

            # replicate the sp_map m times and have the shape of [rows,cols,m), where m in the number of labels
            extended_sp_map = tf.stack([sp_map] * c)

            # initiate to zeros
            prod_tensor = tf.zeros(shape=(c, h, w))

            # iterate over all superpixels
            for sp_indx in random.sample(range(1,256), 5):  # sampling superpixels, otherwise memory is overloaded
                #print(sp_indx)
                # This will put True where where sp index is sp_indx, False otherwise:
                cond_sp_indx = tf.equal(extended_sp_map, sp_indx)

                # put 1 in q_vqls if not belongs to sp_indx:  ## (using tf.tensordot rather than tf.multiply)
                # q_val_for_sp_padded = tf.tensordot(tf.to_float(cond_sp_indx), q_values, axes=1) + tf.to_float(tf.logical_not(cond_sp_indx))
                q_val_for_sp_padded = tf.multiply(tf.to_float(cond_sp_indx), q_values) + tf.to_float(tf.logical_not(cond_sp_indx))

                # compute the product for each label:
                B = tf.reduce_prod(q_val_for_sp_padded, [1, 2])

                # Create a tensor where each cell contains the product for its superpiel sp_indx and its label l:
                C = tf.stack([B] * (h * w))
                C = tf.reshape(tf.transpose(C), (c, h, w))
                C = tf.tensordot(tf.to_float(cond_sp_indx), C, axes=1)  ## (using tf.tensordot rather than tf.multiply

                # add this to the overall product tensor; each cell contains the 'product' for its update rule:
                prod_tensor += q_val_for_sp_padded# C

            # and now the update rule for superpixel
            # the actual product: we need to divide it by the current q_vals
            first_term = tf.divide(tf.to_float(prod_tensor), q_values)

            # multiply by weights:
            #first_term_resp = tf.matmul(self.superpixel_low_weights, tf.reshape(first_term, (c, -1)))
            superpixel_low_weights_duplicated = tf.transpose(tf.stack([self.superpixel_low_weights] * (h * w))) 
            first_term_resp = tf.multiply(superpixel_low_weights_duplicated, tf.reshape(first_term, (c, -1)))
            first_term_resp_back = tf.reshape(first_term_resp, (c, h, w))

            superpixel_update = first_term_resp_back + self.superpixel_high_weight * (tf.ones(shape=(c,h,w)) - first_term)

            # ---- old
            # superpixel_out = tf.multiply(tf.to_float(T), q_values) # + self.superpixel_ker_weights[1] * tf.multiply(tf.to_float(tf.logical_not(T)), q_values)
            # superpixel = tf.matmul(self.superpixel_ker_weights,tf.reshape(superpixel_out, (c, -1)))
            # superpixel_update = tf.reshape(superpixel, (c, h, w))


            #superpixel_out += self.superpixel_ker_weights[0] * tf.to_float(T) + self.superpixel_ker_weights[1] * tf.to_float(tf.logical_not(T))
            #tf.reduce_sum(superpixel_out

            # Weighting filter outputs
            message_passing = (tf.matmul(self.spatial_ker_weights,
                                         tf.reshape(spatial_out, (c, -1))) +
                               tf.matmul(self.bilateral_ker_weights,
                                         tf.reshape(bilateral_out, (c, -1)))
            #                   tf.matmul(self.sp_ker_weights
            #                             tf.reshape(superpixel_out, (c, -1)))
                               )

            # Compatibility transform
            pairwise = tf.matmul(self.compatibility_matrix, message_passing)

            # Adding unary potentials
            pairwise = tf.reshape(pairwise, (c, h, w))

            q_values = unaries - pairwise + superpixel_update
            #pdb.set_trace()
            # for i in range(1):
            #     q_values = tf.Print(q_values, [q_values[i]], message="q_values first 500 ", summarize=500)
            #pdb.set_trace()

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
        self.superpixel_ker_weights = self.add_weight(name='superpixel_ker_weights',
                                                      shape=(self.num_classes, self.num_classes), # shape=(2,1),   # [w_low,w_high] #  #self.num_classes, self.num_classes),
                                                      initializer=_diagonal_initializer,
                                                      trainable=True)

        # Weights of the superpixel term
        self.containment_ker_weights = self.add_weight(name='containment_ker_weights',
                                                      shape=(self.num_classes, self.num_classes),
                                                      initializer=_diagonal_initializer,
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
        segs = [tf.transpose(inputs[i][0, :, :, :], perm=(2, 0, 1)) for i in range(2, len(inputs))]

        c, h, w = self.num_classes, self.image_dims[0], self.image_dims[1]

        all_ones = np.ones((c, h, w), dtype=np.float32)

        # Prepare filter normalization coefficients
        spatial_norm_vals = custom_module.high_dim_filter(all_ones, rgb, bilateral=False,
                                                          theta_gamma=self.theta_gamma)
        bilateral_norm_vals = custom_module.high_dim_filter(all_ones, rgb, bilateral=True,
                                                            theta_alpha=self.theta_alpha,
                                                            theta_beta=self.theta_beta)
      #  my_tensor = tf.where(tf.equal(segs[0],23))

        q_values = unaries

#        pdb.set_trace()
        # for i in range(1):
        #     my_tensor = tf.Print(my_tensor, [my_tensor[i]], message="q_values first 500 ", summarize=500)
        # pdb.set_trace()

        for i in range(self.num_iterations):
            softmax_out = tf.nn.softmax(q_values, 0)

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
            # ----------------------------
            # compute the dominant label:

            sp_map = segs[0][0,:,:]

            # replicate the sp_map m times and have the shape of [rows,cols,m), where m in the number of labels
            extended_sp_map = tf.stack([sp_map] * self.num_classes)

            # This will put True where the max prob label, False otherwise:
            cond_max_label = tf.equal(q_values, tf.reduce_max(q_values, axis=0))

            # initiate to zeros
            #superpixel_out = tf.zeros(extended_sp_map.shape)
            superpixel_cond = tf.constant(False, shape=extended_sp_map.shape)


            # iterate over all superpixels
            for sp_indx in random.sample(range(1,256), 50):  # sampling superpixels, otherwise memory is overloaded
                # This will put True where where sp index is sp_indx, False otherwise:
                cond_sp_indx = tf.equal(extended_sp_map, sp_indx)
                # This is tensor T, where the dominant label for sp_indx superpixel is:
                T = tf.logical_and(cond_max_label, cond_sp_indx)
                  #= tf.logical_or(superpixel_cond,T)

            # update meanfield rule for superpixel:
            superpixel_out = tf.multiply(tf.to_float(T), q_values) # + self.superpixel_ker_weights[1] * tf.multiply(tf.to_float(tf.logical_not(T)), q_values)
            superpixel = tf.matmul(self.superpixel_ker_weights,tf.reshape(superpixel_out, (c, -1)))
            superpixel_update = tf.reshape(superpixel, (c, h, w))

            # update meanfield rule for containment:
            containment_out = tf.multiply(tf.to_float(T),q_values)  # + self.superpixel_ker_weights[1] * tf.multiply(tf.to_float(tf.logical_not(T)), q_values)
            containment = tf.matmul(self.containment_ker_weights, tf.reshape(containment_out, (c, -1)))
            containment_update = tf.reshape(containment, (c, h, w))


            # Weighting filter outputs
            message_passing = (tf.matmul(self.spatial_ker_weights,
                                         tf.reshape(spatial_out, (c, -1))) +
                               tf.matmul(self.bilateral_ker_weights,
                                         tf.reshape(bilateral_out, (c, -1)))
            #                   tf.matmul(self.sp_ker_weights
            #                             tf.reshape(superpixel_out, (c, -1)))
                               )

            # Compatibility transform
            pairwise = tf.matmul(self.compatibility_matrix, message_passing)

            # Adding unary potentials
            pairwise = tf.reshape(pairwise, (c, h, w))

            q_values = unaries - pairwise + superpixel_update
            #pdb.set_trace()
            # for i in range(1):
            #     q_values = tf.Print(q_values, [q_values[i]], message="q_values first 500 ", summarize=500)
            #pdb.set_trace()

        return tf.transpose(tf.reshape(q_values, (1, c, h, w)), perm=(0, 2, 3, 1))

    def compute_output_shape(self, input_shape):
        return input_shape

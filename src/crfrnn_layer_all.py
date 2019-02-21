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

def _low_weight_initializer(shape):
    return np.ones(shape=shape, dtype=np.float32)

def _compute_superpixel_update(q_values,superpixel_low_weights,superpixel_high_weights,superpixel_cliques, sp_indices, c, h, w ):
    # compute superpixel tensor:
    # ---------------------------
    sp_map = superpixel_cliques

    # replicate the sp_map m times and have the shape of [rows,cols,m], where m is the number of labels
    extended_sp_map = tf.tile(tf.reshape(superpixel_cliques, [1, h, w]), [c, 1, 1])
    
    def while_body(index, prod_tensor):
        sp_indx = tf.to_float(index)
        cond_sp_indx = tf.equal(extended_sp_map, sp_indx)
        # Adding epsilon = 0.00001 to q_values before taking log, to prevent nan
        q_val_for_sp = tf.multiply(tf.to_float(cond_sp_indx), q_values + 0.00001) + tf.to_float(tf.logical_not(cond_sp_indx))

        B = tf.reduce_sum(tf.log(q_val_for_sp), axis=[1,2])
    
        C1 = tf.tile(tf.reshape(B, [1,6]), [h*w, 1])    
        C2 = tf.reshape(tf.transpose(C1), (c,h,w))
        C3 = tf.to_float(tf.multiply(tf.to_float(cond_sp_indx), C2))

        prod_tensor += C3
        return index+1, prod_tensor

    i = 0
    # initialize to zeros
    prod_tensor = tf.zeros(shape=(c, h, w))
    cond = lambda i, prod_tensor: tf.less(i, len(sp_indices))
    i, prod_tensor = tf.while_loop(cond, while_body, [i, prod_tensor], parallel_iterations=len(sp_indices))

    # adding epsilon = 0.00001 to q_vals before taking log, to prevent nan
    first_term_unexp = tf.subtract(tf.to_float(prod_tensor), tf.log(q_values+ 0.00001))
    first_term = tf.exp(first_term_unexp)

    # multiply by weights:
    superpixel_low_weights_duplicated = tf.transpose(tf.tile(tf.reshape(superpixel_low_weights, [1,6]), [h*w, 1]))
    first_term_resp = tf.multiply(superpixel_low_weights_duplicated, tf.reshape(first_term, (c, -1)))
    first_term_resp_back = tf.reshape(first_term_resp, (c, h, w))

    superpixel_update = first_term_resp_back + superpixel_high_weights * (tf.ones(shape=(c, h, w)) - first_term)

    #To avoid nan in final update, can also clip if other methods don't work
    #clip = tf.clip_by_value(superpixel_update,low,high)
    return superpixel_update

def _compute_containment_update(q_values,containment_low_weights,containment_high_weights,bd_map, sp_indices, c, h, w ):

    extended_bd_map = tf.stack([bd_map]*c)

    bool_max_label = tf.equal(q_values, tf.reduce_max(q_values,axis=0))
    # While loop for product tensor
    i = 0
    prod_tensor_io = tf.zeros(shape=(c,h,w))
    q_val_sum_tensor = tf.zeros(shape=(c,h,w))
    def while_body(index, prod_tensor_io, q_val_sum_tensor):
        sp_indx = tf.to_float(index)
        # This will put True where bd index is clique_indx, False otherwise:
        bool_bd_indx = tf.equal(extended_bd_map,sp_indx)
        q_val_for_clique = tf.multiply(tf.to_float(bool_bd_indx), q_values + 0.00001)

        # put 1 in q_vals if a pixel is not in sp_indx:
        q_val_for_clique_padded = q_val_for_clique  + tf.to_float(tf.logical_not(bool_bd_indx))
        maxlabel_q_val_for_bd = tf.reduce_max(q_val_for_clique,axis=0)

        # here we put q_val[r,c,l] = q_val[r,c,l'] where l' is the dominant label (only for pixels in clique_indx)
        maxlabel_q_val_for_bd_duplicated = tf.stack([maxlabel_q_val_for_bd] * c)

        # here we compute: q_val(r,c,l) + q_val(r,c,l') where l' is the dominant label in the clique
        A = q_val_for_clique_padded + maxlabel_q_val_for_bd_duplicated
        A_no_padding = q_val_for_clique + maxlabel_q_val_for_bd_duplicated

        # Subtract q_val(r,c,l') from indices where l = l'
        l_prime_equals_l = tf.multiply(tf.to_float(tf.logical_and(bool_max_label, bool_bd_indx)), q_values)
        A_no_padding_minus_l = tf.subtract(A_no_padding, l_prime_equals_l)

        q_val_sum_tensor+= A_no_padding_minus_l  # A

        # compute the product for each label:
        #B = tf.reduce_prod(A, [1, 2]) # less stable
        #B = tf.reduce_logsumexp(A_no_padding_minus_l, [1, 2])  # more stable
        B = tf.reduce_sum(tf.log(A_no_padding_minus_l), axis=[1,2])

        # Create a tensor where each cell contains the product for its boundary clique_indx and its label l:
        C1 = tf.stack([B] * (h * w))
        C2 = tf.reshape(tf.transpose(C1), (c, h, w))
        C3 = tf.multiply(tf.to_float(bool_bd_indx), C2)

        # add this to the overall product tensor; each cell contains the 'product' for its update rule:
        prod_tensor_io += tf.multiply(tf.to_float(bool_bd_indx), C3)

        return index+1, prod_tensor_io, q_val_sum_tensor

    cond = lambda i, prod_tensor_io, q_val_sum_tensor: tf.less(i, len(sp_indices))
    i, prod_tensor_io, q_val_sum_tensor = tf.while_loop(cond, while_body, [i, prod_tensor_io, q_val_sum_tensor], parallel_iterations=len(sp_indices), back_prop=False)
    
    # Add 1 to q_val_sum_tensor where it is 0
    bool_sum_zero = tf.equal(q_val_sum_tensor, 0)
    bool_sum_one = tf.to_float(bool_sum_zero)
    q_val_sum_tensor += bool_sum_one

    # the actual product: we need to divide it by each index's q_val(r,c,l) + q_val(r,c,l')
    #first_term = tf.divide(tf.to_float(prod_tensor_io), q_val_sum_tensor)
    first_term_cont_unexp = tf.subtract(tf.to_float(prod_tensor_io), tf.log(q_val_sum_tensor + 0.00001)) # may not need to add epsilon here
    first_term_cont = tf.exp(first_term_unexp)
    containment_low_weights_duplicated = tf.transpose(tf.stack([containment_low_weights] * (h * w)))
    first_term_resp = tf.multiply(containment_low_weights_duplicated, tf.reshape(first_term_cont, (c,-1)))
    first_term_resp_back = tf.reshape(first_term_resp, (c, h, w))
    containment_update = first_term_resp_back + containment_high_weights * (tf.ones(shape=(c,h,w)) - first_term_cont)

    return containment_update

def _compute_attachment_update(q_values,attachment_low_weights,attachment_high_weights,sp_map, sp_indices, c, h, w ):

    prod_tensor_att = tf.zeros(shape=(c, h, w))

    extended_att_map = tf.stack([sp_map] * c)

    def while_body(l1, prod_tensor_att):
        index_l1 = tf.to_float(l1)
        
        # Get locations of first sp in clique
        bool_sp_indx1 = tf.equal(extended_att_map, index_l1)
        A1 = tf.multiply(tf.to_float(bool_sp_indx1), q_values + 0.00001) + tf.to_float(tf.logical_not(bool_sp_indx1))
        #B1 = tf.reduce_logsumexp(A1, [1, 2])
        B1 = tf.reduce_sum(tf.log(A1), axis=[1,2])
        
        C1_1 = tf.stack([B1] * (h * w))
        C1_2 = tf.reshape(tf.transpose(C1_1), (c, h, w))
        C1_3 = tf.multiply(tf.to_float(bool_sp_indx1), C1_2)
        
        l2 = 0
        inner_cond = lambda l2, prod_tensor_att: tf.less(l2, l1)
        def inner_while_body(l2, prod_tensor_att):
            index_l2 = tf.to_float(l2)
            bool_sp_indx2 = tf.equal(extended_att_map, index_l2)
            
            # Don't put 1 in q_values anymore if doesn't belong to this clique
            A2 = tf.multiply(tf.to_float(bool_sp_indx2), q_values + 0.00001) + tf.to_float(tf.logical_not(bool_sp_indx2))

            #B2 = tf.reduce_logsumexp(A2, [1, 2])
            B2 = tf.reduce_sum(tf.log(A2), axis=[1,2])
            # Create tensor containing products for each cell
            C2_1 = tf.stack([B2] * (h * w))
            C2_2 = tf.reshape(tf.transpose(C2_1), (c, h, w))
            C2_3 = tf.multiply(tf.to_float(bool_sp_indx2), C2_2)
            
            # Add to overall product
            prod_tensor_att += C1_3 + C2_3
            return l2+1, prod_tensor_att

        l2, prod_tensor_att = tf.while_loop(inner_cond, inner_while_body, [l2, prod_tensor_att], parallel_iterations=len(sp_indices), back_prop=False)
        return l1+1, prod_tensor_att

    l1 = 0
    cond = lambda l1, prod_tensor_att: tf.less(l1, len(sp_indices))
    l1, prod_tensor_att = tf.while_loop(cond, while_body, [l1, prod_tensor_att], parallel_iterations=len(sp_indices), back_prop=False)
    
    # Avoid division by zero from q_values
    bool_sum_zero = tf.equal(q_values, 0)
    bool_sum_one = tf.to_float(bool_sum_zero)
    q_values_modified = q_values + bool_sum_one

    #first_term = tf.divide(tf.to_float(prod_tensor_att), q_values_modified)
    first_term_att_unexp = tf.subtract(tf.to_float(prod_tensor_att), tf.log(q_values_modified + 0.00001))
    first_term_att = tf.exp(first_term_att_unexp)
    
    att_low_weights_duplicated = tf.transpose(tf.stack([attachment_low_weights] * (h * w)))
    first_term_resp = tf.multiply(att_low_weights_duplicated, tf.reshape(first_term_att, (c, -1)))
    first_term_resp_back = tf.reshape(first_term_resp, (c, h, w))
    attachment_update = first_term_resp_back + attachment_high_weights * (tf.ones(shape=(c, h, w)) - first_term_att)

    return attachment_update

def _compute_combined_update(q_values,low_weights,high_weights,sp_map,sp_indices, c, h, w ):

    # replicate the sp_map m times and have the shape of [rows,cols,m), where m in the number of labels
    extended_sp_map = tf.stack([sp_map] * c)

    # sp:
    prod_tensor_sp = tf.zeros(shape=(c, h, w))

    # cont:
    bool_max_label = tf.equal(q_values, tf.reduce_max(q_values, axis=0))
    q_val_sum_tensor = tf.zeros(shape=(c, h, w))
    prod_tensor_io = tf.zeros(shape=(c, h, w))

    # att:
    prod_tensor_att = tf.zeros(shape=(c, h, w))

    l1 = 0
    cond = lambda l1, prod_tensor_sp, prod_tensor_io, q_val_sum_tensor, prod_tensor_att: tf.less(l1, len(sp_indices))
    def while_body(l1, prod_tensor_sp, prod_tensor_io, q_val_sum_tensor, prod_tensor_att):
        index_l1 = tf.to_float(l1)
        # iterate over all superpixels, # Sample the center of the image
        #for sp_indx in sp_indices: #range(200,220):#random.sample(range(200, 400), 5):  # sampling superpixels, otherwise memory is overloaded
        #print(sp_indx)
        # This will put True where where sp index is sp_indx, False otherwise:
        cond_sp_indx = tf.equal(extended_sp_map, index_l1)

        q_val_for_clique = tf.multiply(tf.to_float(cond_sp_indx), q_values + 0.00001)
        # put 1 in q_vals if a pixel is not in sp_indx:
        q_val_for_clique_padded = q_val_for_clique + tf.to_float(tf.logical_not(cond_sp_indx))

        # compute the max label:
        maxlabel_q_val_for_sp = tf.reduce_max(q_val_for_clique, axis=0)

        # here we put q_val[r,c,l] = q_val[r,c,l'] where l' is the dominant label (only for pixels in l1)
        maxlabel_q_val_for_bd_duplicated = tf.stack([maxlabel_q_val_for_sp] * c)

        # here we compute: q_val(r,c,l) + q_val(r,c,l') where l' is the dominant label in the clique
        #A = q_val_for_clique_padded + maxlabel_q_val_for_bd_duplicated
        A_no_padding = q_val_for_clique + maxlabel_q_val_for_bd_duplicated

        # Subtract q_val(r,c,l') from indices where l = l'
        l_prime_equals_l = tf.multiply(tf.to_float(tf.logical_and(bool_max_label, cond_sp_indx)), q_values)
        #A = tf.subtract(A, l_prime_equals_l)
        A_no_padding_sub_l = tf.subtract(A_no_padding, l_prime_equals_l)

        q_val_sum_tensor += A_no_padding_sub_l  # A

        # ---- SP --------
        # compute the product for each label:
        #B_sp = tf.reduce_logsumexp(q_val_for_clique, [1, 2])
        B_sp = tf.reduce_sum(tf.log(q_val_for_clique_padded), axis=[1,2])
        
        # Create a tensor where each cell contains the product for its superpixel l1 and its label l:
        C_sp_1 = tf.stack([B_sp] * (h * w))
        C_sp_2 = tf.reshape(tf.transpose(C_sp_1), (c, h, w))
        C_sp_3 = tf.multiply(tf.to_float(cond_sp_indx), C_sp_2)  ## (using tf.tensordot rather than tf.multiply

        # add this to the overall product tensor; each cell contains the 'product' for its update rule:
        prod_tensor_sp += C_sp_3

        # ----- CONT -------
        # compute the product for each label:
        #B_cont = tf.reduce_logsumexp(A_no_padding_sub_l, [1, 2])
        B_cont = tf.reduce_sum(tf.log(A_no_padding_sub_l), axis=[1,2])
        
        # Create a tensor where each cell contains the product for its boundary clique_indx and its label l:
        C_cont_1 = tf.stack([B_cont] * (h * w))
        C_cont_2 = tf.reshape(tf.transpose(C_cont_1), (c, h, w))
        C_cont_3 = tf.multiply(tf.to_float(cond_sp_indx), C_cont_2)

        # add this to the overall product tensor; each cell contains the 'product' for its update rule:
        prod_tensor_io += tf.multiply(tf.to_float(cond_sp_indx), C_cont_3)

        l2 = 0
        inner_cond = lambda l2, prod_tensor_att: tf.less(l2, l1)
        def inner_while_body(l2, prod_tensor_att):
            # ---- ATT ---------
            # compute sp info for:
            #sp_indx2 = sp_indx + 1
            index_l2 = tf.to_float(l2)
            cond_sp_indx2 = tf.equal(extended_sp_map, index_l2)
            # Don't put 1 in q_values anymore if doesn't belong to this clique
            q_val_for_clique2 = tf.multiply(tf.to_float(cond_sp_indx2), q_values+ 0.00001) + tf.to_float(tf.logical_not(bool_sp_indx2))
            
            # Compute product for each cell in sp2:
            #B_sp2 = tf.reduce_logsumexp(q_val_for_clique2, [1, 2])
            B_sp2 = tf.reduce_sum(tf.log(q_val_for_clique2), axis=[1,2])
            # Create tensor containing products for each cell
            C_sp2_1 = tf.stack([B_sp2] * (h * w))
            C_sp2_2 = tf.reshape(tf.transpose(C_sp2_1), (c, h, w))
            C_sp2_3 = tf.multiply(tf.to_float(cond_sp_indx2), C_sp2_2)  ## (using tf.tensordot rather than tf.multiply
            
            # Add to overall product
            prod_tensor_att += C_sp_3 + C_sp2_3

            return l2+1, prod_tensor_att

        l2, prod_tensor_att = tf.while_loop(inner_cond, inner_while_body, [l2, prod_tensor_att], parallel_iterations=len(sp_indices), back_prop=False)

        return l1+1, prod_tensor_sp, prod_tensor_io, q_val_sum_tensor, prod_tensor_att

    l1, prod_tensor_sp, prod_tensor_io, q_val_sum_tensor, prod_tensor_att = tf.while_loop(cond, while_body, [l1, prod_tensor_sp, prod_tensor_io, q_val_sum_tensor, prod_tensor_att], parallel_iterations=len(sp_indices), back_prop=False)
            
    # modified q_values: (Avoid division by zero from q_values)
    bool_sum_zero = tf.equal(q_values, 0)
    bool_sum_one = tf.to_float(bool_sum_zero)
    q_values_modified = q_values + bool_sum_one

    # modified q_val_sum:
    # Add 1 to q_val_sum_tensor where it is 0
    bool_sum_zero = tf.equal(q_val_sum_tensor, 0)
    bool_sum_one = tf.to_float(bool_sum_zero)
    q_val_sum_tensor += bool_sum_one

    # compute update:
    # sp:
    # the actual product: we need to divide it by each index's q_val(r,c,l)
    #first_term = tf.divide(tf.to_float(prod_tensor_sp), q_values_modified)
    first_term_sp_unexp = tf.subtract(tf.to_float(prod_tensor_sp), tf.log(q_values_modified + 0.00001)) # might not be necessary to add epsilon here
    first_term_sp = tf.exp(first_term_sp_unexp)
    
    superpixel_low_weights_duplicated = tf.transpose(tf.stack([low_weights[0]] * (h * w)))
    #superpixel_low_weights_duplicated = tf.transpose(tf.tile(tf.reshape(low_weights[0], (1,))))
    first_term_resp = tf.multiply(superpixel_low_weights_duplicated, tf.reshape(first_term_sp, (c, -1)))
    first_term_resp_back = tf.reshape(first_term_resp, (c, h, w))
    superpixel_update = first_term_resp_back + high_weights[0] * (tf.ones(shape=(c, h, w)) - first_term_sp)
    # cont:
    # the actual product: we need to divide it by each index's q_val(r,c,l) + q_val(r,c,l')
    #first_term = tf.divide(tf.to_float(prod_tensor_io), q_val_sum_tensor)
    first_term_cont_unexp = tf.subtract(tf.to_float(prod_tensor_io), tf.log(q_val_sum_tensor + 0.00001))
    first_term_cont = tf.exp(first_term_cont_unexp)
    containment_low_weights_duplicated = tf.transpose(tf.stack([low_weights[1]] * (h * w)))
    first_term_resp = tf.multiply(containment_low_weights_duplicated, tf.reshape(first_term_cont, (c, -1)))
    first_term_resp_back = tf.reshape(first_term_resp, (c, h, w))
    containment_update = first_term_resp_back + high_weights[1] * (tf.ones(shape=(c, h, w)) - first_term_cont)
    # att:
    #first_term = tf.divide(tf.to_float(prod_tensor_att), q_values_modified)
    first_term_att_unexp = tf.subtract(tf.to_float(prod_tensor_att), tf.log(q_values_modified + 0.00001))
    first_term_att = tf.exp(first_term_att_unexp)
    
    att_low_weights_duplicated = tf.transpose(tf.stack([low_weights[2]] * (h * w)))
    first_term_resp = tf.multiply(att_low_weights_duplicated, tf.reshape(first_term_att, (c, -1)))
    first_term_resp_back = tf.reshape(first_term_resp, (c, h, w))
    attachment_update = first_term_resp_back + high_weights[2] * (tf.ones(shape=(c, h, w)) - first_term_att)

    return superpixel_update + containment_update + attachment_update

def _compute_superpixel_and_containment_update(q_values,low_weights,high_weights,sp_map,sp_indices, c, h, w ):

    # replicate the sp_map m times and have the shape of [rows,cols,m), where m in the number of labels
    #extended_sp_map = tf.stack([sp_map] * c)
    extended_sp_map = tf.tile(tf.reshape(sp_map, [1, h, w]), [c, 1, 1])
    # sp:
    prod_tensor_sp = tf.zeros(shape=(c, h, w))

    # cont:
    bool_max_label = tf.equal(q_values, tf.reduce_max(q_values, axis=0))
    q_val_sum_tensor = tf.zeros(shape=(c, h, w))
    prod_tensor_io = tf.zeros(shape=(c, h, w))
    
    def while_body(index, prod_tensor_sp, prod_tensor_io, q_val_sum_tensor):
        sp_indx = tf.to_float(index)
        # This will put True where where sp index is sp_indx, False otherwise:
        cond_sp_indx = tf.equal(extended_sp_map, sp_indx)

        q_val_for_clique = tf.multiply(tf.to_float(cond_sp_indx), q_values + 0.00001)
        # put 1 in q_vals if a pixel is not in sp_indx:
        q_val_for_clique_padded = q_val_for_clique + tf.to_float(tf.logical_not(cond_sp_indx))

        # compute the max label:
        maxlabel_q_val_for_sp = tf.reduce_max(q_val_for_clique, axis=0)

        # here we put q_val[r,c,l] = q_val[r,c,l'] where l' is the dominant label (only for pixels in clique_indx)
        maxlabel_q_val_for_bd_duplicated = tf.stack([maxlabel_q_val_for_sp] * c)

        # here we compute: q_val(r,c,l) + q_val(r,c,l') where l' is the dominant label in the clique
        #A = q_val_for_clique_padded + maxlabel_q_val_for_bd_duplicated
        A_no_padding = q_val_for_clique + maxlabel_q_val_for_bd_duplicated

        # Subtract q_val(r,c,l') from indices where l = l'
        l_prime_equals_l = tf.multiply(tf.to_float(tf.logical_and(bool_max_label, cond_sp_indx)), q_values)
        #A = tf.subtract(A, l_prime_equals_l)
        A_no_padding_sub_l = tf.subtract(A_no_padding, l_prime_equals_l)

        q_val_sum_tensor += A_no_padding_sub_l  # A

        # ---- SP --------
        # compute the product for each label:
        #B_sp = tf.reduce_logsumexp(q_val_for_clique, [1, 2])
        B_sp = tf.reduce_sum(tf.log(q_val_for_clique), axis=[1,2])

        # Create a tensor where each cell contains the product for its superpiel sp_indx and its label l:
        #C_sp1 = tf.stack([B_sp] * (h * w))
        C_sp1 = tf.tile(tf.reshape(B_sp, [1,6]), [h*w, 1])
        C_sp2 = tf.reshape(tf.transpose(C_sp1), (c, h, w))
        C_sp3 = tf.multiply(tf.to_float(cond_sp_indx), C_sp2)  ## (using tf.tensordot rather than tf.multiply

        # add this to the overall product tensor; each cell contains the 'product' for its update rule:
        prod_tensor_sp += C_sp3

        # ----- CONT -------
        # compute the product for each label:
        #B_cont = tf.reduce_logsumexp(A_no_padding_sub_l, [1, 2])
        B_cont = tf.reduce_sum(tf.log(A_no_padding_sub_l), axis=[1,2])
        # Create a tensor where each cell contains the product for its boundary clique_indx and its label l:
        #C_cont1 = tf.stack([B_cont] * (h * w))
        C_cont1 = tf.tile(tf.reshape(B_cont, [1,6]), [h*w, 1])
        C_cont2 = tf.reshape(tf.transpose(C_cont1), (c, h, w))
        C_cont3 = tf.multiply(tf.to_float(cond_sp_indx), C_cont2)

        # add this to the overall product tensor; each cell contains the 'product' for its update rule:
        prod_tensor_io += tf.multiply(tf.to_float(cond_sp_indx), C_cont3)

        return index+1, prod_tensor_sp, prod_tensor_io, q_val_sum_tensor

    i=0
    cond = lambda i, prod_tensor_sp, prod_tensor_io, q_val_sum_tensor: tf.less(i, len(sp_indices))
    i, prod_tensor_sp, prod_tensor_io, q_val_sum_tensor = tf.while_loop(cond, while_body, [i, prod_tensor_sp, prod_tensor_io, q_val_sum_tensor], parallel_iterations=len(sp_indices))

    # modified q_values: (Avoid division by zero from q_values)
    bool_sum_zero = tf.equal(q_values, 0)
    bool_sum_one = tf.to_float(bool_sum_zero)
    q_values_modified = q_values + bool_sum_one

    # modified q_val_sum:
    # Add 1 to q_val_sum_tensor where it is 0
    bool_sum_zero = tf.equal(q_val_sum_tensor, 0)
    bool_sum_one = tf.to_float(bool_sum_zero)
    q_val_sum_tensor += bool_sum_one

    # compute update:
    # sp:
    # the actual product: we need to divide it by each index's q_val(r,c,l)
    #first_term = tf.divide(tf.to_float(prod_tensor_sp), q_values_modified)
    first_term_sp_unexp = tf.subtract(tf.to_float(prod_tensor_sp), tf.log(q_values_modified + 0.00001))
    first_term_sp = tf.exp(first_term_sp_unexp)
    #superpixel_low_weights_duplicated = tf.transpose(tf.stack([low_weights[0]] * (h * w)))
    superpixel_low_weights_duplicated = tf.transpose(tf.tile(tf.reshape(low_weights[0], [1,6]), [h*w,1]))
    #superpixel_low_weights_duplicated = tf.transpose(tf.tile(tf.reshape(superpixel_low_weights, [1,6]), [h*w, 1]))
    first_term_resp = tf.multiply(superpixel_low_weights_duplicated, tf.reshape(first_term, (c, -1)))
    first_term_resp_back = tf.reshape(first_term_resp, (c, h, w))
    superpixel_update = first_term_resp_back + high_weights[0] * (tf.ones(shape=(c, h, w)) - first_term)
    # cont:
    # the actual product: we need to divide it by each index's q_val(r,c,l) + q_val(r,c,l')
    #first_term = tf.divide(tf.to_float(prod_tensor_io), q_val_sum_tensor)
    first_term_cont_unexp = tf.subtract(tf.to_float(prod_tensor_io), tf.log(q_val_sum_tensor + 0.00001))
    first_term_cont = tf.exp(first_term_cont_unexp)
    #containment_low_weights_duplicated = tf.transpose(tf.stack([low_weights[1]] * (h * w)))
    containment_low_weights_duplicated = tf.transpose(tf.tile(tf.reshape(low_weights[1], [1,6], [h*w,1])))
    first_term_resp = tf.multiply(containment_low_weights_duplicated, tf.reshape(first_term, (c, -1)))
    first_term_resp_back = tf.reshape(first_term_resp, (c, h, w))
    containment_update = first_term_resp_back + high_weights[1] * (tf.ones(shape=(c, h, w)) - first_term)

    return superpixel_update + containment_update

def _compute_superpixel_and_attachment_update(q_values, low_weights, high_weights, sp_map, sp_indices, c, h, w ):

    # replicate the sp_map m times and have the shape of [rows,cols,m), where m in the number of labels
    extended_sp_map = tf.stack([sp_map] * c)
    #extended_sp_map = tf.tile(tf.reshape(sp_map, [1,h,w]), [c,1,1])

    # sp:
    prod_tensor_sp = tf.zeros(shape=(c, h, w))

    # att:
    prod_tensor_att = tf.zeros(shape=(c, h, w))

    l1 = 0
    cond = lambda l1, prod_tensor_sp, prod_tensor_att: tf.less(l1, len(sp_indices))
    def while_body(l1, prod_tensor_sp, prod_tensor_att):
        index_l1 = tf.to_float(l1)
        # This will put True where where sp index is sp_indx, False otherwise:
        cond_sp_indx = tf.equal(extended_sp_map, index_l1)

        q_val_for_clique = tf.multiply(tf.to_float(cond_sp_indx), q_values + 0.00001)
        # put 1 in q_vals if a pixel is not in sp_indx:
        q_val_for_clique_padded = q_val_for_clique + tf.to_float(tf.logical_not(cond_sp_indx))

        # ---- SP --------
        # compute the product for each label:
        #B_sp = tf.reduce_logsumexp(q_val_for_clique, [1, 2])
        B_sp = tf.reduce_sum(tf.log(q_val_for_clique_padded), axis=[1,2])

        # Create a tensor where each cell contains the product for its superpiel sp_indx and its label l:
        C_sp1 = tf.stack([B_sp] * (h * w))
        #C_sp1 = tf.tile(tf.reshape(B_sp, [1, 6]), [h*w, 1])
        C_sp2 = tf.reshape(tf.transpose(C_sp1), (c, h, w))
        C_sp3 = tf.multiply(tf.to_float(cond_sp_indx), C_sp2)  ## (using tf.tensordot rather than tf.multiply

        # add this to the overall product tensor; each cell contains the 'product' for its update rule:
        prod_tensor_sp += C_sp3

        l2 = 0
        inner_cond = lambda l2, prod_tensor_att: tf.less(l2, l1)
        def inner_while_body(l2, prod_tensor_att):
            # ---- ATT ---------
            # compute sp info
            index_l2 = tf.to_float(l2)
            cond_sp_indx2 = tf.equal(extended_sp_map, index_l2)
            q_val_for_clique2 = tf.multiply(tf.to_float(cond_sp_indx2), q_values + 0.00001) + tf.to_float(tf.logical_not(bool_sp_indx2))
            
            # Compute product for each cell in sp2:
            #B_sp2 = tf.reduce_logsumexp(q_val_for_clique2, [1, 2])
            B_sp2 = tf.reduce_sum(tf.log(q_val_for_clique2), axis=[1,2])
            
            # Create tensor containing products for each cell
            C_sp2_1 = tf.stack([B_sp2] * (h * w))
            #C_sp2_1 = tf.tile(tf.reshape(B_sp2, [1,6]), [h*w, 1])
            C_sp2_2 = tf.reshape(tf.transpose(C_sp2_1), (c, h, w))
            C_sp2_3 = tf.multiply(tf.to_float(cond_sp_indx2), C_sp2_2)  ## (using tf.tensordot rather than tf.multiply
            
            # Add to overall product
            prod_tensor_att += C_sp3 + C_sp2_3

            return l2+1, prod_tensor_att

        l2, prod_tensor_att = tf.while_loop(inner_cond, inner_while_body, [l2, prod_tensor_att], parallel_iterations=len(sp_indices), back_prop=False)

        return l1+1, prod_tensor_sp, prod_tensor_att

    l1, prod_tensor_sp, prod_tensor_att = tf.while_loop(cond, while_body, [l1, prod_tensor_sp, prod_tensor_att], parallel_iterations=len(sp_indices), back_prop=False)
    
    # modified q_values: (Avoid division by zero from q_values)
    bool_sum_zero = tf.equal(q_values, 0)
    bool_sum_one = tf.to_float(bool_sum_zero)
    q_values_modified = q_values + bool_sum_one

    # compute update:
    # sp:
    # the actual product: we need to divide it by each index's q_val(r,c,l)
    #first_term = tf.divide(tf.to_float(prod_tensor_sp), q_values_modified)
    first_term_sp_unexp = tf.subtract(tf.to_float(prod_tensor_sp), tf.log(q_values_modified + 0.00001))
    first_term_sp = tf.exp(first_term_sp_unexp)
    superpixel_low_weights_duplicated = tf.transpose(tf.stack([low_weights[0]] * (h * w)))
    #superpixel_low_weights_duplicated = tf.transpose(tf.tile(tf.reshape(low_weights[0], [1,6]), [h*w, 1]))
    first_term_resp = tf.multiply(superpixel_low_weights_duplicated, tf.reshape(first_term_sp, (c, -1)))
    first_term_resp_back = tf.reshape(first_term_resp, (c, h, w))
    superpixel_update = first_term_resp_back + high_weights[0] * (tf.ones(shape=(c, h, w)) - first_term_sp)
    # att:
    #first_term = tf.divide(tf.to_float(prod_tensor_att), q_values_modified)
    first_term_att_unexp = tf.subtract(tf.to_float(prod_tensor_att), tf.log(q_values_modified + 0.00001))
    first_term_att = tf.exp(first_term_att_unexp)
    att_low_weights_duplicated = tf.transpose(tf.stack([low_weights[1]] * (h * w)))
    #att_low_weights_duplicated = tf.transpose(tf.tile(tf.reshape(low_weights[1], [1,6]), [h*w, 1]))
    first_term_resp = tf.multiply(att_low_weights_duplicated, tf.reshape(first_term_att, (c, -1)))
    first_term_resp_back = tf.reshape(first_term_resp, (c, h, w))
    attachment_update = first_term_resp_back + high_weights[1] * (tf.ones(shape=(c, h, w)) - first_term_att)

    return superpixel_update + attachment_update


class CrfRnnLayerSP(Layer):
    """ Implements the CRF-RNN layer described in:

    Conditional Random Fields as Recurrent Neural Networks,
    S. Zheng, S. Jayasumana, B. Romera-Paredes, V. Vineet, Z. Su, D. Du, C. Huang and P. Torr,
    ICCV 2015
    """

    def __init__(self, image_dims, num_classes,
                 theta_alpha, theta_beta, theta_gamma, batch_size,
                 num_iterations, **kwargs):
        self.image_dims = image_dims
        self.num_classes = num_classes
        self.theta_alpha = theta_alpha
        self.theta_beta = theta_beta
        self.theta_gamma = theta_gamma
        self.batch_size = batch_size
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
        superpixel_cliques = tf.transpose(inputs[2][0,:,:])

        c, h, w = self.num_classes, self.image_dims[0], self.image_dims[1]

        all_ones = np.ones((c, h, w), dtype=np.float32)

        # Prepare filter normalization coefficients
        spatial_norm_vals = custom_module.high_dim_filter(all_ones, rgb, bilateral=False,
                                                          theta_gamma=self.theta_gamma)
        bilateral_norm_vals = custom_module.high_dim_filter(all_ones, rgb, bilateral=True,
                                                            theta_alpha=self.theta_alpha,
                                                            theta_beta=self.theta_beta)
        q_values = unaries

        sp_indices = [range(500) for i in range(self.num_iterations)]
        #sp_indices=[range(1) for i in range(self.num_iterations)]
        #t2 = time.time()
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

            # compute superpixel potential update:
            #t0 = time.time()
            superpixel_update = _compute_superpixel_update(softmax_out, self.superpixel_low_weights, self.superpixel_high_weight, superpixel_cliques, sp_indices[i], c, h, w)

            #t1 = time.time()
            #print("time for sp update ", t1-t0)
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

        #t3 = time.time()
        #print("time for entire update ", t3-t2)
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
                 theta_alpha, theta_beta, theta_gamma, batch_size,
                 num_iterations, **kwargs):
        self.image_dims = image_dims
        self.num_classes = num_classes
        self.theta_alpha = theta_alpha
        self.theta_beta = theta_beta
        self.theta_gamma = theta_gamma
        self.batch_size = batch_size
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


        self.complex_rel_low_weights = self.add_weight(name='complex_rel_low_weights',
                                                       shape=([2, self.num_classes]),
                                                       initializer=_low_weight_initializer,
                                                       trainable=True)

        self.complex_rel_high_weights = self.add_weight(name='complex_rel_high_weight',
                                                        shape=(2), 
                                                        initializer=_sp_low_weight_initializer,
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

        sp_indices = [range(500) for i in range(self.num_iterations)]
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

            # compute containment potential update:
            tc0 = time.time()
            containment_update = _compute_superpixel_and_containment_update(softmax_out, self.complex_rel_low_weights, self.complex_rel_high_weights, superpixel_cliques, sp_indices[i], c, h, w)
            tc1 = time.time()
            print("time for containment update ", tc1 - tc0)

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

            q_values = unaries - pairwise - containment_update

        return tf.transpose(tf.reshape(q_values, (1, c, h, w)), perm=(0, 2, 3, 1))

    def compute_output_shape(self, input_shape):
        return input_shape


class CrfRnnLayerSPAT(Layer):
    """ Implements the CRF-RNN layer described in:

    Conditional Random Fields as Recurrent Neural Networks,
    S. Zheng, S. Jayasumana, B. Romera-Paredes, V. Vineet, Z. Su, D. Du, C. Huang and P. Torr,
    ICCV 2015
    """

    def __init__(self, image_dims, num_classes,
                 theta_alpha, theta_beta, theta_gamma, batch_size,
                 num_iterations, **kwargs):
        self.image_dims = image_dims
        self.num_classes = num_classes
        self.theta_alpha = theta_alpha
        self.theta_beta = theta_beta
        self.theta_gamma = theta_gamma
        self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.spatial_ker_weights = None
        self.bilateral_ker_weights = None
        self.compatibility_matrix = None
        super(CrfRnnLayerSPAT, self).__init__(**kwargs)

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


        self.complex_rel_low_weights = self.add_weight(name='complex_rel_low_weights',
                                                       shape=([2, self.num_classes]),
                                                       initializer=_low_weight_initializer,
                                                       trainable=True)

        self.complex_rel_high_weights = self.add_weight(name='complex_rel_high_weight',
                                                        shape=(2),
                                                        initializer=_sp_low_weight_initializer,
                                                        trainable=True)

        # Compatibility matrix
        self.compatibility_matrix = self.add_weight(name='compatibility_matrix',
                                                    shape=(self.num_classes, self.num_classes),
                                                    initializer=_potts_model_initializer,
                                                    trainable=True)

        super(CrfRnnLayerSPAT, self).build(input_shape)

    def call(self, inputs):
        '''
        unary_list, rgb_list, sp_list, q_values_list = [], [], [], []

        for j in range(self.batch_size):
            unary_list.append(tf.transpose(inputs[0][0, :, :, :], perm=(2, 0, 1)))
            rgb_list.append(tf.transpose(inputs[1][0, :, :, :], perm=(2, 0, 1)))
            sp_list.append(tf.transpose(inputs[2][0,:,:]))
        '''
        unaries = tf.transpose(inputs[0][0, :, :, :], perm=(2, 0, 1)) # the fcn_scores
        rgb = tf.transpose(inputs[1][0, :, :, :], perm=(2, 0, 1)) # the raw rgb
        superpixel_cliques = tf.transpose(inputs[2][0,:,:])  # perm=(0,1)
        '''
        unaries_tensor = tf.stack(unary_list)
        rgb_tensor = tf.stack(rgb_list)
        sp_tensor = tf.stack(sp_list)
        
        #for j in range(self.batch_size):
        k = 0
        cond = lambda k, q_values: k < self.batch_size
        def while_loop(index):
        
            unaries = unaries_tensor[j]
            rgb = rgb_tensor[j]
            superpixel_cliques = sp_tensor[j]
        '''    
        c, h, w = self.num_classes, self.image_dims[0], self.image_dims[1]
        
        all_ones = np.ones((c, h, w), dtype=np.float32)
        
        # Prepare filter normalization coefficients
        spatial_norm_vals = custom_module.high_dim_filter(all_ones, rgb, bilateral=False,
                                                          theta_gamma=self.theta_gamma)
        bilateral_norm_vals = custom_module.high_dim_filter(all_ones, rgb, bilateral=True,
                                                            theta_alpha=self.theta_alpha,
                                                            theta_beta=self.theta_beta)
        q_values = unaries
        
        sp_indices = [range(500) for i in range(self.num_iterations)]
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
            
            t0 = time.time()
            attachment_update = _compute_superpixel_and_attachment_update(softmax_out, self.complex_rel_low_weights, self.complex_rel_high_weights, superpixel_cliques, sp_indices[i], c, h, w)
            t1 = time.time()
            print("time ", t1-t0)
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
            
            q_values = unaries - pairwise - attachment_update
            
            #q_values_list.append(q_values)

        #l = tf.stack(q_values_list)
        #return tf.transpose(tf.reshape(l, (self.batch_size, self.num_classes, self.image_dims[0], self.image_dims[1])), perm=(0,2,3,1))
        return tf.transpose(tf.reshape(q_values, (1, c, h, w)), perm=(0, 2, 3, 1))

    def compute_output_shape(self, input_shape):
        return input_shape


class CrfRnnLayerAll(Layer):
    """ Implements the CRF-RNN layer described in:

    Conditional Random Fields as Recurrent Neural Networks,
    S. Zheng, S. Jayasumana, B. Romera-Paredes, V. Vineet, Z. Su, D. Du, C. Huang and P. Torr,
    ICCV 2015
    """

    def __init__(self, image_dims, num_classes,
                 theta_alpha, theta_beta, theta_gamma, batch_size,
                 num_iterations, **kwargs):
        self.image_dims = image_dims
        self.num_classes = num_classes
        self.theta_alpha = theta_alpha
        self.theta_beta = theta_beta
        self.theta_gamma = theta_gamma
        self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.spatial_ker_weights = None
        self.bilateral_ker_weights = None
        self.compatibility_matrix = None
        super(CrfRnnLayerAll, self).__init__(**kwargs)

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

        #'''
        self.complex_rel_low_weights = self.add_weight(name='complex_rel_low_weights',
                                                       shape=([3,self.num_classes]),
                                                       initializer=_low_weight_initializer,
                                                       trainable=True)

        self.complex_rel_high_weights = self.add_weight(name='complex_rel_high_weight',
                                                       shape=(3),
                                                       initializer=_sp_low_weight_initializer,
                                                        trainable=True)
        #'''
        # Compatibility matrix
        self.compatibility_matrix = self.add_weight(name='compatibility_matrix',
                                                    shape=(self.num_classes, self.num_classes),
                                                    initializer=_potts_model_initializer,
                                                    trainable=True)

        super(CrfRnnLayerAll, self).build(input_shape)

    def call(self, inputs):
        #'''
        # Python lists for inputs
        unary_list, rgb_list, sp_clique_list, q_values_list = [], [], [], []

        for j in range(self.batch_size):
            unary_list.append(tf.transpose(inputs[0][j, :, :, :], perm=(2,0,1)))
            rgb_list.append(tf.transpose(inputs[1][0, :, :, :], perm=(2, 0, 1)))
            sp_clique_list.append(tf.transpose(inputs[2][0,:,:]))

        unaries_tensor = tf.stack(unary_list)
        rgb_tensor = tf.stack(rgb_list)
        sp_tensor = tf.stack(sp_clique_list)

        for j in range(self.batch_size):
            unaries = unaries_tensor[j]
            rgb = rgb_tensor[j]
            superpixel_cliques = sp_tensor[j]
            #'''
            #unaries = tf.transpose(inputs[0][0, :, :, :], perm=(2, 0, 1)) # the fcn_scores
            #rgb = tf.transpose(inputs[1][0, :, :, :], perm=(2, 0, 1)) # the raw rgb
            #superpixel_cliques = tf.transpose(inputs[2][0,:,:])  # perm=(0,1)
            
            c, h, w = self.num_classes, self.image_dims[0], self.image_dims[1]
            
            all_ones = np.ones((c, h, w), dtype=np.float32)
            
            # Prepare filter normalization coefficients
            spatial_norm_vals = custom_module.high_dim_filter(all_ones, rgb, bilateral=False,
                                                              theta_gamma=self.theta_gamma)
            bilateral_norm_vals = custom_module.high_dim_filter(all_ones, rgb, bilateral=True,
                                                                theta_alpha=self.theta_alpha,
                                                                theta_beta=self.theta_beta)
            q_values = unaries
            #num_of_sp_samples = 1
            #sp_indices = [random.sample(range(200,400), num_of_sp_samples) for i in range(self.num_iterations)]
            sp_indices = [range(500) for i in range(self.num_iterations)]
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
                
                # compute superpixel potential update:
                #t0 = time.time()
                #superpixel_update = _compute_superpixel_update(softmax_out, self.superpixel_low_weights, self.superpixel_high_weight, superpixel_cliques, sp_indices[i], c, h, w)
                #t1 = time.time()
                #print("time ", t1-t0)
                # compute containment potential update:
                #containment_update = _compute_containment_update(softmax_out, self.containment_low_weights, self.containment_high_weight, superpixel_cliques, sp_indices[i], c, h, w)
                
                # compute attachment potential update:
                #attachment_update = _compute_attachment_update(softmax_out, self.attachment_low_weights, self.attachment_high_weight, superpixel_cliques, sp_indices[i], c, h, w)
                t0 = time.time()
                complex_relations_update = _compute_combined_update(q_values, self.complex_rel_low_weights, self.complex_rel_high_weights, superpixel_cliques, sp_indices[i], c, h, w)
                t1 = time.time()
                print("time ", t1-t0)
                
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
                
                #q_values = unaries - pairwise - superpixel_update
                q_values = unaries - pairwise - complex_relations_update
            q_values_list.append(q_values)

        l = tf.stack(q_values_list)
        return tf.transpose(tf.reshape(l, (self.batch_size, self.num_classes, self.image_dims[0], self.image_dims[1])), perm=(0, 2, 3, 1))

    def compute_output_shape(self, input_shape):
        return input_shape

# sp tem
import tensorflow as tf
import time
# may come in useful: tf.verify_tensor_all_finite
import numpy as np

s = tf.InteractiveSession()
time_in = time.time()
nb_classes = 3
rows, cols = 4, 5
correct_labeling = tf.constant([[0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [1, 1, 2, 2, 2],
                                [1, 1, 2, 2, 2]])
q_vals = tf.constant([[[0.9, 0.9, 0.9, 0.9, 0.9],
                       [0.9, 0.9, 0.9, 0.9, 0.9],
                       [0.01, 0.01, 0.01, 0.01, 0.01],
                       [0.01, 0.01, 0.01, 0.01, 0.01]],
                      [[0.01, 0.01, 0.01, 0.01, 0.01],
                       [0.01, 0.01, 0.01, 0.01, 0.01],
                       [0.9, 0.9, 0.8, 0.7, 0.6],
                       [0.9, 0.9, 0.81, 0.5, 0.4]],
                      [[0.01, 0.01, 0.01, 0.01, 0.01],
                       [0.01, 0.01, 0.01, 0.01, 0.01],
                       [0.4, 0.5, 0.9, 0.9, 0.9],
                       [0.7, 0.8, 0.8, 0.9, 0.9]]])

q_vals_arr = [[[0.9, 0.9, 0.9, 0.9, 0.9],
                       [0.9, 0.9, 0.9, 0.9, 0.9],
                       [0.01, 0.01, 0.01, 0.01, 0.01],
                       [0.01, 0.01, 0.01, 0.01, 0.01]],
                      [[0.01, 0.01, 0.01, 0.01, 0.01],
                       [0.01, 0.01, 0.01, 0.01, 0.01],
                       [0.9, 0.9, 0.8, 0.7, 0.6],
                       [0.9, 0.9, 0.81, 0.5, 0.4]],
                      [[0.01, 0.01, 0.01, 0.01, 0.01],
                       [0.01, 0.01, 0.01, 0.01, 0.01],
                       [0.4, 0.5, 0.9, 0.9, 0.9],
                       [0.7, 0.8, 0.8, 0.9, 0.9]]]

sp_map = tf.constant([[1,1,1,2,2],
                      [1,1,1,2,2],
                      [3,3,4,4,5],
                      [3,3,4,4,5]])

# replicate the sp_map m times and have the shape of [rows,cols,m], where m in the number of labels
extended_sp_map = tf.stack([sp_map] * nb_classes)

# split sp_map into [rows, cols, #sp] so that each layer has 1 at the the indices corresponding to that sp layer
flat = tf.reshape(sp_map, [-1])
y, index = tf.unique(flat)
length = s.run(tf.size(y)) # length = # of cliques
values = [tf.cast(tf.equal(sp_map,i), tf.float32) for i in range(1,length+1)]
split_sp_map = tf.stack(values)
bool_sp_map = tf.stack([tf.equal(sp_map,i) for i in range(1,length+1)])

# This will put True where the max prob label, False otherwise
cond_max_label = tf.equal(q_vals, tf.reduce_max(q_vals,axis=0))
max_label = tf.cast(cond_max_label, tf.float32)
not_max_label = -1*tf.subtract(max_label, 1)

# These would be the learned parameters:
w_low = tf.constant(0.1)
w_high = tf.constant(0.9)

sp_out = tf.get_variable("sp_out", [nb_classes, rows, cols], dtype=tf.float32, initializer=tf.zeros_initializer)
s.run(tf.global_variables_initializer())
max_q_val = tf.multiply(max_label, q_vals)
not_max_q_val = tf.multiply(not_max_label, q_vals)

for l in range(nb_classes):
    for i in range(rows):
        for j in range(cols):
            # clique index is value of (i,j) in sp_map
            clique_index = sp_map[i][j]

            product_matrix = tf.multiply(split_sp_map[clique_index-1], q_vals[l])
            val = s.run(product_matrix[i][j])
            flattened = tf.reshape(product_matrix, [-1])
            # Want the product of all nonzero elements in
            zero = tf.constant(0, dtype=tf.float32)
            where_nonzero = tf.not_equal(flattened, zero)
            reduced = tf.boolean_mask(flattened, where_nonzero)
            product = tf.reduce_prod(reduced)
            # Must divide product by value at q_i = l
            product = tf.divide(product, val)
            op = sp_out[l,i,j].assign(sp_out[l,i,j]+product)
            s.run(op)

print(s.run(sp_out))

time_out = time.time()
print('time passed.. %s sec' % (time_out - time_in))

'''
for sp_indx in range(1,6):

    # This will put True where sp index is sp_indx, False otherwise:
    cond_sp_indx = tf.equal(extended_sp_map,sp_indx)

    # This is tensor T, where the dominant label for sp_indx superpixel is:
    T = tf.logical_and(cond_max_label,cond_sp_indx)

    #The potental added to all pixels in sp_indx:
    #sp_out += w_low * tf.multiply(tf.to_float(T),q_vals) + w_high * tf.multiply(tf.to_float(tf.logical_not(T)),q_vals)
    sp_out += w_low * tf.to_float(T) + w_high * tf.to_float(tf.logical_not(T))

    # show
    #print(s.run(T))
    #print(s.run(sp_out))

# sp_out = tf.reduce_sum(sp_out)
# print(s.run(sp_out))
'''

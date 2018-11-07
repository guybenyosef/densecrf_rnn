# sp tem
import tensorflow as tf
<<<<<<< HEAD
import pdb

ses = tf.InteractiveSession()
=======
# may come in useful: tf.verify_tensor_all_finite
import numpy as np

s = tf.InteractiveSession()
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
>>>>>>> f34854ee10d4eb774d9faa0c7bed2e4f973902f9

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
<<<<<<< HEAD

print(ses.run(q_vals))
=======
max_label = tf.cast(cond_max_label, tf.float32)
not_max_label = -1*tf.subtract(max_label, 1)
>>>>>>> f34854ee10d4eb774d9faa0c7bed2e4f973902f9

# These would be the learned parameters:
w_low = tf.constant(0.1)
#w_low_m = tf.constant([0.11,0.10,0.09])
w_low_m = tf.constant([[0.11,0.,0.],
                       [0., 0.10, 0.],
                       [0., 0., 0.09]])
w_high = tf.constant(0.9)

<<<<<<< HEAD
#sp_indx = 4
sp_out = tf.zeros(q_vals.shape)
sp_cond = tf.constant(False, shape=q_vals.shape)
for sp_indx in range(1,6):

    #print(sp_indx)
=======
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

'''
for sp_indx in range(1,6):

>>>>>>> f34854ee10d4eb774d9faa0c7bed2e4f973902f9
    # This will put True where sp index is sp_indx, False otherwise:
    cond_sp_indx = tf.equal(extended_sp_map,sp_indx)

    # This is tensor T, where the dominant label for sp_indx superpixel is:
    T = tf.logical_and(cond_max_label,cond_sp_indx)

    #pdb.set_trace()
    sp_cond = tf.logical_or(sp_cond,T)
    #The potental added to all pixels in sp_indx:
    #sp_out += w_low * tf.multiply(tf.to_float(T),q_vals) + w_high * tf.multiply(tf.to_float(tf.logical_not(T)),q_vals)
    #sp_out += w_low * tf.to_float(T) + w_high * tf.to_float(tf.logical_not(T))

    # show
<<<<<<< HEAD
    #print(ses.run(T))
=======
    #print(s.run(T))
    #print(s.run(sp_out))
>>>>>>> f34854ee10d4eb774d9faa0c7bed2e4f973902f9

print(ses.run(sp_cond))
first_term = tf.multiply(tf.to_float(sp_cond),q_vals)
first_term_resp = tf.matmul(w_low_m,tf.reshape(first_term, (nb_classes,-1)))
first_term_resp_back = tf.reshape(first_term_resp, (nb_classes, 4, 5))

sp_out =  first_term_resp_back + w_high * tf.multiply(tf.to_float(tf.logical_not(sp_cond)),q_vals)
# sp_out = tf.reduce_sum(sp_out)
<<<<<<< HEAD
print(ses.run(sp_out))
=======
# print(s.run(sp_out))
'''
>>>>>>> f34854ee10d4eb774d9faa0c7bed2e4f973902f9

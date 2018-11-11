# sp tem
import tensorflow as tf
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

# In this case the sp boundary is the same as the sp_map (treated exactly the same way as sp_map)
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
print(s.run(max_q_val))
not_max_q_val = tf.multiply(not_max_label, q_vals)
'''
# Superpixel Update with tensors
# This will put True where the max prob label, False otherwise:
cond_max_label = tf.equal(q_vals, tf.reduce_max(q_vals,axis=0))

# These would be the learned parameters:
w_low_m = tf.constant([[0.1,0.,0.],
                       [0., 0.1, 0.],
                       [0., 0., 0.1]])

prod_tensor = tf.zeros(shape=q_vals.shape)

for sp_indx in range(1,length+1):
    # This will put True where sp index is sp_indx, False otherwise:
    cond_sp_indx = tf.equal(extended_sp_map,sp_indx)
    # put 1 in q_vqls if not belongs to sp_indx:
    A = tf.multiply(tf.to_float(cond_sp_indx), q_vals) + tf.to_float(tf.logical_not(cond_sp_indx))
    # compute the product for each label:
    B = tf.reduce_prod(A, [1, 2])
    # Create a tensor where each cell contains the product for its superpiel sp_indx and its label l:
    C = tf.stack([B]*(rows*cols))
    C = tf.reshape(tf.transpose(C), (nb_classes, rows, cols))
    C = tf.multiply(tf.to_float(cond_sp_indx), C)

    # add this to the overall product tensor; each cell contains the 'product' for its update rule:
    prod_tensor += tf.multiply(tf.to_float(cond_sp_indx), C)

# the actual product: we need to divide it by the current q_vals
first_term = tf.divide(tf.to_float(prod_tensor),q_vals)
first_term_resp = tf.matmul(w_low_m,tf.reshape(first_term, (nb_classes,-1)))
first_term_resp_back = tf.reshape(first_term_resp, (nb_classes, 4, 5))
sp_out =  first_term_resp_back + w_high * (tf.ones(shape=first_term.shape) - first_term)
print("Using tensors")
print(s.run(sp_out))
'''
#'''
# Containment Update with tensors
io_update = tf.get_variable("sp_out", [nb_classes, rows, cols], dtype=tf.float32, initializer=tf.zeros_initializer)
s.run(tf.global_variables_initializer())
l_prime = 1

#for sp_indx in range(1,length+1):
    # Get q_val matrix with Q(l')
    
for l in range(nb_classes):
    for i in range(rows):
        for j in range(cols):
            # clique index is value of (i,j) in sp_map
            clique_index = sp_map[i][j]
            product_matrix = tf.multiply(split_sp_map[clique_index-1], q_vals[l])
            val = s.run(product_matrix[i][j])
            l_matrix = tf.multiply(split_sp_map[clique_index-1], q_vals[l_prime])
            flattened = tf.reshape(product_matrix, [-1])
            flattened_l = tf.reshape(product_matrix, [-1])
            flattened = tf.add(flattened, flattened_l)
            # Want the product of all nonzero elements in flattened
            zero = tf.constant(0, dtype=tf.float32)
            where_nonzero = tf.not_equal(flattened, zero)
            reduced = tf.boolean_mask(flattened, where_nonzero)
            product = tf.reduce_prod(reduced)
            # Must divide product by value at q_i = l
            product = tf.divide(product, val)
            op = sp_out[l,i,j].assign(sp_out[l,i,j]+product)
            s.run(op)
print(s.run(sp_out))
#'''
'''
# Superpixel Update with for loops
for l in range(nb_classes):
    for i in range(rows):
        for j in range(cols):
            # clique index is value of (i,j) in sp_map
            clique_index = sp_map[i][j]
            product_matrix = tf.multiply(split_sp_map[clique_index-1], q_vals[l])
            val = s.run(product_matrix[i][j])
            flattened = tf.reshape(product_matrix, [-1])
            # Want the product of all nonzero elements in flattened
            zero = tf.constant(0, dtype=tf.float32)
            where_nonzero = tf.not_equal(flattened, zero)
            reduced = tf.boolean_mask(flattened, where_nonzero)
            product = tf.reduce_prod(reduced)
            # Must divide product by value at q_i = l
            product = tf.divide(product, val)
            op = sp_out[l,i,j].assign(sp_out[l,i,j]+product)
            s.run(op)
print("Using for loops")
print(s.run(sp_out))
'''

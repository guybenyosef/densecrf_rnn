# Containment tem
import tensorflow as tf
import pdb

s = tf.InteractiveSession()
nb_classes = 3
rows,cols = 4,5
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
bd_map = tf.constant([[1,1,1,2,2],
                      [1,1,1,2,2],
                      [3,3,4,4,5],
                      [3,3,4,4,5]])


# replicate  m times and have the shape of [rows,cols,l] where l is the number of labels
extended_bd_map = tf.stack([bd_map] * nb_classes)
# Get number of cliques
flat = tf.reshape(bd_map, [-1])
y, index = tf.unique(flat)
num_cliques = s.run(tf.size(y))

# This will put True where the max prob label, False otherwise:
bool_max_label = tf.equal(q_vals, tf.reduce_max(q_vals,axis=0))

# These would be the learned parameters:
w_low = tf.constant(0.1)
w_low_m = tf.constant([[0.11,0.,0.],
                       [0., 0.10, 0.],
                       [0., 0., 0.09]])
w_low_m_1d = tf.constant([0.11,0.10,0.9])
w_low_m_1d_duplicated = tf.stack([w_low_m_1d]*(rows*cols))
w_high = tf.constant(0.9)

prod_tensor = tf.zeros(shape=q_vals.shape)
q_val_sum_tensor = tf.zeros(shape=q_vals.shape)
for clique_indx in range(1,num_cliques+1):
    # This will put True where bd index is clique_indx, False otherwise:
    bool_bd_indx = tf.equal(extended_bd_map,clique_indx)
    q_val_for_clique = tf.multiply(tf.to_float(bool_bd_indx), q_vals)

    # put 1 in q_vals if a pixel is not in sp_indx:
    q_val_for_clique_padded = q_val_for_clique  + tf.to_float(tf.logical_not(bool_bd_indx))
    maxlabel_q_val_for_bd = tf.reduce_max(q_val_for_clique,axis=0)

    # here we put q_val[r,c,l] = q_val[r,c,l'] where l' is the dominant label (only for pixels in clique_indx)
    maxlabel_q_val_for_bd_duplicated = tf.stack([maxlabel_q_val_for_bd] * nb_classes)

    # here we compute: q_val(r,c,l) + q_val(r,c,l') where l' is the dominant label in the clique
    A = q_val_for_clique_padded + maxlabel_q_val_for_bd_duplicated
    A_no_padding = q_val_for_clique + maxlabel_q_val_for_bd_duplicated

    # Subtract q_val(r,c,l') from indices where l = l'
    l_prime_equals_l = tf.multiply(tf.to_float(tf.logical_and(bool_max_label, bool_bd_indx)), q_vals)
    A = tf.subtract(A, l_prime_equals_l)
    A_no_padding = tf.subtract(A_no_padding, l_prime_equals_l)
    q_val_sum_tensor+=A_no_padding

    # compute the product for each label:
    B = tf.reduce_prod(A, [1, 2])
    # Create a tensor where each cell contains the product for its boundary clique_indx and its label l:
    C = tf.stack([B]*(rows*cols))
    C = tf.reshape(tf.transpose(C), (nb_classes, rows, cols))
    C = tf.multiply(tf.to_float(bool_bd_indx), C)
    # add this to the overall product tensor; each cell contains the 'product' for its update rule:
    prod_tensor += tf.multiply(tf.to_float(bool_bd_indx), C)

# the actual product: we need to divide it by each index's q_val(r,c,l) + q_val(r,c,l')
first_term = tf.divide(tf.to_float(prod_tensor),q_val_sum_tensor)

# multiply by weights
first_term_resp = tf.multiply(tf.transpose(w_low_m_1d_duplicated),tf.reshape(first_term, (nb_classes,-1)))

first_term_resp_back = tf.reshape(first_term_resp, (nb_classes, rows, cols))

containment_out = first_term_resp_back + w_high * (tf.ones(shape=first_term_resp_back.shape) - first_term_resp_back)

print(s.run(containment_out))

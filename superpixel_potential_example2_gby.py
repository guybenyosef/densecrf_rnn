
# sp tem
import tensorflow as tf
import pdb

ses = tf.InteractiveSession()

sp_map = tf.constant([[1,1,1,2,2],
                      [1,1,1,2,2],
                      [3,3,4,4,5],
                      [3,3,4,4,5]])

nb_classes = 3
rows,cols = 4,5

# replicate the sp_map m times and have the shape of [rows,cols,m), where m in the number of labels
extended_sp_map = tf.stack([sp_map] * nb_classes)

q_vals = tf.random_uniform(shape=[nb_classes,rows,cols])

# This will put True where the max prob label, False otherwise:
cond_max_label = tf.equal(q_vals, tf.reduce_max(q_vals,axis=0))

# These would be the learned parameters:
w_low = tf.constant(0.1)
w_low_m = tf.constant([[0.11,0.,0.],
                       [0., 0.10, 0.],
                       [0., 0., 0.09]])

w_low_m_1d = tf.constant([0.11,0.10,0.09])
w_low_m_1d_duplicated = tf.stack([w_low_m_1d]*(rows*cols))

w_high = tf.constant(0.9)

#sp_indx = 4
prod_tensor = tf.zeros(shape=q_vals.shape)

for sp_indx in range(1,6):

    #print(sp_indx)
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

    # This is tensor T, where the dominant label for sp_indx superpixel is:
#    T = tf.logical_and(cond_max_label,cond_sp_indx)

#    pdb.set_trace()
#    sp_cond = tf.logical_or(sp_cond,T)
    #The potental added to all pixels in sp_indx:
    #sp_out += w_low * tf.multiply(tf.to_float(T),q_vals) + w_high * tf.multiply(tf.to_float(tf.logical_not(T)),q_vals)
    #sp_out += w_low * tf.to_float(T) + w_high * tf.to_float(tf.logical_not(T))

    # show
    #print(ses.run(T))

#print(ses.run(prod_tensor))

# the actual product: we need to divide it by the current q_vals
first_term = tf.divide(tf.to_float(prod_tensor),q_vals)
#print(ses.run(first_term))

# multiply by weights:  (not sure if we need w_high)
#first_term_resp = tf.matmul(w_low_m,tf.reshape(first_term, (nb_classes,-1)))
first_term_resp = tf.multiply(tf.transpose(w_low_m_1d_duplicated),tf.reshape(first_term, (nb_classes,-1)))
#print(ses.run(first_term_resp))

first_term_resp_back = tf.reshape(first_term_resp, (nb_classes, rows, cols))
#print(ses.run(first_term_resp_back))

sp_out = first_term_resp_back + w_high * (tf.ones(shape=first_term_resp_back.shape) - first_term)

print(ses.run(sp_out))

# we then need to add sp_out to q_vals

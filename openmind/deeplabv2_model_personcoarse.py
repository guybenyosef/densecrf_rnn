from datetime import datetime
import os
import sys
import time
import numpy as np
import tensorflow as tf
from PIL import Image
from scipy.misc import imsave

from network_personcoarse import Deeplab_v2
from utils import ImageReader, decode_labels, inv_preprocess, prepare_label, write_log, read_labeled_image_list
from utils.get_devices import get_available_gpus, average_gradients
from src.weighted_categorical_crossentropy import weighted_loss

import tensorflow.contrib.slim as slim

"""
This script trains or evaluates the model on augmented PASCAL VOC 2012 dataset.
The training set contains 10581 training images.
The validation set contains 1449 validation images.

Training:
'poly' learning rate
different learning rates for different layers
"""



IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

class Model_msc(object):

    def __init__(self, sess, conf):
        self.sess = sess
        self.conf = conf

    # train
    def train(self):
        self.train_setup()
        gpu_list = get_available_gpus()

        self.sess.run(tf.global_variables_initializer())
        #'''
        # Load the pre-trained model if provided
        for i in range(len(gpu_list)):
            with tf.device(gpu_list[i]):
                if self.conf.pretrain_file is not None:
                    #self.load(self.loaders[0], self.conf.pretrain_file)
                    #checkpointfile = tf.train.latest_checkpoint("./../Deeplab-v2--ResNet-101--Tensorflow/model_personfine/")
                    #checkpointfile = "./../Deeplab-v2--ResNet-101--Tensorflow/model_personcoarse/model.ckpt-16000"
                    #self.load(self.loaders[0], checkpointfile)
                    checkpointfile = './model_personcoarse_crf/model.ckpt-500'
                    self.load(tf.train.Saver(var_list=tf.global_variables()), checkpointfile)
        #'''
        #checkpointfile = tf.train.latest_checkpoint("./model_multigpu_bs10/")
        #self.load(self.loaders[0], checkpointfile)

        # Get checkpoint right after loading
        #self.save(self.saver, 0)
        #print("save first time")

        # Start queue threads.
        threads = tf.train.start_queue_runners(coord=self.coord, sess=self.sess)
        # Train!
        for step in range(self.conf.num_steps+1):
            start_time = time.time()
            feed_dict = { self.curr_step : step + 500}
            loss_value = 0
            # Clear the accumulated gradients.
            self.sess.run(self.zero_op, feed_dict=feed_dict)
            # Accumulate gradients.
            for i in range(self.conf.grad_update_every):
                _, l_val = self.sess.run([self.accum_grads_op, self.reduced_loss], feed_dict=feed_dict)
                loss_value += l_val

            # Normalise the loss.
            loss_value /= self.conf.grad_update_every

            # Apply gradients.
            if step % self.conf.save_interval == 0:
                batch_list, images, labels, summary, _ = self.sess.run(
                    [self.im_list,
                     self.image_batch,
                     self.label_batch,
                     self.total_summary,
                     self.train_op],
                    feed_dict=feed_dict)

                '''
                # debug
                print(np.array(batch_list).shape)
                for (i,j) in [(i,j) for i in range(2) for j in range(2)]:
                    imsave('tmp/'+'step_'+str(step)+'_gpu_'+str(i)+'_'+str(j)+'.png', batch_list[i][j])
                    
                imsave('tmp/'+str(step)+'_0.png', images[0])
                imsave('tmp/'+str(step)+'_1.png', images[1])
                '''
                
                self.summary_writer.add_summary(summary, step)
                self.save(self.saver, step)
            else:
                self.sess.run(self.train_op, feed_dict=feed_dict)
                
            duration = time.time() - start_time
            print('step {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))
            write_log('{:d}, {:.3f}'.format(step, loss_value), self.conf.logfile)
            
        # finish
        self.coord.request_stop()
        self.coord.join(threads)

    # evaluate
    def test(self):
        self.test_setup()
      
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        # load checkpoint
        #checkpointfile = self.conf.modeldir+ '/model.ckpt-' + str(self.conf.valid_step)
        #checkpointfile = 'deeplab_resnet_init.ckpt'
        checkpointfile = tf.train.latest_checkpoint("./model_personcoarse_crf/")
        #checkpointfile = './model_crf_test'+ '/model.ckpt-' + '0'
        self.load(self.loader, checkpointfile)

        # Start queue threads.
        threads = tf.train.start_queue_runners(coord=self.coord, sess=self.sess)

        # Test!
        confusion_matrix = np.zeros((self.conf.num_classes, self.conf.num_classes), dtype=np.int)
        for step in range(self.conf.valid_num_steps):
            preds, _, _, c_matrix = self.sess.run([self.pred, self.accu_update_op, self.mIou_update_op, self.confusion_matrix])
            confusion_matrix += c_matrix
            if step % 100 == 0:
                print('step {:d}'.format(step))
        print('Pixel Accuracy: {:.3f}'.format(self.accu.eval(session=self.sess)))
        print('Mean IoU: {:.3f}'.format(self.mIoU.eval(session=self.sess)))
        self.compute_IoU_per_class(confusion_matrix)

        # finish
        self.coord.request_stop()
        self.coord.join(threads)

    # prediction
    def predict(self):
        self.predict_setup()

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        # load checkpoint
        #checkpointfile = self.conf.modeldir+ '/model.ckpt-' + str(self.conf.valid_step)
        #checkpointfile = 'deeplab_resnet_init.ckpt'
        checkpointfile = tf.train.latest_checkpoint("./model_multigpu_bs10/")
        #checkpointfile = './model_crf_test0'+ '/model.ckpt-' + '0'
        self.load(self.loader, checkpointfile)

        # Start queue threads.
        threads = tf.train.start_queue_runners(coord=self.coord, sess=self.sess)

        # img_name_list
        image_list, _ = read_labeled_image_list('', self.conf.test_data_list)

        # Predict!
        for step in range(self.conf.test_num_steps):
            preds = self.sess.run(self.pred)
            resized_dec = self.sess.run(self.resized_decoder100)
            img_name = image_list[step].split('/')[2].split('.')[0]
            # Save raw predictions, i.e. each pixel is an integer between [0,20].
            im = Image.fromarray(preds[0,:,:,0], mode='L')
            filename = '/%s_mask.png' % (img_name)
            im.save(self.conf.out_dir + '/prediction' + filename)
            #resized = Image.fromarray(resized_dec[0], mode='RGB')
            #fn = '/%s_resized_dec.png' % (img_name)
            #resized.save(self.conf.out_dir + '/resized_decoding' + fn)

            # Save predictions for visualization.
            # See utils/label_utils.py for color setting
            # Need to be modified based on datasets.
            if self.conf.visual:
                msk = decode_labels(preds, num_classes=self.conf.num_classes)
                im = Image.fromarray(msk[0], mode='RGB')
                filename = '/%s_mask_visual.png' % (img_name)
                im.save(self.conf.out_dir + '/visual_prediction' + filename)

            if step % 100 == 0:
                print('step {:d}'.format(step))

        print('The output files has been saved to {}'.format(self.conf.out_dir))

        # finish
        self.coord.request_stop()
        self.coord.join(threads)
        
    def train_setup(self):
        tf.set_random_seed(self.conf.random_seed)
        
        # Create queue coordinator.
        self.coord = tf.train.Coordinator()

        # Input size
        h, w = (self.conf.input_height, self.conf.input_width)
        input_size = (h, w)

        # Devices
        gpu_list = get_available_gpus()
        zip_encoder, zip_decoder_b, zip_decoder_w, zip_crf = [], [], [], []
        previous_crf_names = []
        restore_vars = []
        self.loaders = []

        self.im_list = []
        
        for i in range(len(gpu_list)):
            with tf.device(gpu_list[i]):
                # Load reader
                with tf.name_scope("create_inputs"):
                    reader = ImageReader(
                        self.conf.data_dir,
                        self.conf.data_list,
                        input_size,
                        self.conf.random_scale,
                        self.conf.random_mirror,
                        self.conf.ignore_label,
                        IMG_MEAN,
                        self.coord)
                    self.image_batch, self.label_batch = reader.dequeue(self.conf.batch_size)
                    self.im_list.append(self.image_batch)
                    image_batch_075 = tf.image.resize_images(self.image_batch, [int(h * 0.75), int(w * 0.75)])
                    image_batch_05 = tf.image.resize_images(self.image_batch, [int(h * 0.5), int(w * 0.5)])

                # Create network
                with tf.variable_scope('', reuse=False):
                    net = Deeplab_v2(self.image_batch, self.conf.num_classes, True, rescale075=False, rescale05=False, crf_type = self.conf.crf_type)
                
                with tf.variable_scope('', reuse=True):
                    net075 = Deeplab_v2(image_batch_075, self.conf.num_classes, True, rescale075=True, rescale05=False, crf_type = self.conf.crf_type)

                with tf.variable_scope('', reuse=True):
                    net05 = Deeplab_v2(image_batch_05, self.conf.num_classes, True, rescale075=False, rescale05=True, crf_type = self.conf.crf_type)

                # Variables that load from pre-trained model.
                restore_var = [v for v in tf.global_variables() if ('fc' not in v.name and 'crfrnn' not in v.name)]
                restore_vars.append(restore_var)

                # Trainable Variables
                all_trainable = tf.trainable_variables()
                # Fine-tune part
                for name in previous_crf_names:
                    for v in all_trainable:
                        if v.name == name:
                            all_trainable.remove(v)

                crf_trainable = [v for v in all_trainable if ('crfrnn' in v.name and v.name not in previous_crf_names)]
                previous_crf_names.extend(v.name for v in crf_trainable)
                encoder_trainable = [v for v in all_trainable if 'fc' not in v.name and 'crfrnn' not in v.name] # lr * 1.0
                
                # Remove encoder_trainable from all_trainable
                #all_trainable = [v for v in all_trainable if v not in encoder_trainable]
                
                # Decoder part
                decoder_trainable = [v for v in all_trainable if 'fc' in v.name and 'crfrnn' not in v.name]

                decoder_w_trainable = [v for v in decoder_trainable if ('weights' in v.name or 'gamma' in v.name) and 'crfrnn' not in v.name] # lr * 10.0
                decoder_b_trainable = [v for v in decoder_trainable if ('biases' in v.name or 'beta' in v.name) and 'crfrnn' not in v.name] # lr * 20.0
                # Check
                assert(len(all_trainable) == len(decoder_trainable) + len(crf_trainable))  + len(encoder_trainable)
                assert(len(decoder_trainable) == len(decoder_w_trainable) + len(decoder_b_trainable))
                
                # Network raw output
                raw_output100 = net.outputs
                
                raw_output075 = net075.outputs
                raw_output05 = net05.outputs
                raw_output = tf.reduce_max(tf.stack([raw_output100,
                                                     tf.image.resize_images(raw_output075, tf.shape(raw_output100)[1:3,]),
                                                     tf.image.resize_images(raw_output05, tf.shape(raw_output100)[1:3,])]), axis=0)
                
                # Ground Truth: ignoring all labels greater or equal than n_classes
                label_proc = prepare_label(self.label_batch, tf.stack(raw_output.get_shape()[1:3]), num_classes=self.conf.num_classes, one_hot=True) # [batch_size, h, w]
                label_proc075 = prepare_label(self.label_batch, tf.stack(raw_output075.get_shape()[1:3]), num_classes=self.conf.num_classes, one_hot=True)
                label_proc05 = prepare_label(self.label_batch, tf.stack(raw_output05.get_shape()[1:3]), num_classes=self.conf.num_classes, one_hot=True)
                
                raw_gt = tf.reshape(label_proc, [-1,])
                raw_gt075 = tf.reshape(label_proc075, [-1,])
                raw_gt05 = tf.reshape(label_proc05, [-1,])
                
                indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, self.conf.num_classes - 1)), 1)
                indices075 = tf.squeeze(tf.where(tf.less_equal(raw_gt075, self.conf.num_classes - 1)), 1)
                indices05 = tf.squeeze(tf.where(tf.less_equal(raw_gt05, self.conf.num_classes - 1)), 1)
                
                gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
                gt075 = tf.cast(tf.gather(raw_gt075, indices075), tf.int32)
                gt05 = tf.cast(tf.gather(raw_gt05, indices05), tf.int32)
                
                raw_prediction = tf.reshape(raw_output, [-1, self.conf.num_classes])
                raw_prediction100 = tf.reshape(raw_output100, [-1, self.conf.num_classes])
                raw_prediction075 = tf.reshape(raw_output075, [-1, self.conf.num_classes])
                raw_prediction05 = tf.reshape(raw_output05, [-1, self.conf.num_classes])
                
                prediction = tf.gather(raw_prediction, indices)
                prediction100 = tf.gather(raw_prediction100, indices)
                prediction075 = tf.gather(raw_prediction075, indices075)
                prediction05 = tf.gather(raw_prediction05, indices05)

                # Pixel-wise softmax_cross_entropy loss
                #loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
                loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=raw_prediction, labels=tf.reshape(label_proc[0], (h*w, self.conf.num_classes)))
                '''
                coefficients = [0.01460247, 1.25147725, 2.88479363, 1.20348121, 1.65261654, 1.67514772,
                                0.62338799, 0.7729363,  0.42038501, 0.98557268, 1.31867536, 0.85313332,
                                0.67227604, 1.21317965, 1.        , 0.24263748, 1.80877607, 1.3082213,
                                0.79664027, 0.72543945, 1.27823374]
                '''
                #loss = weighted_loss(self.conf.num_classes, coefficients, labels=tf.reshape(label_proc[0], (h*w, self.conf.num_classes)), logits=raw_prediction)
                #loss100 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction100, labels=gt)
                loss100 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=raw_prediction100, labels=tf.reshape(label_proc[0], (h*w, self.conf.num_classes)))
                #loss100 = weighted_loss(self.conf.num_classes, coefficients, labels=tf.reshape(label_proc[0], (h*w, self.conf.num_classes)), logits=raw_prediction100)
                #loss075 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction075, labels=gt075)
                loss075 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=raw_prediction075, labels=tf.reshape(label_proc075[0], (int(h * 0.75) * int(w * 0.75), self.conf.num_classes)))
                #loss075 = weighted_loss(self.conf.num_classes, coefficients, labels=tf.reshape(label_proc075[0], (int(h * 0.75) * int(w * 0.75), self.conf.num_classes)), logits=raw_prediction075)
                #loss05 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction05, labels=gt05)
                loss05 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=raw_prediction05, labels=tf.reshape(label_proc05[0], (int(h * 0.5) * int(w * 0.5), self.conf.num_classes)))
                #loss05 = weighted_loss(self.conf.num_classes, coefficients, labels=tf.reshape(label_proc05[0], (int(h * 0.5) * int(w * 0.5), self.conf.num_classes)), logits=raw_prediction05)
                
                # L2 regularization
                l2_losses = [self.conf.weight_decay * tf.nn.l2_loss(v) for v in all_trainable if 'weights' in v.name]

                # Loss function
                self.reduced_loss = tf.reduce_mean(loss) + tf.reduce_mean(loss100) + tf.reduce_mean(loss075) + tf.reduce_mean(loss05) + tf.add_n(l2_losses)
                
                # Define optimizers
                # 'poly' learning rate
                base_lr = tf.constant(self.conf.learning_rate)
                self.curr_step = tf.placeholder(dtype=tf.float32, shape=())
                learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - self.curr_step / self.conf.num_steps), self.conf.power))
                # We have several optimizers here in order to handle the different lr_mult
                # which is a kind of parameters in Caffe. This controls the actual lr for each
                # layer.
                opt_encoder = tf.train.MomentumOptimizer(learning_rate, self.conf.momentum)
                opt_decoder_w = tf.train.MomentumOptimizer(learning_rate * 10.0, self.conf.momentum)
                opt_decoder_b = tf.train.MomentumOptimizer(learning_rate * 20.0, self.conf.momentum)
                opt_crf = tf.train.MomentumOptimizer(learning_rate, self.conf.momentum)
                
                # Gradient accumulation
                # Define a variable to accumulate gradients.
                accum_grads = [tf.Variable(tf.zeros_like(v.initialized_value()),
                                           trainable=False) for v in encoder_trainable + decoder_w_trainable + decoder_b_trainable + crf_trainable]

                # Define an operation to clear the accumulated gradients for next batch.
                self.zero_op = [v.assign(tf.zeros_like(v)) for v in accum_grads]
                # To make sure each layer gets updated by different lr's, we do not use 'minimize' here.
                # Instead, we separate the steps compute_grads+update_params.
                # Compute grads
                grads = tf.gradients(self.reduced_loss, encoder_trainable + decoder_w_trainable + decoder_b_trainable + crf_trainable)
                # Accumulate and normalise the gradients.
                self.accum_grads_op = [accum_grads[i].assign_add(grad / self.conf.grad_update_every) for i, grad in enumerate(grads)]
                
                grads_encoder = accum_grads[:len(encoder_trainable)]
                grads_decoder_w = accum_grads[len(encoder_trainable): len(encoder_trainable) + len(decoder_w_trainable)]
                grads_decoder_b = accum_grads[(len(encoder_trainable)+ len(decoder_w_trainable)):(len(encoder_trainable)+len(decoder_w_trainable)+len(decoder_b_trainable))]
                grads_crf = accum_grads[len(encoder_trainable)+len(decoder_w_trainable)+len(decoder_b_trainable):] # assuming crf gradients are appended to the end

                zip_encoder.append(list(zip(grads_encoder, encoder_trainable)))
                zip_decoder_b.append(list(zip(grads_decoder_b, decoder_b_trainable)))
                zip_decoder_w.append(list(zip(grads_decoder_w, decoder_w_trainable)))
                zip_crf.append(list(zip(grads_crf, crf_trainable)))


        avg_grads_encoder = average_gradients(zip_encoder)
        avg_grads_decoder_w = average_gradients(zip_decoder_w)
        avg_grads_decoder_b = average_gradients(zip_decoder_b)
        avg_grads_crf = average_gradients(zip_crf)

        for i in range(len(gpu_list)):
            with tf.device(gpu_list[i]):
                # Update params
                train_op_conv = opt_encoder.apply_gradients(avg_grads_encoder)
                train_op_fc_w = opt_decoder_w.apply_gradients(avg_grads_decoder_w)
                train_op_fc_b = opt_decoder_b.apply_gradients(avg_grads_decoder_b)
                train_op_crf = opt_crf.apply_gradients(avg_grads_crf)

        # Finally, get the train_op!
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # for collecting moving_mean and moving_variance
        with tf.control_dependencies(update_ops):
            self.train_op = tf.group(train_op_fc_w, train_op_fc_b, train_op_crf) # train_op_conv

        # Saver for storing checkpoints of the model
        self.saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=0)

        # Loader for loading the pre-trained model
        for i in range(len(gpu_list)):
            with tf.device(gpu_list[i]):
                self.loaders.append(tf.train.Saver(var_list=restore_vars[i]))
                #self.loaders.append(tf.train.Saver(var_list=tf.global_variables()))

        # Training summary
        # Processed predictions: for visualisation.
        raw_output_up = tf.image.resize_bilinear(raw_output, input_size)
        raw_output_up = tf.argmax(raw_output_up, axis=3)
        self.pred = tf.expand_dims(raw_output_up, axis=3)
        # Image summary.
        images_summary = tf.py_func(inv_preprocess, [self.image_batch, 1, IMG_MEAN], tf.uint8)
        labels_summary = tf.py_func(decode_labels, [self.label_batch, 1, self.conf.num_classes], tf.uint8)
        preds_summary = tf.py_func(decode_labels, [self.pred, 1, self.conf.num_classes], tf.uint8)
        self.total_summary = tf.summary.image('images',
            tf.concat(axis=2, values=[images_summary, labels_summary, preds_summary]),
            max_outputs=1) # Concatenate row-wise.
        if not os.path.exists(self.conf.logdir):
            os.makedirs(self.conf.logdir)
        self.summary_writer = tf.summary.FileWriter(self.conf.logdir, graph=tf.get_default_graph())

    def test_setup(self):
        # Create queue coordinator.
        self.coord = tf.train.Coordinator()

        # Load reader
        with tf.name_scope("create_inputs"):
            reader = ImageReader(
                self.conf.data_dir,
                self.conf.valid_data_list,
                None, # the images have different sizes
                False, # no data-aug
                False, # no data-aug
                self.conf.ignore_label,
                IMG_MEAN,
                self.coord)
            image, label = reader.image, reader.label # [h, w, 3 or 1]

        # Add one batch dimension [1, h, w, 3 or 1]
        self.image_batch, self.label_batch = tf.expand_dims(image, dim=0), tf.expand_dims(label, dim=0)
        h_orig, w_orig = tf.to_float(tf.shape(self.image_batch)[1]), tf.to_float(tf.shape(self.image_batch)[2])
        image_batch_075 = tf.image.resize_images(self.image_batch, tf.stack([tf.to_int32(tf.multiply(h_orig, 0.75)), tf.to_int32(tf.multiply(w_orig, 0.75))]))
        image_batch_05 = tf.image.resize_images(self.image_batch, tf.stack([tf.to_int32(tf.multiply(h_orig, 0.5)), tf.to_int32(tf.multiply(w_orig, 0.5))]))

        # Create network
        with tf.variable_scope('', reuse=False):
            net = Deeplab_v2(self.image_batch, self.conf.num_classes, False, rescale075=False, rescale05=False, crf_type = self.conf.crf_type)
        with tf.variable_scope('', reuse=True):
            net075 = Deeplab_v2(image_batch_075, self.conf.num_classes, False, rescale075=True, rescale05=False, crf_type = self.conf.crf_type)
        with tf.variable_scope('', reuse=True):
            net05 = Deeplab_v2(image_batch_05, self.conf.num_classes, False, rescale075=False, rescale05=True, crf_type = self.conf.crf_type)

        # predictions
        # Network raw output
        raw_output100 = net.outputs
        raw_output075 = net075.outputs
        raw_output05 = net05.outputs
        raw_output = tf.reduce_max(tf.stack([raw_output100,
                                    tf.image.resize_images(raw_output075, tf.shape(raw_output100)[1:3,]),
                                    tf.image.resize_images(raw_output05, tf.shape(raw_output100)[1:3,])]), axis=0)

        raw_output = tf.image.resize_bilinear(raw_output, tf.shape(self.image_batch)[1:3,])
        raw_output = tf.argmax(raw_output, axis=3)
        pred = tf.expand_dims(raw_output, dim=3)
        self.pred = tf.reshape(pred, [-1,])
        # labels
        gt = tf.reshape(self.label_batch, [-1,])
        # Ignoring all labels greater than or equal to n_classes.
        temp = tf.less_equal(gt, self.conf.num_classes - 1)
        weights = tf.cast(temp, tf.int32)

        # fix for tf 1.3.0
        gt = tf.where(temp, gt, tf.cast(temp, tf.uint8))

        # Pixel accuracy
        self.accu, self.accu_update_op = tf.contrib.metrics.streaming_accuracy(
            self.pred, gt, weights=weights)

        # mIoU
        self.mIoU, self.mIou_update_op = tf.contrib.metrics.streaming_mean_iou(
            self.pred, gt, num_classes=self.conf.num_classes, weights=weights)

        # confusion matrix
        self.confusion_matrix = tf.contrib.metrics.confusion_matrix(
            self.pred, gt, num_classes=self.conf.num_classes, weights=weights)

        # Loader for loading the checkpoint
        self.loader = tf.train.Saver(var_list=tf.global_variables())

    def predict_setup(self):
        # Create queue coordinator.
        self.coord = tf.train.Coordinator()

        # Load reader
        with tf.name_scope("create_inputs"):
            reader = ImageReader(
                self.conf.data_dir,
                self.conf.test_data_list,
                None, # the images have different sizes
                False, # no data-aug
                False, # no data-aug
                self.conf.ignore_label,
                IMG_MEAN,
                self.coord)
            image, label = reader.image, reader.label # [h, w, 3 or 1]
            
        # Add one batch dimension [1, h, w, 3 or 1]
        image_batch, label_batch = tf.expand_dims(image, dim=0), tf.expand_dims(label, dim=0)
        h_orig, w_orig = tf.to_float(tf.shape(image_batch)[1]), tf.to_float(tf.shape(image_batch)[2])
        image_batch_075 = tf.image.resize_images(image_batch, tf.stack([tf.to_int32(tf.multiply(h_orig, 0.75)), tf.to_int32(tf.multiply(w_orig, 0.75))]))
        image_batch_05 = tf.image.resize_images(image_batch, tf.stack([tf.to_int32(tf.multiply(h_orig, 0.5)), tf.to_int32(tf.multiply(w_orig, 0.5))]))

        # Create network
        with tf.variable_scope('', reuse=False):
            net = Deeplab_v2(image_batch, self.conf.num_classes, False, rescale075 = False, rescale05 = False, crf_type = self.conf.crf_type)
        with tf.variable_scope('', reuse=True):
            net075 = Deeplab_v2(image_batch_075, self.conf.num_classes, False, rescale075 = True, rescale05 = False, crf_type = self.conf.crf_type)
        with tf.variable_scope('', reuse=True):
            net05 = Deeplab_v2(image_batch_05, self.conf.num_classes, False, rescale075 = False, rescale05 = True, crf_type = self.conf.crf_type)

        # predictions
        # Network raw output
        raw_output100 = net.outputs
        self.resized_decoder100 = net.decoding
        raw_output075 = net075.outputs
        raw_output05 = net05.outputs
        raw_output = tf.reduce_max(tf.stack([raw_output100,
                                    tf.image.resize_images(raw_output075, tf.shape(raw_output100)[1:3,]),
                                    tf.image.resize_images(raw_output05, tf.shape(raw_output100)[1:3,])]), axis=0)
        raw_output = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3,])
        raw_output = tf.argmax(raw_output, axis=3)
        self.pred = tf.cast(tf.expand_dims(raw_output, dim=3), tf.uint8)

        # Create directory
        if not os.path.exists(self.conf.out_dir):
            os.makedirs(self.conf.out_dir)
            os.makedirs(self.conf.out_dir + '/prediction')
            if self.conf.visual:
                os.makedirs(self.conf.out_dir + '/visual_prediction')
                os.makedirs(self.conf.out_dir + '/resized_decoding')

        # Loader for loading the checkpoint
        self.loader = tf.train.Saver(var_list=tf.global_variables())

    def save(self, saver, step):
        '''
        Save weights.
        '''
        model_name = 'model.ckpt'
        checkpoint_path = os.path.join(self.conf.modeldir, model_name)
        if not os.path.exists(self.conf.modeldir):
            os.makedirs(self.conf.modeldir)
        saver.save(self.sess, checkpoint_path, global_step=step)
        print('The checkpoint has been created.')

    def load(self, saver, filename):
        '''
        Load trained weights.
        '''
        #vars = slim.get_variables_to_restore()
        #variables_to_restore = [v for v in vars if 'crfrnn' not in v.name]
        #saver_new = tf.train.Saver(variables_to_restore)
        saver.restore(self.sess, filename)
        print("Restored model parameters from {}".format(filename))

    def compute_IoU_per_class(self, confusion_matrix):
        mIoU = 0
        for i in range(self.conf.num_classes):
            # IoU = true_positive / (true_positive + false_positive + false_negative)
            TP = confusion_matrix[i,i]
            FP = np.sum(confusion_matrix[:, i]) - TP
            FN = np.sum(confusion_matrix[i]) - TP
            IoU = TP / (TP + FP + FN)
            print ('class %d: %.3f' % (i, IoU))
            mIoU += IoU / self.conf.num_classes
        print ('mIoU: %.3f' % mIoU)

import argparse
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
from model_msc_personfine import Model_msc



"""
This script defines hyperparameters.
"""



def configure():
	flags = tf.app.flags

	# training
        # deeplab paper trains with batch size 10 for 20K iterations -> using batch size 2 means need 100K iterations
        # -> batch size 16 means 12.5K iterations
	flags.DEFINE_integer('num_steps', 20000, 'maximum number of iterations')
	flags.DEFINE_integer('save_interval', 1000, 'number of iterations for saving and visualization')
	flags.DEFINE_integer('random_seed', 1234, 'random seed')
	flags.DEFINE_float('weight_decay', 0.0005, 'weight decay rate')
	flags.DEFINE_float('learning_rate', 2.5e-4, 'learning rate')
	flags.DEFINE_float('power', 0.9, 'hyperparameter for poly learning rate')
	flags.DEFINE_float('momentum', 0.9, 'momentum')
	flags.DEFINE_string('encoder_name', 'deeplab', 'name of pre-trained model, res101, res50 or deeplab')
	flags.DEFINE_string('pretrain_file', 'see model file', 'pre-trained model filename corresponding to encoder_name')
	flags.DEFINE_string('data_list', './dataset/person_train.txt', 'training data list filename')
	flags.DEFINE_integer('grad_update_every', 1, 'gradient accumulation step')
	# Note: grad_update_every = true training batch size
	
	# validation
	flags.DEFINE_integer('valid_step', 20000, 'checkpoint number for validation')
	flags.DEFINE_integer('valid_num_steps', 1514, '= number of validation samples')
	flags.DEFINE_string('valid_data_list', './dataset/person_test.txt', 'validation data list filename')

	# prediction / saving outputs for testing or validation
	flags.DEFINE_string('out_dir', 'output', 'directory for saving outputs')
	flags.DEFINE_integer('test_step', 20000, 'checkpoint number for testing/validation')
	flags.DEFINE_integer('test_num_steps', 1514, '= number of testing/validation samples')
	flags.DEFINE_string('test_data_list', './dataset/person_test.txt', 'testing/validation data list filename')
	flags.DEFINE_boolean('visual', True, 'whether to save predictions for visualization')

	# data
	flags.DEFINE_string('data_dir', '/om2/user/cfmata/crfasrnn_keras/data/person_fine_parts', 'data directory')
	flags.DEFINE_integer('batch_size', 2, 'training batch size PER GPU')
	flags.DEFINE_integer('input_height', 321, 'input image height')
	flags.DEFINE_integer('input_width', 321, 'input image width')
	flags.DEFINE_integer('num_classes', 25, 'number of classes')
	flags.DEFINE_integer('ignore_label', 255, 'label pixel value that should be ignored')
	flags.DEFINE_boolean('random_scale', True, 'whether to perform random scaling data-augmentation')
	flags.DEFINE_boolean('random_mirror', True, 'whether to perform random left-right flipping data-augmentation')
	
	# log
	flags.DEFINE_string('modeldir', 'model_personfine', 'model directory')
	flags.DEFINE_string('logfile', 'log.txt', 'training log filename')
	flags.DEFINE_string('logdir', 'log', 'training log directory')
	
	flags.FLAGS.__dict__['__parsed'] = False
	return flags.FLAGS

def main(_):
	parser = argparse.ArgumentParser()
	parser.add_argument('--option', dest='option', type=str, default='train',
		help='actions: train, test, or predict')
	args = parser.parse_args()

	if args.option not in ['train', 'test', 'predict']:
		print('invalid option: ', args.option)
		print("Please input a option: train, test, or predict")
	else:
		# Set up tf session and initialize variables. 
		# config = tf.ConfigProto()
		# config.gpu_options.allow_growth = True
		# sess = tf.Session(config=config)
		sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
		# Run
		model = Model_msc(sess, configure())
		getattr(model, args.option)()


if __name__ == '__main__':
	# Choose which gpu or cpu to use
	tf.app.run()

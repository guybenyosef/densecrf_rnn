import argparse
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
from deeplabv2_model_horsecowfine import Model_msc
import pdb

"""
This script defines hyperparameters.
"""



def configure(args):
    flags = tf.app.flags

    # training
    flags.DEFINE_integer('num_steps', args.steps, 'maximum number of iterations')
    flags.DEFINE_integer('save_interval', 100, 'number of iterations for saving and visualization')
    flags.DEFINE_integer('random_seed', 1234, 'random seed')
    flags.DEFINE_float('weight_decay', 0.0005, 'weight decay rate')
    flags.DEFINE_float('learning_rate', 2.5e-4, 'learning rate') # 2.5e-4 with bs 10
    flags.DEFINE_float('power', 0.9, 'hyperparameter for poly learning rate')
    flags.DEFINE_float('momentum', 0.9, 'momentum')
    flags.DEFINE_string('encoder_name', 'deeplab', 'name of pre-trained model, res101, res50 or deeplab')
    flags.DEFINE_string('pretrain_file', args.weights, 'pre-trained model filename is latest checkout filenmae')
    #flags.DEFINE_string('pretrain_file', 'see model file', 'pre-trained model filename corresponding to encoder_name')
    flags.DEFINE_string('data_list', './lst/pascal_parts_lists_horse_cow_person/horsecow_train.txt', 'training data list filename')
    flags.DEFINE_integer('grad_update_every', 2, 'gradient accumulation step')
    # Note: grad_update_every = true training batch size

    flags.DEFINE_string('crf_type', args.crftype, 'Type of crf layer to add. Options are crf, crfSP, crfSPAT, crfSPIO, crfALL, None')
    flags.DEFINE_string('dataset', 'horse_fine', 'Name of dataset. Options are pascal_voc12, horse_fine, person_fine, horse_coarse, person_coarse')

    # validation
    flags.DEFINE_integer('valid_step', 20000, 'checkpoint number for validation')
    flags.DEFINE_integer('valid_num_steps', 216, '= number of validation samples')
    flags.DEFINE_string('valid_data_list', './lst/pascal_parts_lists_horse_cow_person/horsecow_test.txt', 'validation data list filename')

    # prediction / saving outputs for testing or validation
    flags.DEFINE_string('out_dir', args.outdir + 'horsecow_fine_parts', 'directory for saving outputs')
    flags.DEFINE_integer('test_step', 20000, 'checkpoint number for testing/validation')
    flags.DEFINE_integer('test_num_steps', 216, '= number of testing/validation samples')
    flags.DEFINE_string('test_data_list', './lst/pascal_parts_lists_horse_cow_person/horsecow_test.txt', 'testing/validation data list filename')
    flags.DEFINE_boolean('visual', True, 'whether to save predictions for visualization')

    # data
    flags.DEFINE_string('data_dir', args.datadir + 'horsecow_fine_parts', 'data directory')
    flags.DEFINE_integer('batch_size', 1, 'training batch size PER GPU')
    flags.DEFINE_integer('input_height', 125, 'input image height') # was 321, 175,  125
    flags.DEFINE_integer('input_width', 125, 'input image width')
    flags.DEFINE_integer('num_classes', 22, 'number of classes')
    flags.DEFINE_integer('ignore_label', 255, 'label pixel value that should be ignored')
    flags.DEFINE_boolean('random_scale', True, 'whether to perform random scaling data-augmentation')
    flags.DEFINE_boolean('random_mirror', True, 'whether to perform random left-right flipping data-augmentation')
    
    # log
    flags.DEFINE_string('modeldir', args.outdir + args.crftype, 'model directory')
    flags.DEFINE_string('logfile', args.outdir + args.crftype + '/log/log.txt', 'training log filename')
    flags.DEFINE_string('logdir', args.outdir + args.crftype + '/log', 'training log directory')
    
    flags.FLAGS.__dict__['__parsed'] = False
    return flags.FLAGS

def argument_parser():

    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('--option', type=str, default='train', help='actions: train, test, or predict')
    parser.add_argument('--crftype', type=str, default='crf',help='Options are crf, crfSP, crfSPAT, crfSPIO, crfALL, None')
    parser.add_argument('--steps', type=int, default=100, help='maximum number of iterations')
    parser.add_argument('--outdir', type=str, default='/storage/gby/semseg/deepLabV2_output/', help='directory for saving outputs')
    parser.add_argument('--datadir', type=str, default='/storage/cfmata/deeplab/openmind_copy/crfasrnn_keras/data/', help='data directory')
    parser.add_argument('--weights', type=str, default='/storage/gby/semseg/deepLabV2_output/deeplab_resnet_init.ckpt', help='pre-trained model filename is latest checkout filenmae')

    return parser.parse_args()

def main(_):

    args = argument_parser()

    conf = configure(args)

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
        model = Model_msc(sess, conf)
        getattr(model, args.option)()


if __name__ == '__main__':
    # Choose which gpu or cpu to use
    tf.app.run()

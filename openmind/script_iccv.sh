#!/usr/bin/env bash

#source /storage/cfmata/deeplab/crf_rnn/crfasrnn_keras/tensorflow/bin/activate
#source /afs/csail.mit.edu/u/g/gby/libs/anaconda3/bin/activate tensorflow

# === train: ===

# melville:
#python train_horsecowfine_tf.py --steps 1000 --option train --crftype crfSPIO --datadir /storage/cfmata/deeplab/openmind_copy/crfasrnn_keras/data/ --weights /storage/gby/semseg/deepLabV2_output/model_horsecowfine/model.ckpt-20000
python train_horsecowfine_tf.py --steps 20000 --option train --crftype crf --datadir /storage/cfmata/deeplab/openmind_copy/crfasrnn_keras/data/ --weights /storage/gby/semseg/deepLabV2_output/model_horsecowfine/model.ckpt-20000


# devon:
python train_horsecowfine_tf.py --steps 20000 --option train --crftype crfSP --datadir /storage/gby/semseg/data/ --weights /storage/gby/semseg/deepLabV2_output/model_horsecowfine/model.ckpt-20000
python train_horsecowfine_tf.py --steps 20000 --option train --crftype crf --datadir /storage/gby/semseg/data/ --weights /storage/gby/semseg/deepLabV2_output/model_horsecowfine/model.ckpt-20000
#python train_horsecowfine_tf.py --steps 20000 --option train --crftype crfSPIO --datadir /storage/gby/semseg/data/ --weights /storage/gby/semseg/deepLabV2_output/model_horsecowfine/model.ckpt-20000
#python train_horsecowfine_tf.py --steps 20000 --option train --crftype crfSPAT --datadir /storage/gby/semseg/data/ --weights /storage/gby/semseg/deepLabV2_output/model_horsecowfine/model.ckpt-20000



# === test: ===
# melville:
python train_horsecowfine_tf.py --steps 100 --option test --crftype crfSP --datadir /storage/cfmata/deeplab/openmind_copy/crfasrnn_keras/data/
python train_horsecowfine_tf.py --steps 100 --option test --crftype crf --datadir /storage/cfmata/deeplab/openmind_copy/crfasrnn_keras/data/
#python train_horsecowfine_tf.py --steps 1000 --option test --crftype crf --datadir /storage/cfmata/deeplab/openmind_copy/crfasrnn_keras/data/ --weights /storage/gby/semseg/deepLabV2_output/model_horsecowfine/model.ckpt-20000

# devon
#python train_horsecowfine_tf.py --steps 100 --option test --crftype crf --datadir /storage/gby/semseg/data/
#python train_horsecowfine_tf.py --steps 100 --option test --crftype crfSP --datadir /storage/gby/semseg/data/



# === predictions: ===
# melville:
python train_horsecowfine_tf.py --steps 100 --option predict --crftype crfSP --datadir /storage/cfmata/deeplab/openmind_copy/crfasrnn_keras/data/
python train_horsecowfine_tf.py --steps 100 --option predict --crftype crf --datadir /storage/cfmata/deeplab/openmind_copy/crfasrnn_keras/data/
# devon:
#python train_horsecowfine_tf.py --steps 100 --option predict --crftype crfSP --datadir /storage/gby/semseg/data/
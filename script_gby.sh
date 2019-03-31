#!/usr/bin/env

# installation:
# ----------------
conda create -n tensorflow pip python=3.5
source /afs/csail.mit.edu/u/g/gby/libs/anaconda3/bin/activate tensorflow
pip install --ignore-installed tensorflow-gpu==1.4
conda install -c conda-forge keras
pip install --ignore-installed tensorflow-gpu==1.4
export KERAS_BACKEND=tensorflow

#
cd src/cpp
make

# run test:
# ----------
source /afs/csail.mit.edu/u/g/gby/libs/anaconda3/bin/activate tensorflow
export KERAS_BACKEND=tensorflow
python run_demo.py
python run_model_on_img.py /storage/gby/semseg/voc12_weights image.jpg

# run train:
# ----------
python train.py




python train_gby.py -m fcn_RESNET50_8s -e 1 -bs 32 -vb 1
python train_gby.py -ds voc2012 -m fcn_RESNET50_8s_crfrnn -vb 1 -is 224 -e 50 -bs 1 


python eval_gby.py -ds horsecoarse -w /storage/gby/semseg/horsecoarse_weights_fcn_RESNET50_8s_100ep
python eval_gby.py -ds voc2012 -w /storage/gby/semseg/voc2012_weights_fcn_RESNET50_8s_10ep


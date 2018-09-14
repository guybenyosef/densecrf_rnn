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
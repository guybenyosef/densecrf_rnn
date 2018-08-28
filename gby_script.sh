#!/usr/bin/env

conda create -n tensorflow pip python=3.5
source /afs/csail.mit.edu/u/g/gby/libs/anaconda3/bin/activate tensorflow
pip install --ignore-installed tensorflow-gpu==1.4
conda install -c conda-forge keras
pip install --ignore-installed tensorflow-gpu==1.4
export KERAS_BACKEND=tensorflow

#
cd src/cpp
make


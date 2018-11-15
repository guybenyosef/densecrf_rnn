#!/usr/bin/env bash

# running experiments for the horsefine:
num_ep=1
echo "num epochs is $num_ep"
python train_gby.py -m fcn_RESNET50_8s_crfrnn -is 224 -ds horsecoarse -e $num_ep -bs 1
python train_gby.py -m fcn_RESNET50_8s_crfrnnSP -is 224 -ds horsecoarse -e $num_ep -bs 1
python train_gby.py -m fcn_RESNET50_8s_crfrnnSPIO -is 224 -ds horsecoarse -e $num_ep -bs 1
python train_gby.py -m fcn_RESNET50_8s_crfrnnSPAT -is 224 -ds horsecoarse -e $num_ep -bs 1
python train_gby.py -m fcn_RESNET50_8s_crfrnnSPIOAT -is 224 -ds horsecoarse -e $num_ep -bs 1

# /storage/gby/semseg/horsefine_weights_fcn_RESNET50_8s_crfrnnSP_1ep
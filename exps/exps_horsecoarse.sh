#!/usr/bin/env bash

# running experiments for the horsecoarse:
num_ep=1
gpu_num=2
echo "num epochs is $num_ep"
echo "gpu device index is $gpu_num"
python train_gby.py -m fcn_RESNET50_8s_crfrnn -is 224 -ds horsecoarse -e $num_ep -bs 1 -g $gpu_num -ft /storage/gby/semseg/horsecoarse_weights_fcn_RESNET50_8s_5000ep
python train_gby.py -m fcn_RESNET50_8s_crfrnnSP -is 224 -ds horsecoarse -e $num_ep -bs 1 -g $gpu_num -ft /storage/gby/semseg/horsecoarse_weights_fcn_RESNET50_8s_5000ep
python train_gby.py -m fcn_RESNET50_8s_crfrnnSPIO -is 224 -ds horsecoarse -e $num_ep -bs 1 -g $gpu_num -ft /storage/gby/semseg/horsecoarse_weights_fcn_RESNET50_8s_5000ep
python train_gby.py -m fcn_RESNET50_8s_crfrnnSPAT -is 224 -ds horsecoarse -e $num_ep -bs 1 -g $gpu_num -ft /storage/gby/semseg/horsecoarse_weights_fcn_RESNET50_8s_5000ep
python train_gby.py -m fcn_RESNET50_8s_crfrnnSPIOAT -is 224 -ds horsecoarse -e $num_ep -bs 1 -g $gpu_num -ft /storage/gby/semseg/horsecoarse_weights_fcn_RESNET50_8s_5000ep

/storage/gby/semseg/horsefine_weights_fcn_RESNET50_8s_1000ep


#python train_gby.py -m fcn_RESNET50_8s_crfrnn -is 224 -ds horsecoarse -e $num_ep -bs 1 -g $gpu_num -w /storage/gby/semseg/horsecoarse_weights_fcn_RESNET50_8s_crfrnn_1ep
#python train_gby.py -m fcn_RESNET50_8s_crfrnnSP -is 224 -ds horsecoarse -e $num_ep -bs 1 -g $gpu_num -w /storage/gby/semseg/horsecoarse_weights_fcn_RESNET50_8s_crfrnnSP_1ep
##python train_gby.py -m fcn_RESNET50_8s_crfrnnSPIO -is 224 -ds horsecoarse -e $num_ep -bs 1 -g $gpu_num -w /storage/gby/semseg/horsecoarse_weights_fcn_RESNET50_8s_crfrnnSPIO_1ep
#python train_gby.py -m fcn_RESNET50_8s_crfrnnSPAT -is 224 -ds horsecoarse -e $num_ep -bs 1 -g $gpu_num -w /storage/gby/semseg/horsecoarse_weights_fcn_RESNET50_8s_crfrnnSPAT_1ep
#python train_gby.py -m fcn_RESNET50_8s_crfrnnSPIOAT -is 224 -ds horsecoarse -e $num_ep -bs 1 -g $gpu_num -w /storage/gby/semseg/horsecoarse_weights_fcn_RESNET50_8s_crfrnnSPIOAT_1ep
#
#cp /storage/gby/semseg/horsecoarse_weights_fcn_RESNET50_8s_crfrnn* /storage/gby/semseg/ep2
#
    #python train_gby.py -m fcn_RESNET50_8s_crfrnn -is 224 -ds horsecoarse -e $num_ep -bs 1 -g $gpu_num -w /storage/gby/semseg/horsecoarse_weights_fcn_RESNET50_8s_crfrnn_1ep
#python train_gby.py -m fcn_RESNET50_8s_crfrnnSP -is 224 -ds horsecoarse -e $num_ep -bs 1 -g $gpu_num -w /storage/gby/semseg/horsecoarse_weights_fcn_RESNET50_8s_crfrnnSP_1ep
##python train_gby.py -m fcn_RESNET50_8s_crfrnnSPIO -is 224 -ds horsecoarse -e $num_ep -bs 1 -g $gpu_num -w /storage/gby/semseg/horsecoarse_weights_fcn_RESNET50_8s_crfrnnSPIO_1ep
#python train_gby.py -m fcn_RESNET50_8s_crfrnnSPAT -is 224 -ds horsecoarse -e $num_ep -bs 1 -g $gpu_num -w /storage/gby/semseg/horsecoarse_weights_fcn_RESNET50_8s_crfrnnSPAT_1ep
#python train_gby.py -m fcn_RESNET50_8s_crfrnnSPIOAT -is 224 -ds horsecoarse -e $num_ep -bs 1 -g $gpu_num -w /storage/gby/semseg/horsecoarse_weights_fcn_RESNET50_8s_crfrnnSPIOAT_1ep
#
#cp /storage/gby/semseg/horsecoarse_weights_fcn_RESNET50_8s_crfrnn* /storage/gby/semseg/ep3
#
#python train_gby.py -m fcn_RESNET50_8s_crfrnn -is 224 -ds horsecoarse -e $num_ep -bs 1 -g $gpu_num -w /storage/gby/semseg/horsecoarse_weights_fcn_RESNET50_8s_crfrnn_1ep
#python train_gby.py -m fcn_RESNET50_8s_crfrnnSP -is 224 -ds horsecoarse -e $num_ep -bs 1 -g $gpu_num -w /storage/gby/semseg/horsecoarse_weights_fcn_RESNET50_8s_crfrnnSP_1ep
##python train_gby.py -m fcn_RESNET50_8s_crfrnnSPIO -is 224 -ds horsecoarse -e $num_ep -bs 1 -g $gpu_num -w /storage/gby/semseg/horsecoarse_weights_fcn_RESNET50_8s_crfrnnSPIO_1ep
#python train_gby.py -m fcn_RESNET50_8s_crfrnnSPAT -is 224 -ds horsecoarse -e $num_ep -bs 1 -g $gpu_num -w /storage/gby/semseg/horsecoarse_weights_fcn_RESNET50_8s_crfrnnSPAT_1ep
#python train_gby.py -m fcn_RESNET50_8s_crfrnnSPIOAT -is 224 -ds horsecoarse -e $num_ep -bs 1 -g $gpu_num -w /storage/gby/semseg/horsecoarse_weights_fcn_RESNET50_8s_crfrnnSPIOAT_1ep
#
#cp /storage/gby/semseg/horsecoarse_weights_fcn_RESNET50_8s_crfrnn* /storage/gby/semseg/ep4
#
#python train_gby.py -m fcn_RESNET50_8s_crfrnn -is 224 -ds horsecoarse -e $num_ep -bs 1 -g $gpu_num -w /storage/gby/semseg/horsecoarse_weights_fcn_RESNET50_8s_crfrnn_1ep
#python train_gby.py -m fcn_RESNET50_8s_crfrnnSP -is 224 -ds horsecoarse -e $num_ep -bs 1 -g $gpu_num -w /storage/gby/semseg/horsecoarse_weights_fcn_RESNET50_8s_crfrnnSP_1ep
##python train_gby.py -m fcn_RESNET50_8s_crfrnnSPIO -is 224 -ds horsecoarse -e $num_ep -bs 1 -g $gpu_num -w /storage/gby/semseg/horsecoarse_weights_fcn_RESNET50_8s_crfrnnSPIO_1ep
#python train_gby.py -m fcn_RESNET50_8s_crfrnnSPAT -is 224 -ds horsecoarse -e $num_ep -bs 1 -g $gpu_num -w /storage/gby/semseg/horsecoarse_weights_fcn_RESNET50_8s_crfrnnSPAT_1ep
#python train_gby.py -m fcn_RESNET50_8s_crfrnnSPIOAT -is 224 -ds horsecoarse -e $num_ep -bs 1 -g $gpu_num -w /storage/gby/semseg/horsecoarse_weights_fcn_RESNET50_8s_crfrnnSPIOAT_1ep
#
#cp /storage/gby/semseg/horsecoarse_weights_fcn_RESNET50_8s_crfrnn* /storage/gby/semseg/ep5

python train_gby.py -m fcn_RESNET50_8s_crfrnn -is 224 -ds horsecoarse -e $num_ep -bs 1 -g $gpu_num -w /storage/gby/semseg/horsecoarse_weights_fcn_RESNET50_8s_crfrnn_1ep
python train_gby.py -m fcn_RESNET50_8s_crfrnnSP -is 224 -ds horsecoarse -e $num_ep -bs 1 -g $gpu_num -w /storage/gby/semseg/horsecoarse_weights_fcn_RESNET50_8s_crfrnnSP_1ep
#python train_gby.py -m fcn_RESNET50_8s_crfrnnSPIO -is 224 -ds horsecoarse -e $num_ep -bs 1 -g $gpu_num -w /storage/gby/semseg/horsecoarse_weights_fcn_RESNET50_8s_crfrnnSPIO_1ep
python train_gby.py -m fcn_RESNET50_8s_crfrnnSPAT -is 224 -ds horsecoarse -e $num_ep -bs 1 -g $gpu_num -w /storage/gby/semseg/horsecoarse_weights_fcn_RESNET50_8s_crfrnnSPAT_1ep
python train_gby.py -m fcn_RESNET50_8s_crfrnnSPIOAT -is 224 -ds horsecoarse -e $num_ep -bs 1 -g $gpu_num -w /storage/gby/semseg/horsecoarse_weights_fcn_RESNET50_8s_crfrnnSPIOAT_1ep

cp /storage/gby/semseg/horsecoarse_weights_fcn_RESNET50_8s_crfrnn* /storage/gby/semseg/ep6
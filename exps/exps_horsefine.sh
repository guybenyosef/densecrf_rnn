#!/usr/bin/env bash

# running experiments for the horsefine:
num_ep=100
gpu_num=0
echo "num epochs is $num_ep"
echo "gpu device index is $gpu_num"
python train_gby.py -m fcn_RESNET50_8s_crfrnn -is 224 -ds horsefine -e $num_ep -bs 1 -g $gpu_num -ft /storage/gby/semseg/horsefine_weights_fcn_RESNET50_8s_1000ep
#python train_gby.py -m fcn_RESNET50_8s_crfrnnSP -is 224 -ds horsefine -e $num_ep -bs 1 -g $gpu_num -ft /storage/gby/semseg/horsefine_weights_fcn_RESNET50_8s_500ep
#python train_gby.py -m fcn_RESNET50_8s_crfrnnSPIO -is 224 -ds horsefine -e $num_ep -bs 1 -g $gpu_num -ft /storage/gby/semseg/horsefine_weights_fcn_RESNET50_8s_500ep
#python train_gby.py -m fcn_RESNET50_8s_crfrnnSPAT -is 224 -ds horsefine -e $num_ep -bs 1 -g $gpu_num -ft /storage/gby/semseg/horsefine_weights_fcn_RESNET50_8s_500ep
#python train_gby.py -m fcn_RESNET50_8s_crfrnnSPIOAT -is 224 -ds horsefine -e $num_ep -bs 1 -g $gpu_num -ft /storage/gby/semseg/horsefine_weights_fcn_RESNET50_8s_500ep

python train_gby.py -m fcn_RESNET50_8s_crfrnn -is 224 -ds horsefine -e $num_ep -bs 1 -g $gpu_num -w /storage/gby/semseg/horsefine_weights_fcn_RESNET50_8s_crfrnn_1ep
python train_gby.py -m fcn_RESNET50_8s_crfrnnSP -is 224 -ds horsefine -e $num_ep -bs 1 -g $gpu_num -w /storage/gby/semseg/horsefine_weights_fcn_RESNET50_8s_crfrnnSP_1ep
python train_gby.py -m fcn_RESNET50_8s_crfrnnSPIO -is 224 -ds horsefine -e $num_ep -bs 1 -g $gpu_num -w /storage/gby/semseg/horsefine_weights_fcn_RESNET50_8s_crfrnnSPIO_1ep
python train_gby.py -m fcn_RESNET50_8s_crfrnnSPAT -is 224 -ds horsefine -e $num_ep -bs 1 -g $gpu_num -w /storage/gby/semseg/horsefine_weights_fcn_RESNET50_8s_crfrnnSPAT_1ep
python train_gby.py -m fcn_RESNET50_8s_crfrnnSPIOAT -is 224 -ds horsefine -e $num_ep -bs 1 -g $gpu_num -w /storage/gby/semseg/horsefine_weights_fcn_RESNET50_8s_crfrnnSPIOAT_1ep

#cp /storage/gby/semseg/horsefine_weights_fcn_RESNET50_8s_crfrnn* /storage/gby/semseg/ep8
#
#python train_gby.py -m fcn_RESNET50_8s_crfrnn -is 224 -ds horsefine -e $num_ep -bs 1 -g $gpu_num -w /storage/gby/semseg/horsefine_weights_fcn_RESNET50_8s_crfrnn_1ep
#python train_gby.py -m fcn_RESNET50_8s_crfrnnSP -is 224 -ds horsefine -e $num_ep -bs 1 -g $gpu_num -w /storage/gby/semseg/horsefine_weights_fcn_RESNET50_8s_crfrnnSP_1ep
#python train_gby.py -m fcn_RESNET50_8s_crfrnnSPIO -is 224 -ds horsefine -e $num_ep -bs 1 -g $gpu_num -w /storage/gby/semseg/horsefine_weights_fcn_RESNET50_8s_crfrnnSPIO_1ep
#python train_gby.py -m fcn_RESNET50_8s_crfrnnSPAT -is 224 -ds horsefine -e $num_ep -bs 1 -g $gpu_num -w /storage/gby/semseg/horsefine_weights_fcn_RESNET50_8s_crfrnnSPAT_1ep
#python train_gby.py -m fcn_RESNET50_8s_crfrnnSPIOAT -is 224 -ds horsefine -e $num_ep -bs 1 -g $gpu_num -w /storage/gby/semseg/horsefine_weights_fcn_RESNET50_8s_crfrnnSPIOAT_1ep
#
#cp /storage/gby/semseg/horsefine_weights_fcn_RESNET50_8s_crfrnn* /storage/gby/semseg/ep9
#
#python train_gby.py -m fcn_RESNET50_8s_crfrnn -is 224 -ds horsefine -e $num_ep -bs 1 -g $gpu_num -w /storage/gby/semseg/horsefine_weights_fcn_RESNET50_8s_crfrnn_1ep
#python train_gby.py -m fcn_RESNET50_8s_crfrnnSP -is 224 -ds horsefine -e $num_ep -bs 1 -g $gpu_num -w /storage/gby/semseg/horsefine_weights_fcn_RESNET50_8s_crfrnnSP_1ep
#python train_gby.py -m fcn_RESNET50_8s_crfrnnSPIO -is 224 -ds horsefine -e $num_ep -bs 1 -g $gpu_num -w /storage/gby/semseg/horsefine_weights_fcn_RESNET50_8s_crfrnnSPIO_1ep
#python train_gby.py -m fcn_RESNET50_8s_crfrnnSPAT -is 224 -ds horsefine -e $num_ep -bs 1 -g $gpu_num -w /storage/gby/semseg/horsefine_weights_fcn_RESNET50_8s_crfrnnSPAT_1ep
#python train_gby.py -m fcn_RESNET50_8s_crfrnnSPIOAT -is 224 -ds horsefine -e $num_ep -bs 1 -g $gpu_num -w /storage/gby/semseg/horsefine_weights_fcn_RESNET50_8s_crfrnnSPIOAT_1ep
#
#cp /storage/gby/semseg/horsefine_weights_fcn_RESNET50_8s_crfrnn* /storage/gby/semseg/ep10
#
##python train_gby.py -m fcn_RESNET50_8s_crfrnn -is 224 -ds horsefine -e $num_ep -bs 1 -g $gpu_num -w /storage/gby/semseg/horsefine_weights_fcn_RESNET50_8s_crfrnn_1ep
##python train_gby.py -m fcn_RESNET50_8s_crfrnnSP -is 224 -ds horsefine -e $num_ep -bs 1 -g $gpu_num -w /storage/gby/semseg/horsefine_weights_fcn_RESNET50_8s_crfrnnSP_1ep
##python train_gby.py -m fcn_RESNET50_8s_crfrnnSPIO -is 224 -ds horsefine -e $num_ep -bs 1 -g $gpu_num -w /storage/gby/semseg/horsefine_weights_fcn_RESNET50_8s_crfrnnSPIO_1ep
##python train_gby.py -m fcn_RESNET50_8s_crfrnnSPAT -is 224 -ds horsefine -e $num_ep -bs 1 -g $gpu_num -w /storage/gby/semseg/horsefine_weights_fcn_RESNET50_8s_crfrnnSPAT_1ep
##python train_gby.py -m fcn_RESNET50_8s_crfrnnSPIOAT -is 224 -ds horsefine -e $num_ep -bs 1 -g $gpu_num -w /storage/gby/semseg/horsefine_weights_fcn_RESNET50_8s_crfrnnSPIOAT_1ep
##
##cp /storage/gby/semseg/horsefine_weights_fcn_RESNET50_8s_crfrnn* /storage/gby/semseg/ep6
##
##python train_gby.py -m fcn_RESNET50_8s_crfrnn -is 224 -ds horsefine -e $num_ep -bs 1 -g $gpu_num -w /storage/gby/semseg/horsefine_weights_fcn_RESNET50_8s_crfrnn_1ep
##python train_gby.py -m fcn_RESNET50_8s_crfrnnSP -is 224 -ds horsefine -e $num_ep -bs 1 -g $gpu_num -w /storage/gby/semseg/horsefine_weights_fcn_RESNET50_8s_crfrnnSP_1ep
##python train_gby.py -m fcn_RESNET50_8s_crfrnnSPIO -is 224 -ds horsefine -e $num_ep -bs 1 -g $gpu_num -w /storage/gby/semseg/horsefine_weights_fcn_RESNET50_8s_crfrnnSPIO_1ep
##python train_gby.py -m fcn_RESNET50_8s_crfrnnSPAT -is 224 -ds horsefine -e $num_ep -bs 1 -g $gpu_num -w /storage/gby/semseg/horsefine_weights_fcn_RESNET50_8s_crfrnnSPAT_1ep
##python train_gby.py -m fcn_RESNET50_8s_crfrnnSPIOAT -is 224 -ds horsefine -e $num_ep -bs 1 -g $gpu_num -w /storage/gby/semseg/horsefine_weights_fcn_RESNET50_8s_crfrnnSPIOAT_1ep
##
##cp /storage/gby/semseg/horsefine_weights_fcn_RESNET50_8s_crfrnn* /storage/gby/semseg/ep7
#
#
#python train_gby.py -is 224 -m fcn_RESNET50_8s_crfrnn -ds horsefine -bs 1 -se 1 -e 1 -vb 1 -ft /storage/gby/semseg/horsefine_weights_fcn_RESNET50_8s_1000ep -g 2
#python train_gby.py -is 224 -m fcn_RESNET50_8s_crfrnnSPIO -ds horsefine -bs 1 -se 1 -e 1 -vb 1 -ft /storage/gby/semseg/horsefine_weights_fcn_RESNET50_8s_1000ep -g 2
#python train_gby.py -is 224 -m fcn_RESNET50_8s_crfrnnSPAT -ds horsefine -bs 1 -se 1 -e 1 -vb 1 -ft /storage/gby/semseg/horsefine_weights_fcn_RESNET50_8s_1000ep -g 3
#python eval_gby.py -is 224 -m fcn_RESNET50_8s -ds horsefine -vb 1 -w /storage/gby/semseg/horsefine_weights_fcn_RESNET50_8s_100ep -g 3
#
#
#python train_gby.py -m fcn_RESNET50_8s_crfrnn -is 224 -ds horsefine -e 1 -bs 1 -g 2 -w /storage/gby/semseg/horsefine_weights_fcn_RESNET50_8s_crfrnn_1ep
#
#
#python train_gby.py -is 224 -m fcn_RESNET50_8s_crfrnn -ds horsefine -bs 1 -se 1 -e 10 -vb 1 -ft /storage/gby/semseg/horsefine_weights_fcn_RESNET50_8s_1000ep -g 3
#python train_gby.py -is 224 -m fcn_RESNET50_8s_crfrnn -ds horsefine -bs 1 -se 1 -e 5 -vb 1 -w /storage/gby/semseg/horsefine_weights_fcn_RESNET50_8s_crfrnn_10ep -g 3

longjob -o run/11_18_crfrnn_no_freeze.txt python train_gby.py -m fcn_RESNET50_8s_crfrnn -is 224 -ds horsefine -e 100 -bs 1 -g 0 -ft /storage/gby/semseg/horsefine_weights_fcn_RESNET50_8s_5000ep -vb 2

longjob -o run/today_fcn224.txt python train_gby.py -m fcn_RESNET50_8s -is 224 -ds horsefine -e 4000 -bs 32 -g 2 -w /storage/gby/semseg/horsefine_weights_fcn_RESNET50_8s_1000ep -vb 2

longjob -o run/11_21_fcn512horsefine.txt python train_gby.py -m fcn_RESNET50_8s -is 512 -ds horsefine -e 1002 -bs 6 -g 3 -vb 2 -w /storage/gby/semseg/horsefine_weights_fcn_RESNET50_8s_2001ep

longjob -o run/20_11_fcn512+dataaug.txt python train_gby.py -m fcn_RESNET50_8s -is 512 -ds horsefine -e 5 -se 325 -bs 1 -g 0 -hf True -ro 10 -w /storage/gby/semseg/horsefine_weights_fcn_RESNET50_8s_1000ep -vb 1


python train_gby.py -m fcn_RESNET50_8s -is 224 -ds horsecoarse -e 10 -bs 32 -g 0 -vb 2

longjob -o run/11_20_fcn512personfine.txt python train_gby.py -m fcn_RESNET50_8s -is 512 -ds personfine -e 200 -bs 6 -g 0 -vb 2

longjob -o run/11_22_fcn512personfine.txt python train_gby.py -m fcn_RESNET50_8s -is 512 -ds personfine -e 201 -bs 6 -g 0 -vb 2 -w /storage/gby/semseg/personfine_weights_fcn_RESNET50_8s_200ep

python predict_gby.py -m fcn_RESNET50_8s -is 512 -nc 25 -w /storage/gby/semseg/personfine_weights_fcn_RESNET50_8s_200ep -im #

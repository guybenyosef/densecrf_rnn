# Complex Relations in a Deep Structured Prediction Model for Fine Image Segmentation
![sample](sample.png)


[comment]: <> This repository contains Keras/Tensorflow code for the "CRF-RNN" semantic image segmentation method, implemented in TensorFlow Keras by https://github.com/sadeepj/crfasrnn_keras


Train:
$ python train_gby.py -m fcn_RESNET50_8s -ds streets -e 1 -bs 32 -vb 1 -is 224

Eval/Test:
$ python eval_gby.py -ds horsecoarse -w /storage/gby/semseg/horsecoarse_weights_fcn_RESNET50_8s_100ep


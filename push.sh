#!/bin/sh

scp -i ~/.ssh/gbar master_run.py models_file.py WGAN_mnist_spectralnorm.py WGAN_weight_celeb.py WGAN_spec_cifar10.py GAN_cifar10.py jobfile.sh FID_score.py WGAN_weight_cifar10.py s173981@transfer.gbar.dtu.dk:~/DeepLearning/

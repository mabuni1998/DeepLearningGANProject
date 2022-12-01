#!/bin/bash

scp -i ~/.ssh/gbar jobfile.sh GAN_mnist.py WGAN_mnist_spectralnorm.py WGAN_mnist_weightclipping.py FID_score.py s173981@transfer.gbar.dtu.dk:~/DeepLearning/

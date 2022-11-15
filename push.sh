#!/bin/bash

scp -i ~/.ssh/gbar jobfile.sh GAN_mnist.py FID_score.py s173981@transfer.gbar.dtu.dk:~/DeepLearning/

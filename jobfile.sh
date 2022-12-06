#!/bin/sh
#BSUB -J deep[1]
#BSUB -e ./err_files/%J-%I.error.txt
#BSUB -o ./log_files/%J-%I.log.txt
#BSUB -q gpua100
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=10GB]"
#BSUB -W 20:00

#Load modules that are used in python-script
source /dtu/sw/dcc/dcc-sw.bash
module load cuda/11.5
module load cudnn/v8.3.2.44-prod-cuda-11.5
source ~/DeepLearning/pytorch_env/bin/activate
echo $LSB_JOBINDEX
#Run script run_gpu.py with inputs
python3 master_run.py mnist gan



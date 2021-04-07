#!/bin/bash

#PBS -N BDRL
#PBS -o JOB_OUT
#PBS -l select=1:ncpus=24:ompthreads=24:mem=2000M
#PBS -l walltime=0:05:00

 cd "${PBS_O_WORKDIR}"

source bdrl/utils/python_env.sh
python main.py -c configs/dqn.gin

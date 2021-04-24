#!/bin/bash -l

#$ -l h_rt=24:00:00   # Specify the hard time limit for the job
#$ -N serialtiming           # Give job a name
#$ -j y               # Merge the error and output streams into a single file
#$ -P ec527
#$ -o serial_timing.csv
#$ -m b

fft_2d_timing

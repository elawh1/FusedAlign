#!/bin/bash
#SBATCH --job-name=job_real_May_24
#SBATCH --output=output/job_real_ay_24.out
#SBATCH --error=error/job_real_May_24.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=350G

export MKL_NUM_THREADS=48 OMP_NUM_THREADS=48 MKL_DYNAMIC=FALSE

DATA_PATH="<REDACTED>/parrot-data"

./../parrot $DATA_PATH/GGI 0
./../parrot $DATA_PATH/acm-dbpl 0
./../parrot $DATA_PATH/arxiv 0
./../parrot $DATA_PATH/dbp15k 0
./../parrot $DATA_PATH/flicker-lastfm 0
./../parrot $DATA_PATH/twitter-foursquare 0
#!/bin/bash
#SBATCH --job-name=job_ER_May_25
#SBATCH --output=output/job_ER_May_25.out
#SBATCH --error=error/job_ER_May_25.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=350G

export MKL_NUM_THREADS=48 OMP_NUM_THREADS=48 MKL_DYNAMIC=FALSE

DATA_PATH="<REDACTED>/graph_generator_parrot"

./../parrot $DATA_PATH/g5k 0
./../parrot $DATA_PATH/g10k 0	
./../parrot $DATA_PATH/g15k 0
./../parrot $DATA_PATH/g20k 0
./../parrot $DATA_PATH/g25k 0
./../parrot $DATA_PATH/g30k 0
./../parrot $DATA_PATH/g35k 0
./../parrot $DATA_PATH/g40k 0
./../parrot $DATA_PATH/g45k 0
./../parrot $DATA_PATH/g50k 0
./../parrot $DATA_PATH/g55k 0
./../parrot $DATA_PATH/g60k 0
./../parrot $DATA_PATH/g65k 0

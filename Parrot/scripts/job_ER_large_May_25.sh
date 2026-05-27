#!/bin/bash
#SBATCH --job-name=job_ER_large_May_25_1
#SBATCH --output=output/job_ER_large_May_25_1.out
#SBATCH --error=error/job_ER_large_May_25_1.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=350G

export MKL_NUM_THREADS=48 OMP_NUM_THREADS=48 MKL_DYNAMIC=FALSE

DATA_PATH="<REDACTED>/graph_generator_parrot"

./../parrot $DATA_PATH/g70k 0
# ./../parrot $DATA_PATH/g75k 0	
# ./../parrot $DATA_PATH/g80k 0
# ./../parrot $DATA_PATH/g85k 0
# ./../parrot $DATA_PATH/g90k 0
# ./../parrot $DATA_PATH/g95k 0
# ./../parrot $DATA_PATH/g100k 0

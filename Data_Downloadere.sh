#!/bin/bash -l        
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --mem=80g
#SBATCH --tmp=20g
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=lee02328@umn.edu


cd /scratch.global/csci8523_group_7/

# Load Miniconda
source /home/boleydl/lee02328/miniconda3/etc/profile.d/conda.sh

conda activate /home/boleydl/lee02328/miniconda3/envs/GenAI
python Data_Downloadere.py
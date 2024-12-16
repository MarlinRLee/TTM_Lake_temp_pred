#!/bin/bash -l        
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --mem=240g
#SBATCH --tmp=20g
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=lee02328@umn.edu
#SBATCH --account=csci8523

cd /scratch.global/csci8523_group_7/

# Load Miniconda
source /home/boleydl/lee02328/miniconda3/etc/profile.d/conda.sh

conda activate /home/boleydl/lee02328/miniconda3/envs/GenAI

conda install tqdm -y

python Data_Downloader.py
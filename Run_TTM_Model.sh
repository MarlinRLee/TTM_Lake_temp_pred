#!/bin/bash -l        
#SBATCH --time=6:00:00
#SBATCH --ntasks=4       
#SBATCH --cpus-per-task=1  
#SBATCH --mem=5G
#SBATCH --tmp=10g
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=lee02328@umn.edu
#SBATCH --account=csci8523
#SBATCH -p a100-4
#SBATCH --gres=gpu:a100:1

cd /scratch.global/csci8523_group_7/

# Load Miniconda
source /home/boleydl/lee02328/miniconda3/etc/profile.d/conda.sh

#conda update -n base -c defaults conda

#conda create --name GenAI2 python=3.10 -y

conda activate GenAI2

#echo Update python

#conda install pandas=2.0.3
#conda install lxml -y

#conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y


#conda install transformers -y


#pip install "tsfm_public[notebooks] @ git+https://github.com/ibm-granite/granite-tsfm.git@v0.2.14"

#conda install torchvision==0.16.1 -c pytorch -c nvidia -y

#python -c "import torch; print(torch.__version__); import torchvision; print(torchvision.__version__)"


python TTM_Zero_Few_Shot.py
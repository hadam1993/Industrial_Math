#!/bin/sh

#SBATCH --partition=gpu
#SBATCH --mem=30000

module load pkgsrc/2019Q1

source ~/Data/sdomke/miniconda3/bin/activate

python KaggleCompetition.py

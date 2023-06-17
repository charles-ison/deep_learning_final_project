#!/bin/bash
#SBATCH -J submix            #job name
#SBATCH -p cascades
#SBATCH -A cascades
#SBATCH --gres=gpu:1		#use one gpu
#SBATCH --nodelist=cn-m-2	#use server 1 or 2
#SBATCH -t 4-00:00:00		#4 day timeout
#SBATCH --export=ALL
#SBATCH -c 8
source activate ai535		#activate env
python slakh-utils/submixes/submixes.py -d slakh-utils/submixes/example_submixes/bass.yaml -i slakh2100_wav_redux/train -t 8
python slakh-utils/submixes/submixes.py -d slakh-utils/submixes/example_submixes/bass.yaml -i slakh2100_wav_redux/validation -t 8
python slakh-utils/submixes/submixes.py -d slakh-utils/submixes/example_submixes/bass.yaml -i slakh2100_wav_redux/test -t 8

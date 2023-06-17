#!/bin/bash
#SBATCH -J convert            #job name
#SBATCH -p cascades
#SBATCH -A cascades
#SBATCH --gres=gpu:1		#use one gpu
#SBATCH --nodelist=cn-m-2	#use server 1 or 2
#SBATCH -t 4-00:00:00		#4 day timeout
#SBATCH --export=ALL
#SBATCH -c 8
source activate ai535		#activate env
python slakh-utils/conversion/flac_converter.py -i slakh2100_flac_redux/test -o slakh2100_wav_redux/test -c False -t 8
python slakh-utils/conversion/flac_converter.py -i slakh2100_flac_redux/train -o slakh2100_wav_redux/train -c False -t 8
python slakh-utils/conversion/flac_converter.py -i slakh2100_flac_redux/validation -o slakh2100_wav_redux/validation -c False -t 8

#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q gpuv100
### -- set the job Name -- 
#BSUB -J ConvLSTMAttAutoLatent128
### -- ask for number of cores (default: 1) -- 
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- specify that we need 30GB of memory per core/slot -- 
#BSUB -R "rusage[mem=100GB]"
#BSUB -R "select[gpu32gb]"
### -- set walltime limit: hh:mm -- 
#BSUB -W 24:00 
### -- set the email address -- 
#BSUB -u s183827@student.dtu.dk
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o Logs/Output_%J_%I.out
#BSUB -e Logs/Output_%J_%I.err

# Load the cuda module
module load cuda/11.6

### Execute script
python3 train_AutoConvAttentionLatent.py > Logs/Out_auto_latent.out

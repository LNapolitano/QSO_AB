#!/bin/bash
#SBATCH -C haswell
#SBATCH -N 10
#SBATCH --qos regular
#SBATCH --account desi
#SBATCH --output SiIV_guadalupe_finders.log
#SBATCH --time=12:00:00
source /project/projectdirs/desi/software/desi_environment.sh master
export PYTHONPATH=$HOME/git/corner:$PYTHONPATH
export PYTHONPATH=/$HOME/git/emcee/src:$PYTHONPATH
srun -N 1 ipython SiIV_AllSteps_HP.py guadalupe full 0 &
srun -N 1 ipython SiIV_AllSteps_HP.py guadalupe full 10 &
srun -N 1 ipython SiIV_AllSteps_HP.py guadalupe full 20 &
srun -N 1 ipython SiIV_AllSteps_HP.py guadalupe full 30 &
srun -N 1 ipython SiIV_AllSteps_HP.py guadalupe full 40 &
srun -N 1 ipython SiIV_AllSteps_HP.py guadalupe full 50 &
srun -N 1 ipython SiIV_AllSteps_HP.py guadalupe full 60 &
srun -N 1 ipython SiIV_AllSteps_HP.py guadalupe full 70 &
srun -N 1 ipython SiIV_AllSteps_HP.py guadalupe full 80 &
srun -N 1 ipython SiIV_AllSteps_HP.py guadalupe full 90 &
wait


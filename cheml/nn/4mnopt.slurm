#!/bin/sh
#SBATCH --time=99:00:00
#SBATCH --job-name="nn"
#SBATCH --output=nn.txt
#SBATCH --clusters=chemistry
#SBATCH --partition=beta
#SBATCH --account=pi-hachmann
#SBATCH --exclusive
#SBATCH --nodes=2

# ====================================================
# For 16-core nodes
# ====================================================
#SBATCH --constraint=CPU-E5-2630v3
#SBATCH --tasks-per-node=1
#SBATCH --mem=64000


echo "SLURM job ID         = "$SLURM_JOB_ID
echo "Working Dir          = "$SLURM_SUBMIT_DIR
echo "Temporary scratch    = "$SLURMTMPDIR
echo "Compute Nodes        = "$SLURM_NODELIST
echo "Number of Processors = "$SLURM_NPROCS
echo "Number of Nodes      = "$SLURM_NNODES
echo "Tasks per Node       = "$TPN
echo "Memory per Node      = "$SLURM_MEM_PER_NODE

ulimit -s unlimited
module load intel-mpi
module load python
module list
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/projects/hachmann/packages/Anaconda:/projects/hachmann/packages/rdkit-Release_2015_03_1:/user/m27/pkg/openbabel/2.3.2/lib
date


echo "Launch job"
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
export I_MPI_FABRICS=shm:tcp

mpirun -np 2 python test.py

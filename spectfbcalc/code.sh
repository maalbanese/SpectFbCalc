#!/bin/bash
#SBATCH --job-name=codexERA5          # Nome del job
#SBATCH --output=output_%j.out      # File di output (%j = job ID)
#SBATCH --error=error_%j.err        # File di errore
#SBATCH --nodes=1                   # Numero di nodi
#SBATCH --ntasks=1                  # Numero di task
#SBATCH --cpus-per-task=4           # CPU per task
#SBATCH --mem=8G                    # Memoria
#SBATCH --time=05:00:00             # Tempo massimo (HH:MM:SS)

source ~/.bashrc
conda activate spectfbcalc

echo "eseguo il codice"
# Esegui il codice
python -u prova.py

echo "finito"
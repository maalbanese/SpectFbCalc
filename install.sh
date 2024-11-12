#!/bin/bash
#CONDADIR=${HOME}/miniconda3
#source $CONDADIR/etc/profile.d/conda.sh

git clone https://github.com/fedef17/ClimTools
cd ClimTools
git pull origin master
cd ..

if command -v mamba &> /dev/null
then
    echo "mamba is installed. Using mamba."
    PACKAGE_MANAGER="mamba"
else
    echo "mamba not found. Using conda."
    PACKAGE_MANAGER="conda"
fi

$PACKAGE_MANAGER env create -n spectfbcalc -f environment.yml



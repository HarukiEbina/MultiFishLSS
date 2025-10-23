#!/bin/bash

export MFISHLSS_BASE=/global/homes/h/hebina/MultiFishLSS/MultiFishLSS
export PYTHONPATH=$MFISHLSS_BASE:$PYTHONPATH

module load python
conda activate myenv

python example_setup.py setup & 
python example_setup.py fs1 &
python example_setup.py fs2 &
python example_setup.py fs3 &
python example_setup.py rec &
python example_setup.py lens1 &
python example_setup.py lens2 &
python example_setup.py lens3 &
# python example_setup.py Alog &
# python example_setup.py Alin &

# ps
# wait
# echo "All done!"
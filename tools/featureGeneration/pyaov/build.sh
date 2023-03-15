# Move the file created by this script to your environment's site-packages directory
# e.g. $HOME/miniforge3/envs/scope-env/lib/python3.10/site-packages/.
python -m numpy.f2py -m aov -c aovconst.f90 aovsub.f90 aov.f90

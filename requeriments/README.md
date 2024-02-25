# Mamitas-Unal Requeriments
## Requeriments for PIP (Pypi)
If you want to install package only using the PIP channel, you must use [requeriments_forPIP.txt](requeriments_forPIP.txt/) file
### Install 
```
pip install -r requeriments/requeriments_forPIP.txt 
```
## Requeriments for conda or miniconda
If you want to use a conda envs, you must use [requeriments_forCondaEnvs.txt](requeriments_forCondaEnvs.txt/) file
### Install
If you want to create a enveiroment with the necessary packages:
```
conda create env --name <name> --file requeriments/requeriments_forCondaEnvs.txt -c defaults conda-forge pypi esri
```
If you only want install the packages use:
```
conda install --file requeriments/requeriments_forCondaEnvs.txt -c defaults conda-forge pypi esri
```
# Warnings
If you have problems with tensorflow-addons package, you need eliminated this package in the requeriments file and use:
```
pip install tensorflow-addons[tensorflow]
```
and then use (without tensoflow-addons package):
```
conda install --file requeriments/requeriments_forCondaEnvs.txt -c defaults conda-forge pypi esri
```
or 
```
pip install -r requeriments/requeriments_forPIP.txt 
```

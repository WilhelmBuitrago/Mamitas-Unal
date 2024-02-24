# Mamitas-Unal
## Requeriments for PIP (Pypi)
If you want to install package using only the PIP channel, you should use [instalación de AlphaFold](requeriments/requeriments_forPIP.txt/) file
### Install 
```
pip install r requeriments/requeriments_forPIP.txt 
```
## Requeriments for conda or miniconda
If you have been use a conda envs, you should use [requeriments_forCondaEnvs.txt](requeriments/requeriments_forCondaEnvs.txt/) file
### Install
If you want to create a enveiroment with the need it packages:
```
conda create env --name <name> --file requeriments/requeriments_forCondaEnvs.txt -c defaults conda-forge pypi esri
```
If you want only install the packages use:
```
conda install --file requeriments/requeriments_forCondaEnvs.txt -c defaults conda-forge pypi esri
```
### Warnings
If you have problems with tensorflow-addons package, you need eliminated this packages for the requeriments file and use:
```
pip install tensorflow-addons
```
and
```
conda install --file requeriments/requeriments_forCondaEnvs.txt -c defaults conda-forge pypi esri
```

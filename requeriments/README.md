# Mamitas-Unal Requeriments
## 1. Creation the environment
You need have conda or miniconda with Python 3.10.12 version, for use this project then for create a environment you must use:
```
conda create --name <name> python==3.10.12
```
## 2. Requeriments with PIP (Pypi)
For install package using the PIP channel, you must use [requeriments_forPIP.txt](requeriments_forPIP.txt/) file
### Install 
```
pip install -r requeriments/requeriments_forPIP.txt 
```
And then for tensorflow-addons use:
```
pip install tensorflow-addons[tensorflow]
```

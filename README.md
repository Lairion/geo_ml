# geo_ml

# How to start Python solution

## Install Python
Download and install latest Python version on https://www.python.org/

## Install package virtualenv

```
pip install virtualenv
```

## Install packeges for project

Create environment for python packeges
```
cd <root_project>/pysolution/
virtualenv env

```
Activate environment
### For Windows
```
env\Scripts\activate

```
### For unix
```
source env\scripts\activate

```
Run packages installation
### For Windows
```
windows_install.bat

```
### For Unix
```
pip install -r req.txt

```

## Raw data for python solution

Folder structure

```
.
└── data/
    ├── check_pictures/
    │   ├── <class_name>/
    │   │   └── <your_png>.png
    │   └── ...
    ├── satelit/
    │   ├── <class_name>/
    │   │   └── <your_png>.png
    │   └── ...
    └── map/
        ├── <geodata>.shp
        ├── <geodata>.shx
        └── <geodata>.dbf

```

## Usage
Run next python script and choose option for run
```
python <root_project>/pysolution/__main__.py
or
python <root_project>/pysolution

```

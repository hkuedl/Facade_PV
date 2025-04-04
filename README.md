# Facade_PV

_This work conducts a comprehensive study on harnessing the carbon mitigation potential of facade photovoltaics (FPV). We select 102 large cities in China for the study. We first reveal the high power generation potential of FPV compared with rooftop photovoltaics (RPV), and then we determine the cost-effective deployment pathway for FPV in 2030-2050._

Codes for submitted paper "Mitigating Carbon Emissions in China’s Large Cities with Facade Photovoltaics".

Authors: Xueyuan Cui, XXXX.

## Requirements
[Python](https://www.python.org/)

Version: 3.8.17

Required libraries include ``osgeo``, ``fiona``, ``shapely``, ``pyproj``, ``scipy``, ``geopandas``, ``sklearn``, ``pyomo``, etc.

[QGIS](https://qgis.org/)

Version: 3.36.3

Plug-in: [UMEP](https://umep-docs.readthedocs.io/en/latest/index.html)

## Data

Sources of required data and outputs of the study are publicly available in [Baidu Netdisk](https://pan.baidu.com/s/1nz6OqH5hKpRSR72fxIoMdQ?pwd=jx8f) (Code: jx8f)

### Introduction

The formats and applications of datasets are introduced in the ``#Introduction.txt`` in each corresponding subfolder.

## Codes
### Reproduction
To reproduce the experiments of the proposed methods and generate the figures, please go to folders
```
cd #Codes/Evaluation
cd #Codes/Optimization
cd #Codes/Figures
```
respectively.

The ``Evaluation`` folder covers the procedure for solar irradiance simulation and power generation potential in the study, and the ``Optimization`` folder covers the procedure of the optimal deployment model for FPV's development pathway. The ``Figures`` folder includes the codes for generating the figures in the manuscript.

The introduction on the running order and each file's function is explained in ```Readme.md``` in the sub-folder in ``#Codes``.

## Citation
```
```

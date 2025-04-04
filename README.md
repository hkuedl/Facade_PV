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

_1. Evaluation_

_1.1 UMEP_input_

It includes the required data for calculating solar irradiance based on UMEP. All data for the selected 30 cities are zipped with .zip format. The data serve as the input source for the code of ```#Codes/Evaluation/UMEP_QGIS```;

_1.2 Sampled_input_output_

It includes the required data for training the regression algorithm. All data are divided into .npy and .xlsx formats as the input features and labels. The data are generated by running the code of ```#Codes/Evaluation/Sample_label_read_.py``` and ```#Codes/Evaluation/Sample_feature_read_.py```, and they are used as the input source for the code of ```#Codes/Evaluation/ML_training.py```;

_1.3 Regression_model_

It includes the trained model that extrapolates the results of facade irradiance, rooftop irradiance, facade power, and facade capacity, respectively. The models are generated by running the code of ```#Codes/Evaluation/ML_training.py```;

_1.4 All_input_

It includes all required inputs of the trained model for extrapolation. All feature metrics for the 102 cities are in the .npy format. The models are generated by running the code of ```#Codes/Evaluation/ML_input_read.py```, and they serve as the input source for the code of ```#Codes/Evaluation/ML_output_power.py``` and ```#Codes/Evaluation/ML_output_capacity.py```;

_1.5 ALl_output_

It includes all the output data that are extrapolated by the trained model. A more detailed introduction of each type of data is presented in ```#Introduction.txt``` in the folder.

_2. Optimization_

_2.1 Electricity_price_

It includes all the required data on electricity prices in all provinces (2024).

_2.2 Loads_

It includes all the required data to generate load profiles of grid cells. The data are used as the input source for the code of ```#Codes/Optimization/Load_generate.py```;

_2.3 Other_parameters_

It includes all other required data information for the optimization problem, including the land use, climate zones, etc.

All data in 2.1-2.3 serve as the input for the optimization problem when running the code of ```#Codes/Optimization/Multi_period_planning.py```;

_2.4 Fig_2_3_data_

It includes all the required data to generate Figures 2-3 in the manuscript, and the corresponding codes are in ```#Codes/Figures/Fig_2(3).ipynb```.

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

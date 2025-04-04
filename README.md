# Facade_PV
Codes for Paper "Mitigating Carbon Emissions in China’s Large Cities with Facade Photovoltaics".

_This work conducts a comprehensive study on harnessing the carbon mitigation potential of facade photovoltaics (FPV). We select 102 large cities in China for study. We first reveal the high power generation potential of FPV compared with rooftop photovoltaics (RPV), then we determine the cost-effective deployment pathway for FPV in 2030-2050._

Codes for submitted Paper "A Smart Model-then-Control Strategy for the Scheduling of Thermostatically Controlled Loads".

Authors: Xueyuan Cui, Boyuan Liu, Yehui Li, and Yi Wang.

## Requirements
Python version: 3.8.17

The must-have packages can be installed by running
```
pip install requirements.txt
```

## Experiments
### Data
All the data for experiments can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1wB3OkMHw7XF4DA5wYUdxXeCu_GbcM-Cv?usp=sharing).

### Reproduction
To reproduce the experiments of the proposed methods and comparisons for single-zone, 22-zone, and 90-zone buildings, please go to folders
```
cd #Codes/Single-zone
cd #Codes/22-zone
cd #Codes/90-zone
```
respectively. The introduction on the running order and each file's function is explained in ```Readme.md``` in the folder.

Note: There is NO multi-GPU/parallelling training in our codes. 

The required models as the warm start of SMC are saved in ```#Results```.

## Citation
```
```

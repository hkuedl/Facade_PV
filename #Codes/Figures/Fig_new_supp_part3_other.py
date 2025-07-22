#%% Building_dataset_compasion
import geopandas as gpd
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
import pyproj
from shapely.ops import transform
from functools import partial

# Read data files
shp1 = gpd.read_file(r"Beijing_new_with_height\beijing.shp")

shp2 = gpd.read_file(r"Beijing_new_without_height\Beijing.shp")

shp3 = gpd.read_file(r"Beijing_90city\Beijing.shp")

shp4 = gpd.read_file(r"Beijing_major\Beijing.shp")

# Define projection transformation function
proj_wgs84 = pyproj.CRS('EPSG:4326')  # WGS84 coordinate system
proj_utm = pyproj.CRS('EPSG:32650')  # UTM coordinate system, UTM 50N for Beijing region

project = partial(
    pyproj.Transformer.from_crs(proj_wgs84, proj_utm, always_xy=True).transform
)

# Filter out None values and transform coordinates to calculate area
area_shp1 = [transform(project, geom).area for geom in shp1['geometry'] if geom is not None]
area_shp2 = [transform(project, geom).area for geom in shp2['geometry'] if geom is not None]
area_shp3 = shp3['Shape_Area'].tolist()
area_shp4 = [transform(project, geom).area for geom in shp4['geometry'] if geom is not None]

# Calculate height
height_shp1 = shp1['Height'].tolist()
height_shp4 = shp4['height'].tolist()

# Calculate volume
volume_shp1 = [(transform(project, geom).area * height) for geom, height in zip(shp1['geometry'], shp1['Height']) if geom is not None]
volume_shp4 = [(transform(project, geom).area * height) for geom, height in zip(shp4['geometry'], shp4['height']) if geom is not None]

# Filter out NaN values
def filter_nan(data):
    return [x for x in data if not np.isnan(x)]

# Custom ScalarFormatter to set font size for scientific notation
class FixedScalarFormatter(ScalarFormatter):
    def __init__(self, useMathText=True, **kwargs):
        super().__init__(useMathText=useMathText, **kwargs)
        self.set_powerlimits((0, 0))
        self.set_useOffset(False)
        self.set_useMathText(False)

    def _set_format(self):
        self.format = "%1.1f"
        self._useMathText = False

    def _set_offset(self):
        self.offset = ""

# Ensure the target folder exists
output_dir = r"picture"
os.makedirs(output_dir, exist_ok=True)

# Plot and save each histogram
def save_histogram(data, bins, color, xlabel, title, filename_prefix, max_x, max_value=None):
    data = filter_nan(data)
    if max_value is not None:
        data = [x for x in data if x <= max_value]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.hist(data, bins=bins, color=color, alpha=0.7)
    ax.set_xlabel(xlabel, fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.set_xlim(0, max_x)
    ax.tick_params(axis='y', labelsize=24)  # Retain y-axis ticks
    ax.set_yticks(ax.get_yticks())  # Retain y-axis ticks
    ax.set_yticklabels([])  # Hide y-axis labels
    ax.set_title(title, fontsize=24)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{filename_prefix}.pdf'), format='pdf', dpi=600)
    fig.savefig(os.path.join(output_dir, f'{filename_prefix}.png'), dpi=600)
    plt.close(fig)

# Building area distributions
save_histogram(area_shp1, bins=20, color='skyblue', xlabel='Area (m²)', title='Dataset-1', filename_prefix='Dataset_1a', max_x=2000, max_value=np.percentile(area_shp1, 99))
save_histogram(area_shp2, bins=20, color='green', xlabel='Area (m²)', title='Dataset-2', filename_prefix='Dataset_1b', max_x=2000, max_value=np.percentile(area_shp2, 99))
save_histogram(area_shp3, bins=20, color='red', xlabel='Area (m²)', title='Dataset-3', filename_prefix='Dataset_1c', max_x=2000, max_value=np.percentile(area_shp3, 99))
save_histogram(area_shp4, bins=20, color='purple', xlabel='Area (m²)', title='Dataset-4', filename_prefix='Dataset_1d', max_x=2000, max_value=np.percentile(area_shp4, 99))

# Height distributions
save_histogram(height_shp1, bins=20, color='skyblue', xlabel='Height (m)', title='Dataset-1', filename_prefix='Dataset_2a', max_x=30, max_value=np.percentile(height_shp1, 99))
save_histogram(height_shp4, bins=20, color='purple', xlabel='Height (m)', title='Dataset-4', filename_prefix='Dataset_2b', max_x=30, max_value=np.percentile(height_shp4, 99))

# Volume distributions
save_histogram(volume_shp1, bins=20, color='skyblue', xlabel='Volume (m³)', title='Dataset-1', filename_prefix='Dataset_3a', max_x=25000, max_value=np.percentile(volume_shp1, 99))
save_histogram(volume_shp4, bins=20, color='purple', xlabel='Volume (m³)', title='Dataset-4', filename_prefix='Dataset_3b', max_x=25000, max_value=np.percentile(volume_shp4, 99))

#%%   Distribution of training labels
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import cross_validate, KFold, train_test_split
import matplotlib.font_manager as fm
from matplotlib import rcParams
font_path = 'arial.ttf'
custom_font = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)
rcParams['font.family'] = custom_font.get_name()

def re_input_new(i_lab,Feas_input,Feas_read_info):
    #indices_zero = np.where(Feas_input[:,11] == 0)[0]
    indices_non_zero = np.where(Feas_input[:,11] != 0)[0]
    if i_lab >= 1:
        i_dele = [3]
        Feas_input_new = np.delete(Feas_input[indices_non_zero,:], i_dele, axis=1)
    if i_lab == 0:
        Feas_input_new = Feas_input[:,[0,1,3,16,17,18,19,20]]
        Feas_input_new[:,2] = Feas_read_info[:,0]
    return Feas_input_new

save_ML_file = ''
path_save = 'ALL_sample_results'
city_path = 'ALL_102_cities/'
ALL_city = pd.read_excel('Fig_input_data/Climate_WWR.xlsx',sheet_name = 'Climate').iloc[:,1].tolist()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
set_seed(20)

City_name = ['Beijing', 'Changchun', 'Changsha', 'Chengdu', 'Chongqing', 'Fuzhou', 'Guangzhou', 'Guiyang', 'Haerbin', 'Haikou', 'Hangzhou', 'Hefei', 'Huhehaote', 'Jinan', \
             'Kunming', 'Lanzhou', 'Nanchang', 'Nanjing', 'Nanning', 'Shanghai', 'Shenyang', 'Shijiazhuang', 'Taiyuan', 'Tianjin', 'Wuhan', 'Wulumuqi', 'Xian', 'Xining', 'Yinchuan', 'Zhengzhou']

day_list = ['0115','0215','0315','0415','0515','0615','0715','0815','0915','1015','1115','1215']
day_days = [0,31,28,31,30,31,30,31,31,30,31,30,31]
time_list = ['08','10','12','14','16','18']

City_sele = [_ for _ in range(0,30)] #+ [_ for _ in range(20,30)]

Feas_read_fea = []
Feas_read_lab = []
Feas_read_inf = []
Feas_read_eff_area = []

for cc in City_sele:
    city_name = City_name[cc]
    Feas_read_fea.append(np.load(path_save+'/'+city_name+'_ML_features.npy')[:,1:22])
    Feas_read_lab.append(np.load(path_save+'/'+city_name+'_ML_features.npy')[:,25:]/1e3)
    Feas_read_inf.append(np.load(path_save+'/'+city_name+'_ML_features.npy')[:,23:24]/1e6)
    # _,_,_,_,df_read_facade_area,_,_,_ = Functions.read(path_save,city_name)
    # Feas_read_eff_area.append(df_read_facade_area)

# 1sample, 
# 21fea:lon, lat, Fea_density,Fea_coverage,Fea_Mhei,Fea_SDhei,Fea_SDarea,Fea_complexity,Fea_compact,Fea_number,Fea_mean_outdoor,Fea_12_ratio,Fea_skew,Fea_WWR,'can_avg','can_std','date','time','Global','Diff','Direct'
# 3info,3labels
Feas_input = np.vstack((Feas_read_fea))
Feas_output = np.vstack((Feas_read_lab))
Feas_read_inf = np.vstack((Feas_read_inf))
print("Array contains NaN:",np.isnan(Feas_output).any())
if np.isnan(Feas_output).any():
    Feas_input = np.nan_to_num(Feas_input, nan=0.0)
    Feas_output = np.nan_to_num(Feas_output, nan=0.0)

#0: roof; 1: facade ideal;
i_lab = 1

print('Sample numbers')
print(Feas_input.shape[0]/72)

pcc_fea_lab = []
for i in range(2,14):
    pcc_fea_lab.append(np.corrcoef(Feas_input[:,i], Feas_output[:,i_lab])[0,1])
pccs_fea_fea = np.corrcoef(Feas_input[:,2:14].T)

Feas_input = re_input_new(i_lab,Feas_input,Feas_read_inf)

X_train, X_test, y_train, y_test = train_test_split(Feas_input, Feas_output[:,i_lab:i_lab+1], test_size=0.2, random_state=20)
fig = plt.figure(figsize=(8, 6))
plt.hist(y_train, bins=100, color='steelblue', edgecolor='k', alpha=0.7)

plt.xlim(0,200)
plt.xlabel('Label values (MW)',fontsize = 20)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
fig.savefig('Figs_new_supp/sOther_label_distribution.pdf', format='pdf', dpi=600, bbox_inches='tight')
fig.savefig('Figs_new_supp/sOther_label_distribution.png', dpi=600, bbox_inches='tight')
plt.show()

#%% Selection_of_zone_type
import json
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, shape, MultiPolygon
from tqdm import tqdm
import numpy as np
from matplotlib.patches import Patch
from geopy.distance import geodesic
from scipy.io import savemat,loadmat

path = ''
city_path = 'ALL_102_cities/'
path_type = 'Power'
path_cap = 'Capacity'
City_statistic = pd.read_excel(path+'City_statistic.xlsx',sheet_name = 'Class_Volume', index_col=0)
Statis_all = pd.read_excel(path+'City_statistic.xlsx',sheet_name = 'Key_information', index_col=0)

#3: Beijing; 74: Wuhan; 63: Suzhou; 92：Changchun; 56: Shanghai； 15: Guangzhou；59：shenzhen；60：shenyang
cc = 3
city_name = City_statistic.index[cc]
print(city_name)

if cc in [97,92,2,6,8,69]:
    data_path = '#Opt_results/'+city_name+'_hybrid_n.mat'
else:
    data_path = '#Opt_results/'+city_name+'_hybrid.mat'

C1 = path_cap+'/Cap_facade_'+city_name+'.npy'
C2 = path_cap+'/Cap_roof_'+city_name+'.npy'

G_type = np.load('#ML_results/Grid_type/'+'Grid_type_'+city_name+'.npy')
Feas_read_sta = np.load(city_path+city_name+'_ALL_Featuers.npy')[:,[i for i in range(14)]+[15,16]]
indices_non_zero = np.where(Feas_read_sta[:,11] != 0)[0]
WWR = np.load(city_path+city_name+'_ALL_Featuers.npy')[indices_non_zero,13:14]
N_grid = G_type.shape[0]
N_gg = np.where(G_type[:,0] != 888)[0]

Data = loadmat(data_path)
R_Cap_r = Data['R_Cap_r']  #np.zeros((N_grid, 2, Y))
R_Cap_s = Data['R_Cap_s']  #np.zeros((N_grid, 2, Y))
R_Cap_f = Data['R_Cap_f']  #np.zeros((N_grid, 3, Y))
R_Ele = Data['R_Ele']      #np.zeros((N_grid, Y))
R_Pow = Data['R_Pow']      #np.zeros((N_grid, Y, D, T))
R_Pow_f = Data['R_Pow_f']  #np.zeros((N_grid,Y, D, T))
R_Pow_ch = Data['R_Pow_ch']  #np.zeros((N_grid,2,Y, D, T))
R_Pow_dis = Data['R_Pow_dis']  #np.zeros((N_grid,2,Y, D, T))
R_Pow_G = Data['R_Pow_G']     #np.zeros((N_grid,2,Y, D, T))
R_Pow_r = Data['R_Pow_r']  #np.zeros((N_grid, 2, Y, D, T))
R_Pow_Buy = Data['R_Pow_Buy']  #np.zeros((N_grid,2,Y, D, T))
R_Pow_AB = Data['R_Pow_AB']   #np.zeros((N_grid,2,Y, D, T))
R_Car = Data['R_Car']     #np.zeros((N_grid,2,Y))
R_Car_t = Data['R_Car_t']  #np.zeros((N_grid,2,Y,D,T))

th_hh = 18
th_std = [0,0]  #[np.percentile(G_type[N_gg,2], 50),np.percentile(G_type[N_gg,2], 50)]     #4  # 3.6#4[3.6,3.6]  # 
th_aa = [200,200]  # [np.percentile(G_type[N_gg,3], 50),np.percentile(G_type[N_gg,3], 50)]   #200 #120  # 55#120[120,120]  #  
list_form = [list(np.where((G_type[N_gg,1]>=th_hh)&(G_type[N_gg,2]>=th_std[1])&(G_type[N_gg,3]>=th_aa[1]))[0]),\
       list(np.where((G_type[N_gg,1]>=th_hh)&(G_type[N_gg,2]>=th_std[1])&(G_type[N_gg,3]<th_aa[0]))[0]),\
       list(np.where((G_type[N_gg,1]<th_hh)&(G_type[N_gg,2]>=th_std[0])&(G_type[N_gg,3]>=th_aa[1]))[0]),\
       list(np.where((G_type[N_gg,1]<th_hh)&(G_type[N_gg,2]>=th_std[0])&(G_type[N_gg,3]<th_aa[0]))[0])]


s_font,s_title = 16,14
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), gridspec_kw={'hspace': 0.2})
sorted_indices = np.argsort(G_type[:,1])
ax1.plot(G_type[sorted_indices,1])
ax1.scatter(2000,G_type[sorted_indices,:][2000,1],color = 'red',s = 100,marker = 'o')
ax1.set_xlabel('')
ax1.set_ylabel('Average height (m)',fontsize = s_font)
ax1.xaxis.set_tick_params(labelsize=s_font-2)
ax1.yaxis.set_tick_params(labelsize=s_font-2)
sorted_indices = np.argsort(G_type[:,3])
ax2.plot(G_type[sorted_indices,3])
ax2.scatter(2000,G_type[sorted_indices,:][2000,3],color = 'red',s = 100,marker = 'o')
ax2.set_xlabel('Index of grid cell',fontsize = s_font)
ax2.set_ylabel('Building number',fontsize = s_font)
ax2.xaxis.set_tick_params(labelsize=s_font-2)
ax2.yaxis.set_tick_params(labelsize=s_font-2)
fig.savefig('Figs_new_supp/sOther_zone_threshold.pdf', format='pdf', dpi=600, bbox_inches='tight')
fig.savefig('Figs_new_supp/sOther_zone_threshold.png', dpi=600, bbox_inches='tight')
plt.show()
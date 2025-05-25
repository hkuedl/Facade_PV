#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap

ratio_plot = 3

figwidth =  8.5 * ratio_plot
fs = 8 * ratio_plot
lw = 0.4 * ratio_plot
lw2 = 0.75 * ratio_plot
lw3 = 1 * ratio_plot

grid_alpha = 0.5

# ------------------- Read area and capacuty data -------------------
# Corresponds to roof area, maximum facade area, and filtered facade area

area_df = pd.read_excel('Fig_input_data/City_statistic.xlsx', sheet_name='Key_information',
        usecols=[0, 3, 4, 5], index_col=[0])

cap_df = pd.read_excel('Fig_input_data/City_Cap.xlsx', index_col=[0])

area_data = area_df.to_numpy()
cap_data = cap_df.to_numpy()

population_df = pd.read_excel('Fig_input_data/City_info.xlsx', sheet_name=0,
        usecols=list(np.arange(8)))  # The last column 'City' is the index of the cities in the study

population_df.set_index('City', inplace=True)

scale_mapping = {
    "II型大城市": 0,
    "I型大城市": 1,
    "特大城市": 2,
    "超大城市": 3
}

population_df["city_level"] = population_df["规模等级"].map(scale_mapping)

file_path = 'Fig_input_data/City_statistic.xlsx'  # Path to the Excel file
building_volume_df = pd.read_excel(file_path, index_col=0)
total_building_volume = building_volume_df.iloc[:, 0].copy()
key_information = pd.read_excel(file_path, sheet_name=2, index_col=0)
total_building_area = key_information['Area_roof-0(km2)']
average_building_height = total_building_volume / total_building_area * 1000
average_building_height.name = 'average_building_height'

# ------------------- Read power curve data -------------------
# Read .npy files
file_path = '.'
ideal_facade_data_1_path='Fig_input_data/Power_facade_ideal_1.npy'
ideal_facade_data_1 = np.sum((np.load(ideal_facade_data_1_path)),axis=1)/1e6

real_facade_data_1_path='Fig_input_data/Power_facade_1.npy'
real_facade_data_1 = np.sum((np.load(real_facade_data_1_path)),axis=1)/1e6

roof_data_1_path = 'Fig_input_data/Power_roof_1.npy'
roof_data_1 = np.sum(np.load(roof_data_1_path),axis=1)/1e6

cities_hku_dest = pd.read_hdf('Fig_input_data/cities_hku_dest.hdf', key='cities_hku_dest')

import os

# Specify the file path
file_path = "Fig_input_data/City_adcode.xlsx"

# Ensure the file exists
if os.path.exists(file_path):
    # Read the Excel data
    city_adcode_df = pd.read_excel(file_path, engine='openpyxl', index_col=0)
    print("Data loaded successfully!")
    print(city_adcode_df.head())  # Display the first few rows of the data
else:
    print(f"The file {file_path} does not exist. Please check if the path is correct.")

result_folder = r'Fig_input_data'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
hdf_file = os.path.join(result_folder, 'Power_roof_ideal_1_sum_all_df.h5')

# If the HDF file already exists, read the data directly; otherwise, perform data processing and save the result
Power_roof_ideal_1_sum_all_df = pd.read_hdf(hdf_file, key='df')
print("Data read from the HDF file.")

Power_roof_ideal_1_sum_all_df

result_folder = r'Fig_input_data/'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
hdf_file = os.path.join(result_folder, 'Power_facade_ideal_1_sum_all_df.h5')

Power_facade_ideal_1_sum_all_df = pd.read_hdf(hdf_file, key='df')
print("Data read from the HDF file.")

Power_facade_ideal_1_sum_all_df


result_folder = r'Fig_input_data/'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
hdf_file = os.path.join(result_folder, 'Grid_feas_wea_all_df.h5')

Grid_feas_wea_all_df = pd.read_hdf(hdf_file, key='df')
print("Data read from the HDF file.")

solar_df = Grid_feas_wea_all_df.groupby(level=0).mean()/1000  # kWh/m²

Grid_feas_wea_all_df

df_plot = area_df.iloc[:, :2].copy()  # km^2
roof_power = roof_data_1  # TWh
facade_power = ideal_facade_data_1  # TWh
df_plot['roof_power'] = roof_power
df_plot['facade_power'] = facade_power

df_plot = df_plot.merge(average_building_height, left_index=True, right_index=True)

df_plot['facade_roof_ratio_area'] = df_plot['Area_facade-0(km2)'] / df_plot['Area_roof-0(km2)']
df_plot['facade_roof_ratio_generation'] = df_plot['facade_power'] / df_plot['roof_power']

import Functions
Carbon_evaluation = np.zeros((102, 1))
df_plot['facade_carbon'] = facade_power
for i in range(len(df_plot)):
    city_name = df_plot.index[i]
    _,K3_TOU_indu_i,K3_TOU_resi_i,K3_net_i,Carbon_F = Functions.TOU_period(city_name)
    df_plot.iloc[i,-1] = facade_power[i]*Carbon_F   #Million ton
    Carbon_evaluation[i,0] = Carbon_F*facade_power[i]

df_plot = df_plot.merge(city_adcode_df, left_index=True, right_index=True, how='left')

df_plot['city_adcode'] = df_plot['city_adcode'].astype(int)

df_plot

import geopandas as gpd
import pyproj
file_path_taiwan = 'Fig_input_data/台湾矢量地图shp数据/'
taiwan = gpd.read_file(file_path_taiwan + '台湾省-市矢量shp.shp').to_crs(epsg=4326)

proj_aeqd = (
    "+proj=aeqd "
    "+lat_0=35 "   # Central latitude
    "+lon_0=105 "  # Central longitude
    "+datum=WGS84 "
    "+units=m "
    "+no_defs "
)
aeqd_crs = pyproj.CRS.from_proj4(proj_aeqd)

taiwan_aeqd = taiwan.to_crs(aeqd_crs)


#%
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

import frykit.plot as fplt
import frykit.shp as fshp

# Parameter settings
col_plot = "facade_carbon"

cmap = plt.get_cmap('YlGnBu')
bad_color = '#E1E1E1'
cmap.set_bad(bad_color)

lw_cn_map = lw * 0.1
lw_axis = lw * 0.5

# Data preparation
city_table = fshp.get_cn_city_table(data_source='tianditu')  # Retrieve city boundary metadata
city_table_with_values = city_table.merge(df_plot, left_on='city_adcode', right_on='city_adcode', how='left')  # Merge by city_adcode

city_adcode = city_table_with_values['city_adcode'].astype(int)
cities = fshp.get_cn_city(city_adcode, data_source='tianditu')
data = city_table_with_values[col_plot]

vmin, vmax = np.floor(data.min() * 1) / 1, np.ceil(data.max() * 1) / 1
norm = plt.Normalize(vmin=vmin, vmax=vmax)

# Set map projection
map_crs = fplt.CN_AZIMUTHAL_EQUIDISTANT
data_crs = fplt.PLATE_CARREE

# Set tick marks
xticks = np.arange(-180, 181, 10)
yticks = np.arange(-90, 91, 10)

# Prepare main map
#font_options = {'family': 'Arial', 'size': fs}
font_options = {'size': fs}
plt.rc('font', **font_options)

fig = plt.figure(figsize=np.array([figwidth * 2/3, figwidth * 2/3 * 0.7]) / 2.5)
main_ax = fig.add_subplot(projection=map_crs)

fplt.set_map_ticks(main_ax, (74, 136, 17, 55), xticks, yticks)
main_ax.gridlines(xlocs=xticks, ylocs=yticks, lw=0.5, ls='--', color='gray')
main_ax.set_adjustable('datalim')  # Adjust xlim and ylim to fit data
main_ax.tick_params(axis='x', labelsize=10)  
main_ax.tick_params(axis='y', labelsize=10)  
#main_ax.set_xticks([])
#main_ax.set_yticks([])
#main_ax.axis('off')

# Prepare inset map
mini_ax = fplt.add_mini_axes(main_ax, shrink=0.35)
mini_ax.spines[:].set_linewidth(lw_axis)
mini_ax.set_extent((105, 122, 2, 25), data_crs)
mini_ax.gridlines(xlocs=xticks, ylocs=yticks, lw=0.5, ls='--', color='gray')


# Add map features
for ax in [main_ax, mini_ax]:
    fplt.add_cn_border(ax, lw=lw_cn_map, fc='none', zorder=-10)
    fplt.add_cn_line(ax, lw=lw_cn_map)

# Draw filled polygons
for ax in [main_ax, mini_ax]:
    fplt.add_geometries(
        ax, cities, array=data,
        cmap=cmap, norm=norm,
        ec='black', lw=lw_cn_map,
    )
    taiwan_aeqd.plot(
        ax=ax, color='none', edgecolor='black',
        linewidth=lw_cn_map,
        )

# Set title
main_ax.set_title(
    'Annual carbon mitigation potential of FPV',
    y=0.92,
    fontsize=fs-9,
    weight='normal',
    fontweight='bold'
)


# --------------------- Create colorbar ---------------------
# Main colorbar
sm = ScalarMappable(cmap=cmap, norm=norm)
sm._A = []
cax = fig.add_axes([0.08 + 0.06, 0.12, 0.25, 0.03])  # Add space on the left for NaN block
cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')
cbar.set_ticks(np.linspace(vmin, vmax, 2))
cbar.ax.tick_params(labelsize=fs - 9)
cbar.outline.set_visible(True)
cbar.outline.set_linewidth(lw_axis)

# Align tick labels: left-aligned for min, right-aligned for max
tick_labels = cbar.ax.get_xticklabels()
tick_labels[0].set_horizontalalignment('left')
tick_labels[-1].set_horizontalalignment('right')

# Add NaN color block (gray)
na_cax = fig.add_axes([0.08, 0.12, 0.045, 0.03])  # Position of mini colorbar
na_cax.set_facecolor(bad_color)
na_cax.set_xticks([])
na_cax.set_yticks([])
na_cax.spines[:].set_linewidth(lw_axis)
na_cax.text(0.5, -1.3, 'N/A', ha='center', va='center', fontsize=fs - 9, transform=na_cax.transAxes)

# Add unit label
main_ax.text(
    0.08, 0.12 + 0.03 + 0.01,  # Position: just above the NaN block
    '(Million ton)',                   # Annotation text
    transform=fig.transFigure,
    ha='left', va='bottom', fontsize=fs - 8
)

# Adjust layout
plt.subplots_adjust(left=0.04, right=0.96, bottom=0.04, top=0.96)
#plt.tight_layout(rect=[0, 0, 0.9, 0.9])  # rect参数控制图形范围（左, 下, 右, 上）
fig.savefig('Figs_new/Fig1b_left.pdf',format='pdf',dpi=600,bbox_inches='tight')
fig.savefig("Figs_new/Fig1b_left.png", dpi=600,bbox_inches='tight')
plt.show()


#%

path = '#ML_results/'
Statis_all = pd.read_excel(path+'/City_statistic.xlsx',sheet_name = 'Key_information', index_col=0)

City_SLC_list = []
indices = Statis_all.index[Statis_all.iloc[:,-1] == ['超大城市','特大城市','I型大城市','II型大城市'][0]]
for index in indices:
    City_SLC_list.append(Statis_all.index.get_loc(index))

City_VLC_list = []
indices1 = Statis_all.index[Statis_all.iloc[:,-1] == ['超大城市','特大城市','I型大城市','II型大城市'][1]]
for index in indices1:
    City_VLC_list.append(Statis_all.index.get_loc(index))

City_LCI_list = []
indices2 = Statis_all.index[Statis_all.iloc[:,-1] == ['超大城市','特大城市','I型大城市','II型大城市'][2]]
for index in indices2:
    City_LCI_list.append(Statis_all.index.get_loc(index))

City_LCII_list = []
indices3 = Statis_all.index[Statis_all.iloc[:,-1] == ['超大城市','特大城市','I型大城市','II型大城市'][3]]
for index in indices3:
    City_LCII_list.append(Statis_all.index.get_loc(index))

data_list = [Carbon_evaluation[City_SLC_list,0],Carbon_evaluation[City_VLC_list,0],Carbon_evaluation[City_LCI_list,0],Carbon_evaluation[City_LCII_list,0]]

fig = plt.figure(figsize=(8, 8))

plt.title("Annual carbon mitigation potential of FPV", fontsize=fs-6, y=0.94, fontweight='bold')

plt.ylim(0, 40)
plt.boxplot(data_list, widths=0.4, patch_artist=True, boxprops=dict(facecolor='mediumseagreen'), tick_labels=[""] * 4, showfliers = False)
plt.grid(axis="y", linestyle="--", alpha=1)
plt.xlabel("")
plt.ylabel("Million ton")
plt.xticks(ticks=np.arange(1,5), labels=['SLC','VLC','LC-I','LC-II'])
plt.tick_params(axis='y', labelsize=fs-5)
plt.tight_layout()

fig.savefig('Figs_new/Fig1b_right.pdf',format='pdf',dpi=600,bbox_inches='tight')
fig.savefig("Figs_new/Fig1b_right.png", dpi=600,bbox_inches='tight')
plt.show()


from scipy.io import savemat,loadmat

path = '#ML_results/'
city_path = 'ALL_102_cities/'
path_type = '#ML_results/Power'
path_cap = '#ML_results/Capacity'
City_statistic = pd.read_excel(path+'City_statistic.xlsx',sheet_name = 'Class_Volume', index_col=0)
Statis_all = pd.read_excel(path+'City_statistic.xlsx',sheet_name = 'Key_information', index_col=0)

Total_Cap_total = np.zeros((102,2,5))
Total_Ele = np.zeros((102,5))
Total_Cost_total = np.zeros((102,6,5))

Total_price = np.zeros((102,2,5))
Total_price_true = np.zeros((102,2,5))
Total_Carbon = np.zeros((102,2,5))

Total_Carbon_00 = np.zeros((102,2,5))
Total_Carbon_00_ty = np.zeros((102,2,5))
Total_price_00 = np.zeros((102,2,5))
Total_price_00_true = np.zeros((102,2,5))

Total_area = np.zeros((102,1))

for cc in range(102):  #[3,15,56,59]:
    city_name = City_statistic.index[cc]
    print(city_name)
    C1 = path_cap+'/Cap_facade_'+city_name+'.npy'
    C2 = path_cap+'/Cap_roof_'+city_name+'.npy'

    G_type = np.load('#ML_results/Grid_type/'+'Grid_type_'+city_name+'.npy')
    Feas_read_sta = np.load(city_path+city_name+'_ALL_Featuers.npy')[:,[i for i in range(14)]+[15,16]]
    indices_non_zero = np.where(Feas_read_sta[:,11] != 0)[0]
    WWR = np.load(city_path+city_name+'_ALL_Featuers.npy')[indices_non_zero,13:14]
    N_grid = G_type.shape[0]
    N_gg = np.where(G_type[:,0] != 888)[0]
    Total_area[cc,0] = np.sum(G_type[:,4])/1e6 #(km2)

    th_hh = 18
    th_std = [0,0]  #[np.percentile(G_type[N_gg,2], 50),np.percentile(G_type[N_gg,2], 50)]
    th_aa = [200,200]  # [np.percentile(G_type[N_gg,3], 50),np.percentile(G_type[N_gg,3], 50)]   #200 #120  # 55#120[120,120]  #  
    list_form = [list(np.where((G_type[N_gg,1]>=th_hh)&(G_type[N_gg,2]>=th_std[1])&(G_type[N_gg,3]>=th_aa[1]))[0]),\
       list(np.where((G_type[N_gg,1]>=th_hh)&(G_type[N_gg,2]>=th_std[1])&(G_type[N_gg,3]<th_aa[0]))[0]),\
       list(np.where((G_type[N_gg,1]<th_hh)&(G_type[N_gg,2]>=th_std[0])&(G_type[N_gg,3]>=th_aa[1]))[0]),\
       list(np.where((G_type[N_gg,1]<th_hh)&(G_type[N_gg,2]>=th_std[0])&(G_type[N_gg,3]<th_aa[0]))[0])]

    if cc in [97,92,2,6,8,69]:
        data_path = '#Opt_results/'+city_name+'_hybrid_n.mat'
    else:
        data_path = '#Opt_results/'+city_name+'_hybrid.mat'
    Data = loadmat(data_path)
    R_Cap_r = Data['R_Cap_r']  #np.zeros((N_grid, 2, Y))
    R_Cap_f = Data['R_Cap_f']  #np.zeros((N_grid, 3, Y))
    R_Ele = Data['R_Ele']      #np.zeros((N_grid, Y))
    R_Pow_f = Data['R_Pow_f']  #np.zeros((N_grid,Y, D, T))
    R_Pow_r = Data['R_Pow_r']  #np.zeros((N_grid, 2, Y, D, T))
    R_Car = Data['R_Car']     #np.zeros((N_grid,2,Y))
    R_Cost = Data['R_Cost']    #np.zeros((N_grid,2,Y))
    R_Cost_true = Data['R_Cost_true']    #np.zeros((N_grid,2,Y))

    Total_Cap_total[cc,:,:] = np.sum(R_Cap_f[:,[0,2],:],axis = 0)
    Total_Carbon[cc,:,:] = 1e3*np.sum(R_Car[:,:,:],axis = 0)/np.sum(R_Ele[:,:],axis = 0)
    Total_price[cc,:,:] = np.sum(R_Cost[:,:,:],axis = 0)/np.sum(R_Ele[:,:],axis = 0)
    Total_price_true[cc,:,:] = np.sum(R_Cost_true[:,:,:],axis = 0)/np.sum(R_Ele[:,:],axis = 0)
    Total_Carbon_00[cc,:,:] = np.sum(R_Car[:,:,:],axis = 0)
    Total_Carbon_00_ty[cc,:,:] = np.sum(R_Car[list_form[0],:,:],axis = 0)
    
    Total_price_00[cc,:,:] = np.sum(R_Cost[:,:,:],axis = 0)
    Total_price_00_true[cc,:,:] = np.sum(R_Cost_true[:,:,:],axis = 0)

s_font,s_title = 16,14

from matplotlib.patches import Patch
list_all_city_name,list_all_city_j,list_all_city_ii = [],[],[]
list_all_city_order = np.zeros((102,3))
list_all_city_order[:,0] = np.arange(102)
for i in range(4):
    indices = Statis_all.index[Statis_all.iloc[:,-1] == ['超大城市','特大城市','I型大城市','II型大城市'][i]]
    for index in indices:
        i_loc = Statis_all.index.get_loc(index)
        list_all_city_order[i_loc,2] = 4-i
        list_all_city_order[i_loc,1] = Total_Cap_total[i_loc,0,-1]/1e6
sorted_indices = np.lexsort((-list_all_city_order[:, 1], -list_all_city_order[:, 2]))  # 注意负号实现降序
data_sorted = list_all_city_order[sorted_indices]
list_all_city_j = list(data_sorted[:,0].astype(int))
list_all_city_ii = [list_all_city_j[:7],list_all_city_j[7:22],list_all_city_j[22:35],list_all_city_j[35:]]

aaCar_wo,aaCar_ww = [],[]
aaPri_woFPV,aaPri_wFPV = [],[]
for i in range(4):
    aaCar_wo.append(np.sum(Total_Carbon[list_all_city_ii[i],0,:],axis=1)/5)
    aaCar_ww.append(np.sum(Total_Carbon[list_all_city_ii[i],1,:],axis=1)/5)
    aaPri_woFPV.append(np.sum(Total_price_true[list_all_city_ii[i],0,:],axis=1)*1e10/5)
    aaPri_wFPV.append(np.sum(Total_price_true[list_all_city_ii[i],1,:],axis=1)*1e10/5)
# #%%
# fig, ax = plt.subplots(1,2,figsize=(20,6))
# ax[0].boxplot(aaCar_wo, positions = np.arange(4) * 2.0,widths=0.4, patch_artist=True, boxprops=dict(facecolor='skyblue'))
# ax[0].boxplot(aaCar_ww, positions = np.arange(4) * 2.0+0.5,widths=0.4, patch_artist=True, boxprops=dict(facecolor='mediumseagreen'))#, showfliers = False)
# #ax[0].set_xlabel('City types', fontsize=s_font+4)
# ax[0].set_ylabel('ton/MWh',fontsize=s_font+4)
# ax[0].set_title("Unit carbon emission", fontsize=s_font+4, y=0.9, fontweight='bold')

# ax[0].set_xticks(ticks=np.arange(4) * 2.0+0.25, labels=['SLC','VLC','LC-I','LC-II'],fontsize=s_font+2)
# #ax[0].set_xticks(ticks=np.arange(4) * 2.0+0.25, labels=['','','',''],fontsize=s_font-0)
# ax[0].tick_params(axis='y', labelsize=s_font+2)
# #plt.legend(['RS', 'RS+F'], loc="upper right",fontsize=s_font-2)
# legend_elements = [
#     Patch(facecolor='skyblue', label='RS'),
#     Patch(facecolor='mediumseagreen', label='RS+F')
# ]

# ax[0].set_ylim(0.1,0.24)
# ax[0].set_yticks(0.01*np.arange(10, 24, 4))
# ax[0].legend(handles=legend_elements,fontsize=s_font+2,loc='lower left',ncol=2,frameon=True)
# #bbox_to_anchor=(1.1, -0.06),

# for i in [1.25,3.25,5.25]:
#     ax[0].axvline(x = i - 0, color='grey', linestyle='--', linewidth=0.6)

# ax[1].boxplot(aaPri_woFPV, positions = np.arange(4) * 2.0,widths=0.4, patch_artist=True, boxprops=dict(facecolor='skyblue'), tick_labels=[""] * 4, showfliers = False)
# ax[1].boxplot(aaPri_wFPV, positions = np.arange(4) * 2.0+0.5,widths=0.4, patch_artist=True, boxprops=dict(facecolor='mediumseagreen'), tick_labels=[""] * 4, showfliers = False)
# # ax[1].set_xlabel('City types', fontsize=s_font+4)
# ax[1].set_ylabel('CNY/kWh',fontsize=s_font+4)
# ax[1].tick_params(axis='y', labelsize=s_font+2)
# ax[1].set_xticks(ticks=np.arange(4) * 2.0+0.25, labels=['SLC','VLC','LC-I','LC-II'],fontsize=s_font+2)
# ax[1].set_ylim(0.36,0.65)
# ax[1].set_title("Unit cost", fontsize=s_font+4, y=0.9, fontweight='bold')
# ax[1].legend(handles=legend_elements,fontsize=s_font+2,loc='lower left',ncol=2,frameon=True)
# for i in [1.25,3.25,5.25]:
#     ax[1].axvline(x = i - 0, color='grey', linestyle='--', linewidth=0.6)
# ax[1].set_yticks(0.01*np.arange(36, 65, 9))

# #plt.subplots_adjust(hspace=0.1)
# fig.savefig('Figs_new/Fig1c.pdf',format='pdf',dpi=600,bbox_inches='tight')
# fig.savefig("Figs_new/Fig1c.png", dpi=600,bbox_inches='tight')
# plt.show()

#%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

fig, ax = plt.subplots(2, 1, figsize=(20, 6))
violin1 = ax[0].violinplot(
    aaCar_wo, 
    positions=np.arange(4) * 2.0, 
    widths=0.6,
    showmeans=True,
    showextrema=True
)
violin2 = ax[0].violinplot(
    aaCar_ww, 
    positions=np.arange(4) * 2.0 + 0.8, 
    widths=0.6,
    showmeans=True,
    showextrema=True
)

for pc in violin1['bodies']:
    pc.set_facecolor('skyblue')
    pc.set_alpha(0.7)
for pc in violin2['bodies']:
    pc.set_facecolor('mediumseagreen')
    pc.set_alpha(0.7)

ax[0].set_ylabel('ton/MWh', fontsize=s_font+4)
ax[0].set_title("Unit carbon emission", fontsize=s_font+4, y=0.82, fontweight='bold')
#ax[0].set_xticks(ticks=np.arange(4) * 2.0 + 0.4, labels=['SLC','VLC','LC-I','LC-II'], fontsize=s_font+2)
ax[0].set_xticks([])
ax[0].tick_params(axis='y', labelsize=s_font+2)
ax[0].set_ylim(0.1, 0.24)
ax[0].set_yticks(0.01 * np.arange(10, 24, 4))

legend_elements = [
    Patch(facecolor='skyblue', label='RS'),
    Patch(facecolor='mediumseagreen', label='RS+F')
]
ax[0].legend(handles=legend_elements, fontsize=s_font+2, loc='lower center',bbox_to_anchor=(0.5, -0.3), ncol=2, frameon=False)

for i in [1.4, 3.4, 5.4]:
    ax[0].axvline(x=i, color='grey', linestyle='--', linewidth=0.8)


violin3 = ax[1].violinplot(
    aaPri_woFPV, 
    positions=np.arange(4) * 2.0, 
    widths=0.6,
    showmeans=True,
    showextrema=True
)
violin4 = ax[1].violinplot(
    aaPri_wFPV, 
    positions=np.arange(4) * 2.0 + 0.8, 
    widths=0.6,
    showmeans=True,
    showextrema=True
)

for pc in violin3['bodies']:
    pc.set_facecolor('skyblue')
    pc.set_alpha(0.7)
for pc in violin4['bodies']:
    pc.set_facecolor('mediumseagreen')
    pc.set_alpha(0.7)

ax[1].set_ylabel('CNY/kWh', fontsize=s_font+4)
ax[1].set_title("Unit cost", fontsize=s_font+4, y=0.82, fontweight='bold')
ax[1].set_xticks(ticks=np.arange(4) * 2.0 + 0.4, labels=['SLC','VLC','LC-I','LC-II'], fontsize=s_font+2)
ax[1].tick_params(axis='y', labelsize=s_font+2)
ax[1].set_ylim(0.36, 0.65)
ax[1].set_yticks(0.01 * np.arange(36, 65, 9))

for i in [1.4, 3.4, 5.4]:
    ax[1].axvline(x=i, color='grey', linestyle='--', linewidth=0.8)

plt.subplots_adjust(hspace=0.28)

fig.savefig('Figs_new/Fig1c.pdf', format='pdf', dpi=600, bbox_inches='tight')
fig.savefig("Figs_new/Fig1c.png", dpi=600, bbox_inches='tight')
plt.show()


#%%

aa1 = (aaPri_woFPV[0]-aaPri_wFPV[0])/aaPri_woFPV[0]
aa2 = (aaPri_woFPV[1]-aaPri_wFPV[1])/aaPri_woFPV[1]
aa3 = (aaPri_woFPV[2]-aaPri_wFPV[2])/aaPri_woFPV[2]
aa4 = (aaPri_woFPV[3]-aaPri_wFPV[3])/aaPri_woFPV[3]
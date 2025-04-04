#%%
import json
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, shape, MultiPolygon
from tqdm import tqdm
import numpy as np
from matplotlib.patches import Patch
from geopy.distance import geodesic
from scipy.io import savemat,loadmat

path_type = 'Power'
path_cap = 'Capacity'

City_statistic = pd.read_excel('City_statistic.xlsx',sheet_name = 'Class_Volume', index_col=0)
Statis_all = pd.read_excel('City_statistic.xlsx',sheet_name = 'Key_information', index_col=0)

cc = 3
city_name = City_statistic.index[cc]
print(city_name)

if cc in [97,92,2,6,8,69]:
    data_path = city_name+'_hybrid_n.mat'
else:
    data_path = city_name+'_hybrid.mat'

C1 = path_cap+'/Cap_facade_'+city_name+'.npy'
C2 = path_cap+'/Cap_roof_'+city_name+'.npy'

G_type = np.load('Grid_type_'+city_name+'.npy')
Feas_read_sta = np.load(city_name+'_ALL_Featuers.npy')[:,[i for i in range(14)]+[15,16]]
indices_non_zero = np.where(Feas_read_sta[:,11] != 0)[0]
WWR = np.load(city_name+'_ALL_Featuers.npy')[indices_non_zero,13:14]
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
# R_Cost = Data['R_Cost']    #np.zeros((N_grid,2,Y))

th_hh = 18
th_std = [0,0]  #[np.percentile(G_type[N_gg,2], 50),np.percentile(G_type[N_gg,2], 50)]     #4  # 3.6#4[3.6,3.6]  # 
th_aa = [200,200]  # [np.percentile(G_type[N_gg,3], 50),np.percentile(G_type[N_gg,3], 50)]   #200 #120  # 55#120[120,120]  #  
list_form = [list(np.where((G_type[N_gg,1]>=th_hh)&(G_type[N_gg,2]>=th_std[1])&(G_type[N_gg,3]>=th_aa[1]))[0]),\
       list(np.where((G_type[N_gg,1]>=th_hh)&(G_type[N_gg,2]>=th_std[1])&(G_type[N_gg,3]<th_aa[0]))[0]),\
       list(np.where((G_type[N_gg,1]<th_hh)&(G_type[N_gg,2]>=th_std[0])&(G_type[N_gg,3]>=th_aa[1]))[0]),\
       list(np.where((G_type[N_gg,1]<th_hh)&(G_type[N_gg,2]>=th_std[0])&(G_type[N_gg,3]<th_aa[0]))[0])]

import matplotlib.patches as patches
import matplotlib.lines as mlines

if cc == 3:
    i_lon,i_lat = 116.4, 39.92
    i_rad1,i_rad2,i_rad3 = 0.18,0.18+0.15,0.18+0.15+0.13
elif cc == 74:
    i_lon,i_lat = 114.3, 30.55
    i_rad1,i_rad2,i_rad3 = 0.16,0.16+0.1,0.16+0.1+0.09
elif cc == 56:
    i_lon,i_lat = 121.4, 31.2
    i_rad1,i_rad2,i_rad3 = 0.16,0.16+0.13,0.16+0.13+0.14
elif cc == 15:
    i_lon,i_lat = 113.3, 23.15
    i_rad1,i_rad2,i_rad3 = 0.14,0.14+0.13,0.14+0.13+0.16
elif cc == 59:
    i_lon,i_lat = 114.11, 22.58
    i_rad1,i_rad2,i_rad3 = 0.15,0.15+0.15,0.15+0.15+0.0

if cc == 15:
    out11 = patches.Circle((i_lon,i_lat), radius=i_rad1, edgecolor='grey', facecolor='none', linewidth=3, linestyle = '-')
    out21 = patches.Circle((i_lon+0.1,i_lat), radius=i_rad2, edgecolor='grey', facecolor='none', linewidth=3, linestyle = '--')
    out31 = patches.Circle((i_lon+0.2,i_lat), radius=i_rad3, edgecolor='grey', facecolor='none', linewidth=3, linestyle = '-.')
    out12 = patches.Circle((i_lon,i_lat), radius=i_rad1, edgecolor='grey', facecolor='none', linewidth=3, linestyle = '-')
    out22 = patches.Circle((i_lon+0.1,i_lat), radius=i_rad2, edgecolor='grey', facecolor='none', linewidth=3, linestyle = '--')
    out32 = patches.Circle((i_lon+0.2,i_lat), radius=i_rad3, edgecolor='grey', facecolor='none', linewidth=3, linestyle = '-.')
else:
    out11 = patches.Circle((i_lon,i_lat), radius=i_rad1, edgecolor='grey', facecolor='none', linewidth=3, linestyle = '-')
    out21 = patches.Circle((i_lon,i_lat), radius=i_rad2, edgecolor='grey', facecolor='none', linewidth=3, linestyle = '--')
    out31 = patches.Circle((i_lon,i_lat), radius=i_rad3, edgecolor='grey', facecolor='none', linewidth=3, linestyle = '-.')
    out12 = patches.Circle((i_lon,i_lat), radius=i_rad1, edgecolor='grey', facecolor='none', linewidth=3, linestyle = '-')
    out22 = patches.Circle((i_lon,i_lat), radius=i_rad2, edgecolor='grey', facecolor='none', linewidth=3, linestyle = '--')
    out32 = patches.Circle((i_lon,i_lat), radius=i_rad3, edgecolor='grey', facecolor='none', linewidth=3, linestyle = '-.')

s_font,s_title = 16,14
Grid = pd.DataFrame(np.zeros((len(N_gg),5)),columns = ['Long.','Lat.','Type','Capacity-2030(KW)','Capacity-2050(KW)'])
Forms = ['HnD','HnS','MnD','MnS']
for i in range(4):
    Grid.iloc[list_form[i],:2] = G_type[N_gg,:][list_form[i],6:8]
    Grid.iloc[list_form[i],2] = Forms[i]
    Grid.iloc[list_form[i],3] = R_Cap_f[N_gg,:,:][list_form[i],0,0]
    Grid.iloc[list_form[i],4] = R_Cap_f[N_gg,:,:][list_form[i],0,-1]

writer = pd.ExcelWriter(city_name+'_grids.xlsx')
Grid.to_excel(writer,sheet_name='Grids')
writer.close()

s_font,s_title = 16+5,14+5

geojson_path = city_name+'.json'
with open(geojson_path, 'r', encoding='utf-8') as f:
    beijing_geojson = json.load(f)

data_path = city_name+'_grids.xlsx'
data = pd.read_excel(data_path)

data_array = data.values
data_array = data_array.astype(str)

def create_square(lat, lon, size_km = 2.5):
    half_size_km = size_km / 2
    bottom_left = geodesic(kilometers=half_size_km).destination((lat, lon), 225)
    bottom_right = geodesic(kilometers=half_size_km).destination((lat, lon), 315)
    top_right = geodesic(kilometers=half_size_km).destination((lat, lon), 45)
    top_left = geodesic(kilometers=half_size_km).destination((lat, lon), 135)
    
    return Polygon([
        (bottom_left.longitude, bottom_left.latitude),
        (bottom_right.longitude, bottom_right.latitude),
        (top_right.longitude, top_right.latitude),
        (top_left.longitude, top_left.latitude)
    ])

tqdm.pandas(desc="Processing coordinates")
data['geometry'] = data.progress_apply(lambda row: create_square(float(row['Lat.']), float(row['Long.'])), axis=1)

types = ['HnD','HnS','MnD','MnS']
colors = [ 'Blues', 'Oranges', 'Greens', 'Purples']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 15), gridspec_kw={'wspace': 0.1})

for ax in [ax1, ax2]:
    for feature in beijing_geojson['features']:
        geom = shape(feature['geometry'])
        if isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                x, y = poly.exterior.xy
                ax.plot(x, y, color='black')
        else:
            x, y = geom.exterior.xy
            ax.plot(x, y, color='black')
norm = plt.Normalize(data['Capacity-2030(KW)'].min(), 0.5*data['Capacity-2050(KW)'].max())

ax1.get_legend_handles_labels()[1]

for t, cmap_name in zip(types, colors):
    cmap = plt.get_cmap(cmap_name)
    for _, row in data[data['Type'] == t].iterrows():
        x, y = row['geometry'].exterior.xy
        color1 = cmap(norm(row['Capacity-2030(KW)']))
        color2 = cmap(norm(row['Capacity-2050(KW)']))        
        ax1.fill(x, y, color=color1, alpha=0.9, label=t if t not in ax1.get_legend_handles_labels()[1] else "")
        ax2.fill(x, y, color=color2, alpha=0.9, label=t if t not in ax2.get_legend_handles_labels()[1] else "")

ax1.set_title('2030',fontsize = s_font+5, fontweight='bold')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.tick_params(left=False, bottom=False)
ax1.set_xticks([])
ax1.set_yticks([])
if cc != 59:
    ax1.add_patch(out11)
    ax1.add_patch(out21)
    ax1.add_patch(out31)

custom_lines = [Patch(facecolor='darkorange', edgecolor='darkorange', label=types[0]),
                Patch(facecolor='indigo', edgecolor='indigo', label=types[1]),
                Patch(facecolor='green', edgecolor='green', label=types[2]),
                Patch(facecolor='blue', edgecolor='blue', label=types[3])]

ax2.set_title('2050',fontsize = s_font+5, fontweight='bold')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.tick_params(left=False, bottom=False)
ax2.set_xticks([])
ax2.set_yticks([])
if cc != 59:
    ax2.add_patch(out12)
    ax2.add_patch(out22)
    ax2.add_patch(out32)

center_line = mlines.Line2D([], [], color='gray', linestyle='-', linewidth=2, label='Center')
neighbor_line = mlines.Line2D([], [], color='gray', linestyle='--', linewidth=2, label='Expansion')
outlier_dot = mlines.Line2D([], [], color='gray', linestyle='-.', linewidth=2, label='Suburb')
if cc != 59:
    ax2.legend(handles=[center_line, neighbor_line, outlier_dot], loc='upper left',frameon=True, fontsize = s_font)

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

for handle in handles1:
    handle.set_alpha(1)
for handle in handles2:
    handle.set_alpha(1)

fig.subplots_adjust(bottom=0.3)
cbar_ax1 = fig.add_axes([0.14, 0.27, 0.15, 0.02])
cbar_ax2 = fig.add_axes([0.34, 0.27, 0.15, 0.02])
cbar_ax3 = fig.add_axes([0.54, 0.27, 0.15, 0.02])
cbar_ax4 = fig.add_axes([0.74, 0.27, 0.15, 0.02])

sm1 = plt.cm.ScalarMappable(cmap=plt.get_cmap(colors[0]), norm=norm)
sm2 = plt.cm.ScalarMappable(cmap=plt.get_cmap(colors[1]), norm=norm)
sm3 = plt.cm.ScalarMappable(cmap=plt.get_cmap(colors[2]), norm=norm)
sm4 = plt.cm.ScalarMappable(cmap=plt.get_cmap(colors[3]), norm=norm)

cbar1 = fig.colorbar(sm1, cax=cbar_ax1, orientation='horizontal')
cbar1.ax.tick_params(labelsize = s_font-2)
cbar1.set_label(types[0], fontsize = s_font, labelpad=15)

cbar2 = fig.colorbar(sm2, cax=cbar_ax2, orientation='horizontal')
cbar2.ax.tick_params(labelsize = s_font-2)
cbar2.set_label(types[1], fontsize = s_font, labelpad=15)

cbar3 = fig.colorbar(sm3, cax=cbar_ax3, orientation='horizontal')
cbar3.ax.tick_params(labelsize = s_font-2)
cbar3.set_label(types[2], fontsize = s_font, labelpad=15)

cbar4 = fig.colorbar(sm4, cax=cbar_ax4, orientation='horizontal')
cbar4.ax.tick_params(labelsize = s_font-2)
cbar4.set_label(types[3], fontsize = s_font, labelpad=15)

fig.savefig('Figs/Fig5-1'+city_name+'.pdf',format='pdf',dpi=600)
fig.savefig('Figs/Fig5-1'+city_name+'.png', dpi=600)
plt.show()

#%%

from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(10, 8))
gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], wspace=0.2, hspace=0.2)

list_form.append(list(np.arange(len(N_gg))))
types = ['HnD','HnS','MnD','MnS']
colors = [ 'Blue', 'Orange', 'Green', 'Purple']
Cap_form = []
Cap_carbon = []
for i1 in range(2):
    for i2 in range(2):
        i = i1*2+i2
        Uni_Cap = np.zeros((len(list_form[i]),4))
        Uni_Cap[:,0] = G_type[list_form[i],4]
        Uni_Cap[:,1] = R_Cap_f[list_form[i],0,-1]    #/G_type[:,4]  #np.sum(Uni_pow,axis = 1)/G_type[:,4]
        Uni_Cap[:,2] = np.sum(R_Pow_f[list_form[i],:,:,:], axis = (1,2,3))    #/G_type[:,4]  #np.sum(Uni_pow,axis = 1)/G_type[:,4]
        Uni_Cap[:,3] = np.sum(R_Car[list_form[i],0,:], axis = 1)-np.sum(R_Car[list_form[i],1,:],axis = 1)
        
        x,y = Uni_Cap[:,0]/1e6, Uni_Cap[:,1]/1e3
        Cap_form.append(np.sum(R_Cap_f[list_form[i],0,-1]))
        Cap_carbon.append(np.sum(Uni_Cap[:,3]))
        
        ax1 = fig.add_subplot(gs[i1, i2])
        #ax1.scatter(np.arange(len(x)), y/x, c = colors[i],alpha = 0.5)
        ax1.scatter(x,y,c = colors[i],alpha = 0.5)
        ax1.set_title(types[i],fontsize = s_font-4, y=0.85)
        slope1, slope2, intercept = np.polyfit(x, y, 2)
        x0 = np.linspace(0, 6, 100)
        fit_line = slope1 * x0**2 + slope2 * x0 + intercept
        ax1.plot(x0, fit_line, color = 'grey', linestyle = '--', label='Fitted Line')

        if i1 == 1:
            ax1.set_xlabel('Facade area (km2)',fontsize = s_font-4)
        if i2 == 0:
            x1,y1 = 5,160     #175, 20000  #
            ax1.set_ylabel('Capacity (MW)',fontsize = s_font-4)
        if i2 == 1:
            x1,y1 = 0.5,30    #25, 3500  #
        ax1.set_xlim(0,x1)
        ax1.set_ylim(0,y1)
        
fig.savefig('Figs/Fig5-2'+city_name+'.pdf',format='pdf',dpi=600)
fig.savefig('Figs/Fig5-2'+city_name+'.png', dpi=600)
plt.show()

print(Cap_form[0]/sum(Cap_form))
print(Cap_carbon[0]/sum(Cap_carbon))


#%%
Cap_indu = np.zeros((6,4))
Cap_indu_uni = []  #np.zeros((6,4))
Cap_indu_uni_hour = []  #np.zeros((6,4))
Cap_car = np.zeros((6,4))
Cap_car_ori = np.zeros((6,4))
Cap_car_uni = []
Cap_car_uni_ori = []
Cap_rpv = np.zeros((6,4))
Cap_rpv_uni = []  #np.zeros((6,4))

if cc in [0,4,10,11,1,3,93,94,5,13,12,7,9]:
    Feas_read_info = np.load(city_name+'_ALL_Featuers_supplementary.npy')
else:
    Feas_read_info = np.load(city_name+'_ALL_Featuers.npy')[:,range(17,29)]

R_area = Feas_read_info[indices_non_zero,3:4]

Clu_center,Clu_days = np.load('Clu_center.npy'),np.load('Clu_days.npy')
Ele_FRL = np.zeros((3,4))

for i in range(4):
    Total_Ele = sum(Clu_days[cc,d]*np.sum(R_Pow_f[list_form[i],:,d,:],axis = (2)) for d in range(12))
    Ele_FRL[0,i] = sum(Clu_days[cc,d]*np.sum(R_Pow_f[list_form[i],-1,d,:],axis = (0,1)) for d in range(12))
    Ele_FRL[1,i] = sum(Clu_days[cc,d]*np.sum(R_Pow_r[list_form[i],1,-1,d,:],axis = (0,1)) for d in range(12))
    Ele_FRL[2,i] = sum(Clu_days[cc,d]*np.sum(R_Pow[list_form[i],-1,d,:],axis = (0,1)) for d in range(12))

    Cap_indu[1:,i] = np.sum(R_Cap_f[N_gg,:,:][list_form[i],0,:],axis = 0)/1e6  #kW
    Cap_indu_uni.append(1e3*R_Cap_f[N_gg,:,:][list_form[i],0,:]/G_type[N_gg,:][list_form[i],4:5])  #W/m2
    non_zeros = np.where(R_Cap_f[list_form[i],0,0] != 0)[0]
    Cap_indu_uni_hour.append(Total_Ele[non_zeros,:]/R_Cap_f[list_form[i],0,:][non_zeros,:])  #W/m2
    
    for y in range(5):
        Cap_car[1+y,i] = (np.sum(R_Car[N_gg,:,:][list_form[i],1,:(y+1)],axis = (0,1)))/1e6
        Cap_car_ori[1+y,i] = (np.sum(R_Car[N_gg,:,:][list_form[i],0,:(y+1)],axis = (0,1)))/1e6
    Cap_car_uni.append((1e3*R_Car[N_gg,:,:][list_form[i],1,:])/R_Ele[N_gg,:][list_form[i],:])
    Cap_car_uni_ori.append((1e3*R_Car[N_gg,:,:][list_form[i],0,:])/R_Ele[N_gg,:][list_form[i],:])

    Cap_rpv[1:,i] = np.sum(R_Cap_r[N_gg,:,:][list_form[i],1,:],axis = 0)/1e6
    Cap_rpv_uni.append(1e3*R_Cap_r[N_gg,:,:][list_form[i],1,:]/R_area[list_form[i],0:1])

Caps = [Cap_indu,Cap_indu_uni]
for jj in range(1):
    Cap_cum = np.cumsum(Caps[jj], axis=1)
    fig, ax = plt.subplots(figsize = (10,6))
    colors = [ 'Blue', 'Orange', 'Green', 'Purple']
    stages = ['2030', '2035', '2040', '2045', '2050']
    for i in range(Cap_cum.shape[1]):
        if i == 0:
            ax.fill_between(stages, 0, Cap_cum[1:,i], color=colors[i], alpha=0.5)
        else:
            ax.fill_between(stages, Cap_cum[1:,i-1], Cap_cum[1:,i], color=colors[i], alpha=0.5)
        ax.plot(stages,Cap_cum[1:,i],marker = '.',color=colors[i], markersize=10)
    custom_lines = [plt.Line2D([0], [0], color=colors[i], lw=4) for i in range(4)]
    ax.legend(custom_lines, Forms,loc='upper left', fontsize = 10, ncol=1)

    ax.set_xlabel('Stages',fontsize = s_font-2)
    ax.set_ylabel(['Planned Capacity (GW)','Unit Capacity (kW/m2)'][jj],fontsize = s_font-2)
    fig.savefig('Figs/SFig5-1'+city_name+'.pdf',format='pdf',dpi=600)
    fig.savefig('Figs/SFig5-1'+city_name+'.png', dpi=600)
    plt.show()

print(Cap_indu[1,0]/np.sum(Cap_indu[1,:]))
print(np.sum(Cap_indu[1,1:3])/np.sum(Cap_indu[1,:]))

#print(Ele_FRL/np.sum(Ele_FRL,axis = 1).reshape(-1,1))
print(Ele_FRL/Ele_FRL[2:3,:])


#%%
stages = 5
cities = 4
data = [[Cap_indu_uni[j][:,i] for j in range(cities)] for i in range(stages)]
flattened_data = [item for sublist in data for item in sublist]
group_spacing = 4
bar_width = 0.2
positions = []
group_spacing = 2
for i in range(stages):
    positions.extend([i * group_spacing + j * 0.2 for j in range(cities)])
plt.figure(figsize=(12, 6))
box = plt.boxplot(flattened_data, positions=positions, patch_artist=True, widths=0.15)
colors = [ 'Blue', 'Orange', 'Green', 'Purple']
for i, box in enumerate(box['boxes']):
    box.set_facecolor(colors[i % cities])
xticks = [i * group_spacing + (cities - 1) * bar_width / 2 for i in range(stages)]
xtick_labels = ['2030', '2035', '2040', '2045', '2050']
plt.xticks(xticks, xtick_labels, fontsize=12)
plt.ylabel('Unit capacity (W/m2)',fontsize = s_font-2)
plt.xlabel('Stages',fontsize = s_font-2)
custom_lines = [plt.Line2D([0], [0], color=colors[i], lw=4) for i in range(4)]
plt.legend(custom_lines, Forms,loc='upper right', fontsize = 13, ncol=4)
fig.savefig('Figs/SFig5-3a'+city_name+'.pdf',format='pdf',dpi=600)
fig.savefig('Figs/SFig5-3a'+city_name+'.png', dpi=600)
plt.show()



stages = 5
cities = 4
data = [[Cap_indu_uni_hour[j][:,i] for j in range(cities)] for i in range(stages)]
print(Cap_indu_uni_hour[0].shape)
flattened_data = [item for sublist in data for item in sublist]
group_spacing = 4
bar_width = 0.2
positions = []
group_spacing = 2
for i in range(stages):
    positions.extend([i * group_spacing + j * 0.2 for j in range(cities)])
plt.figure(figsize=(12, 6))
box = plt.boxplot(flattened_data, positions=positions, patch_artist=True, widths=0.15)
colors = [ 'Blue', 'Orange', 'Green', 'Purple']
for i, box in enumerate(box['boxes']):
    box.set_facecolor(colors[i % cities])
xticks = [i * group_spacing + (cities - 1) * bar_width / 2 for i in range(stages)]
xtick_labels = ['2030', '2035', '2040', '2045', '2050']
plt.xticks(xticks, xtick_labels, fontsize=12)
plt.ylabel('Annual utilization hours',fontsize = s_font-2)
plt.xlabel('Stages',fontsize = s_font-2)
if cc == 3 or cc == 74:
    plt.ylim(500,2000)
custom_lines = [plt.Line2D([0], [0], color=colors[i], lw=4) for i in range(4)]
plt.legend(custom_lines, Forms,loc='upper right', fontsize = 13, ncol=4)
fig.savefig('Figs/SFig5-3b'+city_name+'.pdf',format='pdf',dpi=600)
fig.savefig('Figs/SFig5-3b'+city_name+'.png', dpi=600)
plt.show()




#%%
Car_cum = np.cumsum(Cap_car_ori, axis=1)
if cc == 3:
    y_lim = 800
elif cc == 74:
    y_lim = 520
elif cc == 59:
    y_lim = 360
fig, ax = plt.subplots(figsize = (10,6))
colors = [ 'Blue', 'Orange', 'Green', 'Purple']
stages = ['2030', '2035', '2040', '2045', '2050']
for i in range(Car_cum.shape[1]):
    if i == 0:
        ax.fill_between(stages, 0, Car_cum[1:,i], color=colors[i], alpha=0.5)
    else:
        ax.fill_between(stages, Car_cum[1:,i-1], Car_cum[1:,i], color=colors[i], alpha=0.5)
    ax.plot(stages,Car_cum[1:,i],marker = '.',color=colors[i], markersize=10)
custom_lines = [plt.Line2D([0], [0], color=colors[i], lw=4) for i in range(4)]
#ax.set_xticks(np.arange(0.5, 6.5, 1))
#ax.set_xticklabels(['2025', '2030', '2035', '2040', '2045', '2050'])
ax.legend(custom_lines, Forms,loc='upper left', fontsize = 10, ncol=1)
ax.set_ylim(0,y_lim)
#ax.set_title(['Others','Residential'][jj],fontsize = s_title)
ax.set_xlabel('Stages',fontsize = s_font-2)
ax.set_ylabel('Carbon emission (Million ton)',fontsize = s_font-2)
fig.savefig('Figs/SFig5-2a'+city_name+'.pdf',format='pdf',dpi=600)
fig.savefig('Figs/SFig5-2a'+city_name+'.png', dpi=600)
plt.show()

print((np.cumsum(Cap_car_ori, axis=1)[1,-1]-np.cumsum(Cap_car, axis=1)[1,-1])/np.cumsum(Cap_car_ori, axis=1)[1,-1])


#%%
Car_cum = np.cumsum(Cap_car, axis=1)
fig, ax = plt.subplots(figsize = (10,6))
colors = [ 'Blue', 'Orange', 'Green', 'Purple']
stages = ['2030', '2035', '2040', '2045', '2050']
for i in range(Car_cum.shape[1]):
    if i == 0:
        ax.fill_between(stages, 0, Car_cum[1:,i], color=colors[i], alpha=0.5)
    else:
        ax.fill_between(stages, Car_cum[1:,i-1], Car_cum[1:,i], color=colors[i], alpha=0.5)
    ax.plot(stages,Car_cum[1:,i],marker = '.',color=colors[i], markersize=10)
custom_lines = [plt.Line2D([0], [0], color=colors[i], lw=4) for i in range(4)]
#ax.set_xticks(np.arange(0.5, 6.5, 1))
#ax.set_xticklabels(['2025', '2030', '2035', '2040', '2045', '2050'])
ax.legend(custom_lines, Forms,loc='upper left', fontsize = 10, ncol=1)
ax.set_ylim(0,y_lim)
#ax.set_title(['Others','Residential'][jj],fontsize = s_title)
ax.set_xlabel('Stages',fontsize = s_font-2)
ax.set_ylabel('Carbon emission (Million ton)',fontsize = s_font-2)
fig.savefig('Figs/SFig5-2b'+city_name+'.pdf',format='pdf',dpi=600)
fig.savefig('Figs/SFig5-2b'+city_name+'.png', dpi=600)
plt.show()

#%%
if cc == 3:
    yy_lim = 0.52
elif cc == 74:
    yy_lim = 0.52
elif cc == 59:
    yy_lim = 0.5
stages = 5
cities = 4
data = [[Cap_car_uni_ori[j][:,i] for j in range(cities)] for i in range(stages)]
flattened_data = [item for sublist in data for item in sublist]
group_spacing = 4
bar_width = 0.2
positions = []
group_spacing = 2
for i in range(stages):
    positions.extend([i * group_spacing + j * 0.2 for j in range(cities)])
plt.figure(figsize=(12, 6))
box = plt.boxplot(flattened_data, positions=positions, patch_artist=True, widths=0.15)
colors = [ 'Blue', 'Orange', 'Green', 'Purple']
for i, box in enumerate(box['boxes']):
    box.set_facecolor(colors[i % cities])
xticks = [i * group_spacing + (cities - 1) * bar_width / 2 for i in range(stages)]
xtick_labels = ['2030', '2035', '2040', '2045', '2050']
plt.xticks(xticks, xtick_labels, fontsize=12)
plt.ylabel('Unit carbon emission (kg/kWh)',fontsize = s_font-2)
plt.xlabel('Stages',fontsize = s_font-2)
plt.ylim(-0.05,yy_lim)
custom_lines = [plt.Line2D([0], [0], color=colors[i], lw=4) for i in range(4)]
plt.legend(custom_lines, Forms,loc='upper right', fontsize = 13, ncol=4)
fig.savefig('Figs/SFig5-4a'+city_name+'.pdf',format='pdf',dpi=600)
fig.savefig('Figs/SFig5-4a'+city_name+'.png', dpi=600)
plt.show()


#%%
stages = 5
cities = 4
data = [[Cap_car_uni[j][:,i] for j in range(cities)] for i in range(stages)]
flattened_data = [item for sublist in data for item in sublist]
group_spacing = 4
bar_width = 0.2
positions = []
group_spacing = 2
for i in range(stages):
    positions.extend([i * group_spacing + j * 0.2 for j in range(cities)])
plt.figure(figsize=(12, 6))
box = plt.boxplot(flattened_data, positions=positions, patch_artist=True, widths=0.15)
colors = [ 'Blue', 'Orange', 'Green', 'Purple']
for i, box in enumerate(box['boxes']):
    box.set_facecolor(colors[i % cities])
xticks = [i * group_spacing + (cities - 1) * bar_width / 2 for i in range(stages)]
xtick_labels = ['2030', '2035', '2040', '2045', '2050']
plt.xticks(xticks, xtick_labels, fontsize=12)
plt.ylabel('Unit carbon emission (kg/kWh)',fontsize = s_font-2)
plt.xlabel('Stages',fontsize = s_font-2)
plt.ylim(-0.05,yy_lim)
custom_lines = [plt.Line2D([0], [0], color=colors[i], lw=4) for i in range(4)]
plt.legend(custom_lines, Forms,loc='upper right', fontsize = 13, ncol=4)
fig.savefig('Figs/SFig5-4b'+city_name+'.pdf',format='pdf',dpi=600)
fig.savefig('Figs/SFig5-4b'+city_name+'.png', dpi=600)
plt.show()

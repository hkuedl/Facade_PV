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
import matplotlib.font_manager as fm
from matplotlib import rcParams

path = ''
city_path = 'ALL_102_cities/'
path_type = 'Power'
path_cap = 'Capacity'
City_statistic = pd.read_excel(path+'City_statistic.xlsx',sheet_name = 'Class_Volume', index_col=0)
Statis_all = pd.read_excel(path+'City_statistic.xlsx',sheet_name = 'Key_information', index_col=0)

#3: Beijing; 74: Wuhan; 63: Suzhou; 92：Changchun; 56: Shanghai； 15: Guangzhou；59：shenzhen；60：shenyang
cc = 3  #74, 59
city_name = City_statistic.index[cc]
print(city_name)

if cc in [97,92,2,6,8,69]:
    data_path = '#Opt_results/'+city_name+'_hybrid_n.mat'
else:
    data_path = '#Opt_results/'+city_name+'_hybrid.mat'

C1 = path_cap+'/Cap_facade_'+city_name+'.npy'
C2 = path_cap+'/Cap_roof_'+city_name+'.npy'

G_type = np.load('Grid_type/'+'Grid_type_'+city_name+'.npy')
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
    out13 = patches.Circle((i_lon,i_lat), radius=i_rad1, edgecolor='grey', facecolor='none', linewidth=3, linestyle = '-')
    out23 = patches.Circle((i_lon+0.1,i_lat), radius=i_rad2, edgecolor='grey', facecolor='none', linewidth=3, linestyle = '--')
    out33 = patches.Circle((i_lon+0.2,i_lat), radius=i_rad3, edgecolor='grey', facecolor='none', linewidth=3, linestyle = '-.')
else:
    out11 = patches.Circle((i_lon,i_lat), radius=i_rad1, edgecolor='grey', facecolor='none', linewidth=3, linestyle = '-')
    out21 = patches.Circle((i_lon,i_lat), radius=i_rad2, edgecolor='grey', facecolor='none', linewidth=3, linestyle = '--')
    out31 = patches.Circle((i_lon,i_lat), radius=i_rad3, edgecolor='grey', facecolor='none', linewidth=3, linestyle = '-.')
    out12 = patches.Circle((i_lon,i_lat), radius=i_rad1, edgecolor='grey', facecolor='none', linewidth=3, linestyle = '-')
    out22 = patches.Circle((i_lon,i_lat), radius=i_rad2, edgecolor='grey', facecolor='none', linewidth=3, linestyle = '--')
    out32 = patches.Circle((i_lon,i_lat), radius=i_rad3, edgecolor='grey', facecolor='none', linewidth=3, linestyle = '-.')
    out13 = patches.Circle((i_lon,i_lat), radius=i_rad1, edgecolor='grey', facecolor='none', linewidth=3, linestyle = '-')
    out23 = patches.Circle((i_lon,i_lat), radius=i_rad2, edgecolor='grey', facecolor='none', linewidth=3, linestyle = '--')
    out33 = patches.Circle((i_lon,i_lat), radius=i_rad3, edgecolor='grey', facecolor='none', linewidth=3, linestyle = '-.')

Grid = pd.DataFrame(np.zeros((len(N_gg),6)),columns = ['Long.','Lat.','Type','Capacity-2030(KW)','Capacity-2040(KW)','Capacity-2050(KW)'])
Forms = ['HnD','HnS','MnD','MnS']
#Indus = ['Residential','Commercial','Industrial','Transportation','Public']
for i in range(4):
    Grid.iloc[list_form[i],:2] = G_type[N_gg,:][list_form[i],6:8]
    Grid.iloc[list_form[i],2] = Forms[i]
    Grid.iloc[list_form[i],3] = R_Cap_f[N_gg,:,:][list_form[i],0,0]  #/G_type[N_gg,:][list_form[i],4]
    Grid.iloc[list_form[i],4] = R_Cap_f[N_gg,:,:][list_form[i],0,2] #/G_type[N_gg,:][list_form[i],4]
    Grid.iloc[list_form[i],5] = R_Cap_f[N_gg,:,:][list_form[i],0,-1] #/G_type[N_gg,:][list_form[i],4]



s_font_title,s_font_legend,s_font_label,s_font_label_title = 60,60,60,60

geojson_path = '#Opt_results/'+city_name+'.json'
with open(geojson_path, 'r', encoding='utf-8') as f:
    beijing_geojson = json.load(f)

data = Grid.copy()
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

# types = ['HnS', 'MnS', 'MnD', 'HnD']
# colors = ['Oranges', 'Purples', 'Greens', 'Blues']
types = ['HnD','HnS','MnD','MnS']
colors = [ 'Blues', 'Oranges', 'Greens', 'Purples']

font_path = 'arial.ttf'
custom_font = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)
rcParams['font.family'] = custom_font.get_name()
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(45, 15), gridspec_kw={'wspace': 0.1})

for ax in [ax1, ax2, ax3]:
    for feature in beijing_geojson['features']:
        geom = shape(feature['geometry'])
        if isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                x, y = poly.exterior.xy
                ax.plot(x, y, color='black')
        else:
            x, y = geom.exterior.xy
            ax.plot(x, y, color='black')

#norm = plt.Normalize(data['Capacity-2030(KW)'].min()/1000, 0.5*data['Capacity-2050(KW)'].max()/1000)
if cc == 3:
    norm = plt.Normalize(0, 80)
elif cc == 74:
    norm = plt.Normalize(0, 80)
elif cc == 59:
    norm = plt.Normalize(0, 80)


Data_ratio = np.zeros((len(types),3))
for i in range(3):
    for j in range(4):
        Data_ratio[j,i] = data[data['Type'] == types[j]]['Capacity-20'+str(30+i*10)+'(KW)'].sum() / data['Capacity-20'+str(30+i*10)+'(KW)'].sum()
        Data_ratio[j,i] = round(100*Data_ratio[j,i],1)
    if sum(Data_ratio[:,i]) != 100:
        Data_ratio[0,i] = 100 - sum(Data_ratio[1:,i])
ax1.get_legend_handles_labels()[1]

for t, cmap_name in zip(types, colors):
    cmap = plt.get_cmap(cmap_name)
    for _, row in data[data['Type'] == t].iterrows():
        x, y = row['geometry'].exterior.xy
        color1 = cmap(norm(row['Capacity-2030(KW)']/1000))
        color2 = cmap(norm(row['Capacity-2040(KW)']/1000))        
        color3 = cmap(norm(row['Capacity-2050(KW)']/1000))        
        ax1.fill(x, y, color=color1, alpha=0.9, label=t if t not in ax1.get_legend_handles_labels()[1] else "")
        ax2.fill(x, y, color=color2, alpha=0.9, label=t if t not in ax2.get_legend_handles_labels()[1] else "")
        ax3.fill(x, y, color=color3, alpha=0.9, label=t if t not in ax3.get_legend_handles_labels()[1] else "")

ax1.set_title('2030',fontsize = s_font_title, fontweight='bold',y=0.97)
#ax1.set_xlabel('Longitude',fontsize = s_font)
#ax1.set_ylabel('Latitude',fontsize = s_font)
#ax1.legend(handles=custom_lines, loc="upper left", fontsize = s_font)
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

ax2.set_title('2040',fontsize = s_font_title, fontweight='bold',y=0.97)
if cc == 59:
    ax2.text(0.5, 1.13, 'Shenzhen (SLC)', fontsize=s_font_title,fontweight='bold', ha='center', va='center', transform=ax2.transAxes)
elif cc == 74:
    ax2.text(0.5, 1.13, 'Wuhan (VLC)', fontsize=s_font_title,fontweight='bold', ha='center', va='center', transform=ax2.transAxes)
elif cc == 3:
    ax2.text(0.5, 1.13, 'Beijing (SLC)', fontsize=s_font_title,fontweight='bold', ha='center', va='center', transform=ax2.transAxes)
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

ax3.set_title('2050',fontsize = s_font_title, fontweight='bold',y=0.97)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['bottom'].set_visible(False)
ax3.spines['left'].set_visible(False)
ax3.tick_params(left=False, bottom=False)
ax3.set_xticks([])
ax3.set_yticks([])
if cc != 59:
    ax3.add_patch(out13)
    ax3.add_patch(out23)
    ax3.add_patch(out33)


center_line = mlines.Line2D([], [], color='gray', linestyle='-', linewidth=8, label='Center')
neighbor_line = mlines.Line2D([], [], color='gray', linestyle='--', linewidth=8, label='Expansion')
outlier_dot = mlines.Line2D([], [], color='gray', linestyle='-.', linewidth=8, label='Suburb')
if cc == 3:
    ax2.legend(handles=[center_line, neighbor_line, outlier_dot], loc='upper left',bbox_to_anchor=(-0.2, 1),frameon=True, fontsize = s_font_legend-20)

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
handles3, labels3 = ax3.get_legend_handles_labels()

for handle in handles1:
    handle.set_alpha(1)
for handle in handles2:
    handle.set_alpha(1)
for handle in handles3:
    handle.set_alpha(1)

fig.subplots_adjust(bottom=0.3)
cbar_ax1 = fig.add_axes([0.14+0.14, 0.25, 0.13, 0.02])
cbar_ax2 = fig.add_axes([0.14+0.34-0.04, 0.25, 0.13, 0.02])
cbar_ax3 = fig.add_axes([0.14+0.54-0.08, 0.25, 0.13, 0.02])
cbar_ax4 = fig.add_axes([0.14+0.74-0.12, 0.25, 0.13, 0.02])

sm1 = plt.cm.ScalarMappable(cmap=plt.get_cmap(colors[0]), norm=norm)
sm2 = plt.cm.ScalarMappable(cmap=plt.get_cmap(colors[1]), norm=norm)
sm3 = plt.cm.ScalarMappable(cmap=plt.get_cmap(colors[2]), norm=norm)
sm4 = plt.cm.ScalarMappable(cmap=plt.get_cmap(colors[3]), norm=norm)

cbar1 = fig.colorbar(sm1, cax=cbar_ax1, orientation='horizontal')
cbar1.ax.tick_params(labelsize = s_font_label-5)
cbar1.ax.xaxis.set_ticks_position('top')
vmin, vmax = data['Capacity-2030(KW)'].min()/1000, 0.5*data['Capacity-2050(KW)'].max()/1000
cbar1.set_ticks([0, 80])
cbar1.set_ticklabels([f'{0}', f'{80}'])
cbar1.set_label(types[0]+'\n'+str(Data_ratio[0,0])+'\u2192'+str(Data_ratio[0,1])+'\u2192'+str(Data_ratio[0,2]), fontsize = s_font_label_title-5, labelpad=35)
#cbar1.set_label(types[0], fontsize = s_font_label_title-5, labelpad=35)

cbar2 = fig.colorbar(sm2, cax=cbar_ax2, orientation='horizontal')
cbar2.ax.tick_params(labelsize = s_font_label-5)
cbar2.ax.xaxis.set_ticks_position('top')
cbar2.set_ticks([0, 80])
cbar2.set_ticklabels([f'{0}', f'{80}'])
cbar2.set_label(types[1]+'\n'+str(Data_ratio[1,0])+'\u2192'+str(Data_ratio[1,1])+'\u2192'+str(Data_ratio[1,2]), fontsize = s_font_label_title-5, labelpad=35)
#cbar2.set_label(types[1], fontsize = s_font_label_title-5, labelpad=35)

cbar3 = fig.colorbar(sm3, cax=cbar_ax3, orientation='horizontal')
cbar3.ax.tick_params(labelsize = s_font_label-5)
cbar3.ax.xaxis.set_ticks_position('top')
cbar3.set_ticks([0, 80])
cbar3.set_ticklabels([f'{0}', f'{80}'])
cbar3.set_label(types[2]+'\n'+str(Data_ratio[2,0])+'\u2192'+str(Data_ratio[2,1])+'\u2192'+str(Data_ratio[2,2]), fontsize = s_font_label_title-5, labelpad=35)
#cbar3.set_label(types[2], fontsize = s_font_label_title-5, labelpad=35)

cbar4 = fig.colorbar(sm4, cax=cbar_ax4, orientation='horizontal')
cbar4.ax.tick_params(labelsize = s_font_label-5)
cbar4.ax.xaxis.set_ticks_position('top')
cbar4.set_ticks([0, 80])
cbar4.set_ticklabels([f'{0}', f'{80}'])
cbar4.set_label(types[3]+'\n'+str(Data_ratio[3,0])+'\u2192'+str(Data_ratio[3,1])+'\u2192'+str(Data_ratio[3,2]), fontsize = s_font_label_title-5, labelpad=35)
#cbar4.set_label(types[3], fontsize = s_font_label_title-5, labelpad=35)

ax1.text(0.25, -0.07, 'Capacity (MW)', fontsize=s_font_label, ha='center', va='center', transform=ax1.transAxes)
ax1.text(0.25, -0.23, 'Capacity ratio (%)', fontsize=s_font_label, ha='center', va='center', transform=ax1.transAxes)
# ax1.text(-0.18, 0.55, 'Capacity ratio\n\nHnD\n\nHnS\n\nMnD\n\nMnS', fontsize=s_font_label, ha='center', va='center', transform=ax1.transAxes)
# ax1.text(1, 0.55, ' \n\n'+str(Data_ratio[0,0])+'\n\n'+str(Data_ratio[1,0])+'\n\n'+str(Data_ratio[2,0])+'\n\n'+str(Data_ratio[3,0]), fontsize=s_font_label, ha='center', va='center', transform=ax1.transAxes)
# ax2.text(1, 0.55, ' \n\n'+str(Data_ratio[0,1])+'\n\n'+str(Data_ratio[1,1])+'\n\n'+str(Data_ratio[2,1])+'\n\n'+str(Data_ratio[3,1]), fontsize=s_font_label, ha='center', va='center', transform=ax2.transAxes)
# ax3.text(1, 0.55, ' \n\n'+str(Data_ratio[0,2])+'\n\n'+str(Data_ratio[1,2])+'\n\n'+str(Data_ratio[2,2])+'\n\n'+str(Data_ratio[3,2]), fontsize=s_font_label, ha='center', va='center', transform=ax3.transAxes)

if cc == 59:
    ttx = 'c'
elif cc == 74:
    ttx = 'b'
elif cc == 3:
    ttx = 'a'

ax1.text(0.0, 1.1, ttx,fontweight='bold', fontsize=s_font_label+40, ha='center', va='center', transform=ax1.transAxes)

fig.savefig('Figs_new/Fig5-'+city_name+'.pdf',format='pdf',dpi=600,bbox_inches='tight')
fig.savefig('Figs_new/Fig5-'+city_name+'.png', dpi=600,bbox_inches='tight')
plt.show()


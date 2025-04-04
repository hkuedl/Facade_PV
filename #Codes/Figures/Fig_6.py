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
s_font = 16
C1 = path_cap+'/Cap_facade_'+city_name+'.npy'
C2 = path_cap+'/Cap_roof_'+city_name+'.npy'

if cc in [97,92,2,6,8,69]:
    data_path = city_name+'_hybrid_n.mat'
else:
    data_path = city_name+'_hybrid.mat'

G_type = np.load(city_name+'.npy')
Feas_read_sta = np.load(city_name+'_ALL_Featuers.npy')[:,[i for i in range(14)]+[15,16]]
indices_non_zero = np.where(Feas_read_sta[:,11] != 0)[0]
WWR = np.load(city_name+'_ALL_Featuers.npy')[indices_non_zero,13:14]
N_grid = G_type.shape[0]
N_gg = np.where(G_type[:,0] != 888)[0]

Data = loadmat(data_path)
R_Cap_r = Data['R_Cap_r']#[list_form[0][:],:,:]  #np.zeros((N_grid, 2, Y))
R_Cap_s = Data['R_Cap_s']#[list_form[0][:],:,:]  #np.zeros((N_grid, 2, Y))
R_Cap_f = Data['R_Cap_f']#[list_form[0][:],:,:]  #np.zeros((N_grid, 3, Y))
R_Ele = Data['R_Ele']      #np.zeros((N_grid, Y))
R_Pow = Data['R_Pow']      #np.zeros((N_grid, Y, D, T))
R_Pow_f = Data['R_Pow_f']#[list_form[0][:],:,:,:]  #np.zeros((N_grid,Y, D, T))
R_Pow_ch = Data['R_Pow_ch']  #np.zeros((N_grid,2,Y, D, T))
R_Pow_dis = Data['R_Pow_dis']  #np.zeros((N_grid,2,Y, D, T))
R_Pow_G = Data['R_Pow_G']     #np.zeros((N_grid,2,Y, D, T))
R_Pow_r = Data['R_Pow_r']#[list_form[0][:],:,:,:,:]  #np.zeros((N_grid, 2, Y, D, T))
R_Pow_Buy = Data['R_Pow_Buy']#[list_form[0][:],:,:,:,:]  #np.zeros((N_grid,2,Y, D, T))
R_Pow_AB = Data['R_Pow_AB']   #np.zeros((N_grid,2,Y, D, T))
R_Car = Data['R_Car']     #np.zeros((N_grid,2,Y))
R_Car_t = Data['R_Car_t']#[list_form[0][:],:,:,:,:]  #np.zeros((N_grid,2,Y,D,T))
# R_Cost = Data['R_Cost']    #np.zeros((N_grid,2,Y))

labels = [['RPV', 'Storage'],['RPV', 'FPV', 'Storage'],['RPV', 'Grid'],['RPV', 'FPV', 'Grid']]
sizes_list = [
    [np.sum(R_Cap_r[:,0,-1]), np.sum(R_Cap_s[:,0,-1])],
    [np.sum(R_Cap_r[:,1,-1]), np.sum(R_Cap_f[:,0,-1]), np.sum(R_Cap_s[:,1,-1])],
    [np.sum(R_Pow_r[:,0,:,:,:]), np.sum(R_Pow_Buy[:,0,:,:,:])],
    [np.sum(R_Pow_r[:,1,:,:,:]), np.sum(R_Pow_f[:,:,:,:]), np.sum(R_Pow_Buy[:,1,:,:,:])]]

categories = [f"{i}h" for i in range(24)]

list_form = np.load('list_form_'+city_name+'.npy')

PU_car = np.max(np.sum(R_Car_t[list_form,0,:,:,:],axis = (0)))

if cc == 3: #4-2;3-3;2-2;
    i_row = 2
elif cc == 74:
    i_row = 11
elif cc == 59:
    i_row = 7
values1 = np.sum(R_Car_t[list_form,0,:,:,:],axis = (0,1,2))/(12*5)/PU_car
values2 = np.sum(R_Car_t[list_form,1,:,:,:],axis = (0,1,2))/(12*5)/PU_car

min_data = np.min(np.sum(R_Car_t[list_form,0,:,:,:],axis = (0,1))/5,axis = 0)/PU_car
max_data = np.max(np.sum(R_Car_t[list_form,0,:,:,:],axis = (0,1))/5,axis = 0)/PU_car
min_datax = np.min(np.sum(R_Car_t[list_form,1,:,:,:],axis = (0,1))/5,axis = 0)/PU_car
max_datax = np.max(np.sum(R_Car_t[list_form,1,:,:,:],axis = (0,1))/5,axis = 0)/PU_car

x = np.arange(0, 24, 1)
fig, ax = plt.subplots(figsize=(10, 6))
ax.fill_between(x, min_data, max_data, color='lightblue', alpha=0.4)
ax.fill_between(x, min_datax, max_datax, color='lightgreen', alpha=0.4)
ax.plot(x, values1, label='RS', color='blue', linestyle='-', marker='o', markersize=5, linewidth=2, alpha=0.7)
ax.plot(x, values2, label='RS+F', color='green', linestyle='-', marker='s', markersize=5, linewidth=2, alpha=0.7)
ax.set_xlabel('Time', fontsize=s_font)
ax.set_ylabel('Carbon emissions (p.u.)', fontsize=s_font)
ax.set_xticks(np.arange(0, 25, 6))
ax.set_xticklabels(['0:00', '6:00', '12:00', '18:00', '24:00'],fontsize = s_font-2)
ax.tick_params(axis='y', labelsize=s_font-2)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper left', fontsize=s_font)
plt.tight_layout()
fig.savefig('Figs/Fig6-1'+city_name+'.pdf',format='pdf',dpi=600)
fig.savefig('Figs/Fig6-1'+city_name+'.png', dpi=600)
plt.show()


hours = np.arange(0, 24)
months = np.arange(0, 4)

i_index = 1
PU_ele = np.max(np.sum(R_Pow[list_form,:,:,:],axis = 0))
load_curve = np.sum(R_Pow[list_form,:,:,:],axis = (0,1,2))/(12*5)/PU_ele
RPV_output = np.sum(R_Pow_r[list_form,i_index,:,:,:],axis = (0,1,2))/(12*5)/PU_ele
FPV_output = i_index*np.sum(R_Pow_f[list_form,:,:,:],axis = (0,1,2))/(12*5)/PU_ele
St_ch = np.sum(R_Pow_ch[list_form,i_index,:,:,:],axis = (0,1,2))/(12*5)/PU_ele
St_dis = np.sum(R_Pow_dis[list_form,i_index,:,:,:],axis = (0,1,2))/(12*5)/PU_ele
grid_purchase = np.sum(R_Pow_Buy[list_form,i_index,:,:,:],axis = (0,1,2))/(12*5)/PU_ele
grid_discard = np.sum(R_Pow_AB[list_form,i_index,:,:,:],axis = (0,1,2))/(12*5)/PU_ele
grid_sell = np.sum(R_Pow_G[list_form,i_index,:,:,:],axis = (0,1,2))/(12*5)/PU_ele

fig, axs = plt.subplots(2, 1, figsize=(10, 12))
axs[0].set_position([0.15, 0.23, 0.6, 0.4])
axs[1].set_position([0.15, 0.05, 0.6, 0.15])
colors = plt.cm.tab20.colors
axs[0].plot(hours, load_curve, label='Load', color='black', linestyle = '--' ,linewidth=2)
axs[0].stackplot(hours, RPV_output, FPV_output, grid_purchase, St_dis,
              labels=['RPV', 'FPV', 'Purchase', 'Discharge'], colors=colors)
#axs[0].set_title('Load balance with FPV', fontsize=s_font)
axs[0].set_ylabel('Generation (p.u.)', fontsize=s_font)
legend = axs[0].legend(
    loc='center left', 
    bbox_to_anchor=(1.02, 0.5), 
    ncol=1,                    
    frameon=False,           
    fontsize=s_font,         
    labelspacing=1.5    
)
axs[0].grid(True, linestyle='--', alpha=0.5)
axs[0].set_xticks(np.arange(0, 25, 6))
axs[0].set_xticklabels([])
axs[0].tick_params(axis='y', labelsize=s_font-2)

axs[1].stackplot(hours, -St_ch, -grid_discard, -grid_sell,
              labels=['Charge', 'Discard', 'Sell'], colors=colors[4:7],alpha = 0.5)
axs[1].set_xlabel('Time', fontsize=s_font)
axs[1].set_ylabel('Sink (p.u.)', fontsize=s_font)
legend = axs[1].legend(
    loc='center left',       
    bbox_to_anchor=(1.02, 0.5), 
    ncol=1,             
    frameon=False,     
    fontsize=15,     
    labelspacing=1.5    
)
axs[1].grid(True, linestyle='--', alpha=0.5)
axs[1].tick_params(axis='y', labelsize=s_font-2)
axs[1].set_xticks(np.arange(0, 25, 6))
axs[1].set_xticklabels(['0:00', '6:00', '12:00', '18:00', '24:00'],fontsize = s_font-2)
fig.savefig('Figs/Fig6-2'+city_name+'.pdf',format='pdf',dpi=600)
fig.savefig('Figs/Fig6-2'+city_name+'.png', dpi=600)
plt.show()


#%% 
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

hours = np.arange(0, 24)
months = np.arange(0, 12)

for i_index in [0,1]:
    load_curve = (np.sum(R_Pow[list_form,:,:,:],axis = (0,1))/(5)/PU_ele)[[3,4,5,6,7,8,9,10,11,0,1,2]]
    RPV_output = (np.sum(R_Pow_r[list_form,i_index,:,:,:],axis = (0,1))/(5)/PU_ele)[[3,4,5,6,7,8,9,10,11,0,1,2]]
    FPV_output = (i_index*np.sum(R_Pow_f[list_form,:,:,:],axis = (0,1))/(5)/PU_ele)[[3,4,5,6,7,8,9,10,11,0,1,2]]
    St_ch = (np.sum(R_Pow_ch[list_form,i_index,:,:,:],axis = (0,1))/(5)/PU_ele)[[3,4,5,6,7,8,9,10,11,0,1,2]]
    St_dis = (np.sum(R_Pow_dis[list_form,i_index,:,:,:],axis = (0,1))/(5)/PU_ele)[[3,4,5,6,7,8,9,10,11,0,1,2]]
    grid_purchase = (np.sum(R_Pow_Buy[list_form,i_index,:,:,:],axis = (0,1))/(5)/PU_ele)[[3,4,5,6,7,8,9,10,11,0,1,2]]
    grid_discard = (np.sum(R_Pow_AB[list_form,i_index,:,:,:],axis = (0,1))/(5)/PU_ele)[[3,4,5,6,7,8,9,10,11,0,1,2]]
    grid_sell = (np.sum(R_Pow_G[list_form,i_index,:,:,:],axis = (0,1))/(5)/PU_ele)[[3,4,5,6,7,8,9,10,11,0,1,2]]

    title_name = ['Spring','Summer','Autumn','Winter']

    fig = plt.figure(figsize=(15, 20))
    gs = GridSpec(5, 3, figure=fig, height_ratios=[1, 1, 1, 1, 0.1]) 

    for i in range(4): 
        for j in range(3): 
            ii = i*3+j

            inner_gs = gs[i, j].subgridspec(2, 1, height_ratios=[1, 1])

            ax1 = fig.add_subplot(inner_gs[0])
            ax2 = fig.add_subplot(inner_gs[1])

            ax1.set_position([0.32*j+0.07, 0.21*(4-i), 0.23, 0.12])  
            ax2.set_position([0.32*j+0.07, 0.21*(4-i)-0.04, 0.23, 0.03]) 
            colors = plt.cm.tab20.colors  
            ax1.plot(hours, load_curve[ii,:], label='Load', color='black', linestyle = '--' ,linewidth=2)
            ax1.stackplot(hours, RPV_output[ii,:], FPV_output[ii,:], grid_purchase[ii,:], St_dis[ii,:],
                        labels=['RPV', 'FPV', 'Purchase', 'Storage Discharge'], colors=colors[:4])
            title_ii = ii//3
            title_jj = ii%3
            ax1.set_title(title_name[title_ii] + '_' + str(title_jj+1) , fontsize=s_font)
            ax1.set_ylabel('Generation (p.u.)', fontsize=s_font)
            ax1.grid(True, linestyle='--', alpha=0.5)
            ax1.set_xticks(np.arange(0, 25, 6))
            ax1.set_xticklabels([])
            ax1.tick_params(axis='y', labelsize=s_font-3)
            
            ax2.stackplot(hours, -St_ch[ii,:], -grid_discard[ii,:], -grid_sell[ii,:],
                        labels=['Storage Charge', 'Discard', 'Sell'], colors=colors[4:7],alpha = 0.5)
            ax2.set_xlabel('Time', fontsize=s_font)
            ax2.set_ylabel('Sink (p.u.)', fontsize=s_font)
            ax2.set_ylim(-0.2,0)
            ax2.tick_params(axis='y', labelsize=s_font-3)
            ax2.set_xticks(np.arange(0, 25, 6))
            ax2.set_xticklabels(['0:00', '6:00', '12:00', '18:00', '24:00'],fontsize = s_font-2)
            ax2.grid(True, linestyle='--', alpha=0.5)

    legend_ax = fig.add_subplot(gs[4, :])
    labels_ele = ['RPV', 'FPV', 'Purchase', 'Storage Discharge','Storage Charge', 'Discard', 'Sell']
    if i_index == 0:
        elements = []
        for iii in [0,2,3,4,5,6]:
            elements.append(mpatches.Patch(color=colors[iii], label=labels_ele[iii]))
    else:
        elements = []
        for iii in [0,1,2,3,4,5,6]:
            elements.append(mpatches.Patch(color=colors[iii], label=labels_ele[iii]))
    load_line = mlines.Line2D([], [], color='black', linestyle='--', label='Load')
    legend_ax.legend(handles=[load_line]+elements,loc='center', ncol=4, fontsize=s_font,bbox_to_anchor=(0.5, -0.0))
    legend_ax.axis('off')
    fig.savefig('Figs/SFig6-'+str(i_index+1)+city_name+'.pdf',format='pdf',dpi=600)
    fig.savefig('Figs/SFig6-'+str(i_index+1)+city_name+'.png', dpi=600)
    plt.show()

#%%
values1 = (np.sum(R_Car_t[list_form,0,:,:,:],axis = (0,1))/(5)/PU_car)[[3,4,5,6,7,8,9,10,11,0,1,2]]
values2 = (np.sum(R_Car_t[list_form,1,:,:,:],axis = (0,1))/(5)/PU_car)[[3,4,5,6,7,8,9,10,11,0,1,2]]

fig = plt.figure(figsize=(15, 15))
gs = GridSpec(5, 3, figure=fig, height_ratios=[1, 1, 1, 1, 0.1]) 
for i in range(4): 
    for j in range(3): 
        ii = i*3+j
        title_ii = ii//3
        title_jj = ii%3
        ax = fig.add_subplot(gs[i, j]) 
        x = np.arange(0, 24, 1)
        ax.plot(x, values1[ii,:], label='RS', color='blue', linestyle='-', marker='o', markersize=5, linewidth=2, alpha=0.7)
        ax.plot(x, values2[ii,:], label='RS+F', color='green', linestyle='-', marker='s', markersize=5, linewidth=2, alpha=0.7)
        ax.set_title(title_name[title_ii] + '_' + str(title_jj+1) , fontsize=s_font)
        ax.set_xlabel('Time', fontsize=14)
        ax.set_ylabel('Carbon emissions (p.u.)', fontsize=s_font)
        #ax.set_xlim(0, 10)
        #ax.set_ylim(0, 20000)
        ax.tick_params(axis='y', labelsize=s_font-2)
        ax.set_xticks(np.arange(0, 25, 6))
        ax.set_xticklabels(['0:00', '6:00', '12:00', '18:00', '24:00'],fontsize = s_font-2)
        ax.grid(True, linestyle='--', alpha=0.5)
legend_ax = fig.add_subplot(gs[4, :]) 
elements = [mlines.Line2D([], [], color='blue', linestyle='-', label='RS'),mlines.Line2D([], [], color='green', linestyle='-', label='RS+F')]
legend_ax.legend(handles=elements,loc='center', ncol=2, fontsize=s_font,bbox_to_anchor=(0.5, -0.0))
legend_ax.axis('off')
plt.tight_layout()
fig.savefig('Figs/SFig6-3'+city_name+'.pdf',format='pdf',dpi=600)
fig.savefig('Figs/SFig6-3'+city_name+'.png', dpi=600)
plt.show()
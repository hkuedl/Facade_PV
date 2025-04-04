#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Functions
import pyomo.environ as pyo
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
import time
from scipy.io import savemat,loadmat
import os

path_type = 'Power'
path_cap = 'Capacity'

City_statistic = pd.read_excel('City_statistic.xlsx',sheet_name = 'Class_Volume', index_col=0)
Statis_all = pd.read_excel('City_statistic.xlsx',sheet_name = 'Key_information', index_col=0)

Clu_center,Clu_days = np.load('Clu_center.npy'),np.load('Clu_days.npy')

Total_Cap_total = np.zeros((102,2,5))
Total_Ele = np.zeros((102,5))
Total_Cost_total = np.zeros((102,6,5))
Total_price = np.zeros((102,2,5))
Total_price_true = np.zeros((102,2,5))
Total_Carbon = np.zeros((102,2,5))

Total_Carbon_00 = np.zeros((102,2,5))
Total_price_00 = np.zeros((102,2,5))
Total_price_00_true = np.zeros((102,2,5))

Total_area = np.zeros((102,1))

for cc in range(102):
    city_name = City_statistic.index[cc]
    print(city_name)
    C1 = path_cap+'/Cap_facade_'+city_name+'.npy'
    C2 = path_cap+'/Cap_roof_'+city_name+'.npy'
    G_type = np.load('Grid_type_'+city_name+'.npy')
    Feas_read_sta = np.load(city_name+'_ALL_Featuers.npy')[:,[i for i in range(14)]+[15,16]]
    indices_non_zero = np.where(Feas_read_sta[:,11] != 0)[0]
    WWR = np.load(city_name+'_ALL_Featuers.npy')[indices_non_zero,13:14]
    N_grid = G_type.shape[0]
    N_gg = np.where(G_type[:,0] != 888)[0]
    Total_area[cc,0] = np.sum(G_type[:,4])/1e6 #(km2)
    if cc in [97,92,2,6,8,69]:
        data_path = city_name+'_hybrid_n.mat'
    else:
        data_path = city_name+'_hybrid.mat'

    Data = loadmat(data_path)
    R_Cap_r = Data['R_Cap_r']  #np.zeros((N_grid, 2, Y))
    
    R_Cap_f = Data['R_Cap_f']  #np.zeros((N_grid, 3, Y))
    R_Ele = Data['R_Ele']      #np.zeros((N_grid, Y))
    #R_Pow = Data['R_Pow']      #np.zeros((N_grid, Y, D, T))
    R_Pow_f = Data['R_Pow_f']  #np.zeros((N_grid,Y, D, T))
    #R_Pow_ch = Data['R_Pow_ch']  #np.zeros((N_grid,2,Y, D, T))
    #R_Pow_dis = Data['R_Pow_dis']  #np.zeros((N_grid,2,Y, D, T))
    #R_Pow_G = Data['R_Pow_G']     #np.zeros((N_grid,2,Y, D, T))
    R_Pow_r = Data['R_Pow_r']  #np.zeros((N_grid, 2, Y, D, T))
    #R_Pow_Buy = Data['R_Pow_Buy']  #np.zeros((N_grid,2,Y, D, T))
    #R_Pow_AB = Data['R_Pow_AB']   #np.zeros((N_grid,2,Y, D, T))
    R_Car = Data['R_Car']     #np.zeros((N_grid,2,Y))
    #R_Car_t = Data['R_Car_t']  #np.zeros((N_grid,2,Y,D,T))
    R_Cost = Data['R_Cost']    #np.zeros((N_grid,2,Y))
    R_Cost_true = Data['R_Cost_true']    #np.zeros((N_grid,2,Y))

    Total_Cap_total[cc,:,:] = np.sum(R_Cap_f[:,[0,2],:],axis = 0)
    Total_Carbon[cc,:,:] = 1e3*np.sum(R_Car[:,:,:],axis = 0)/np.sum(R_Ele[:,:],axis = 0)
    Total_price[cc,:,:] = np.sum(R_Cost[:,:,:],axis = 0)/np.sum(R_Ele[:,:],axis = 0)
    Total_price_true[cc,:,:] = np.sum(R_Cost_true[:,:,:],axis = 0)/np.sum(R_Ele[:,:],axis = 0)
    Total_Carbon_00[cc,:,:] = np.sum(R_Car[:,:,:],axis = 0)
    Total_price_00[cc,:,:] = np.sum(R_Cost[:,:,:],axis = 0)
    Total_price_00_true[cc,:,:] = np.sum(R_Cost_true[:,:,:],axis = 0)

    wall_0 = [3.18]+[2.98,2.75,2.70,2.65,2.60]
    win_0 = [4.80]+[4.50,4.15,4.08,4.00,3.92]
    K1_pri_wall = [1e3*wall_0[i] for i in range(6)]
    K1_pri_win  = [1e3*win_0[i] for i in range(6)]
    year_op = 25
    K3_D_fa_id = [np.zeros((len(N_gg),12,24)) for _ in range(5)]
    for case in range(5):
        Total_Ele[cc,case] = sum(Clu_days[cc,d]*np.sum(R_Pow_f[:,case,d,:],axis = (0,1)) for d in range(12))
    
        Total_Cost_total[cc,0,case] = np.sum((K1_pri_wall[case+1]*(1-WWR[:,0])+K1_pri_win[1+case]*WWR[:,0])*R_Cap_f[:,0,case])
        Total_Cost_total[cc,1,case] = (Total_Cost_total[cc,0,case]+sum(0.04*Total_Cost_total[cc,0,case]/(1.08)**i for i in range(1,year_op+1)))/sum(Total_Ele[cc,case]/(1.08)**i for i in range(1,year_op+1))

        Total_Cost_total[cc,2,case] = np.sum(K1_pri_wall[case+1]*R_Cap_r[:,1,case])
        Total_Ele_r = sum(Clu_days[cc,d]*np.sum(R_Pow_r[:,1,case,d,:],axis = (0,1)) for d in range(12))
        Total_Cost_total[cc,3,case] = (Total_Cost_total[cc,2,case]+sum(0.02*Total_Cost_total[cc,2,case]/(1.08)**i for i in range(1,year_op+1)))/sum(Total_Ele_r/(1.08)**i for i in range(1,year_op+1))
        
        P1 = path_type+str(case+2)+'/N_P_facade_ideal_'+str(case+2)+'_'+city_name+'.npy'
        for i in range(12):
            K3_D_fa_id[case][:,i,:] = (1/1e-3)*np.load(P1)[N_gg,(Clu_center[cc,i])*24:(Clu_center[cc,i]+1)*24]
        Total_Cost_total[cc,4,case] = np.sum((K1_pri_wall[case+1]*(1-WWR[N_gg,0])+K1_pri_win[case+1]*WWR[N_gg,0])*R_Cap_f[N_gg,2,case])
        Total_Ele_f = sum(Clu_days[cc,d]*np.sum(K3_D_fa_id[case][:,d,:],axis = (0,1)) for d in range(12))
        Total_Cost_total[cc,5,case] = (Total_Cost_total[cc,4,case]+sum(0.04*Total_Cost_total[cc,4,case]/(1.08)**i for i in range(1,year_op+1)))/sum(Total_Ele_f/(1.08)**i for i in range(1,year_op+1))

    print(Total_Cost_total[cc,1,-1])
    print(Total_Cost_total[cc,3,-1])
    print(Total_Cost_total[cc,5,-1])



#%%

s_font,s_title = 16,14
list_all_city_name,list_all_city_j,list_all_city_ii = [],[],[]
list_all_city_order = np.zeros((102,3))
list_all_city_order[:,0] = np.arange(102)
for i in range(4):
    indices = Statis_all.index[Statis_all.iloc[:,-1] == ['超大城市','特大城市','I型大城市','II型大城市'][i]]
    for index in indices:
        i_loc = Statis_all.index.get_loc(index)
        list_all_city_order[i_loc,2] = 4-i
        list_all_city_order[i_loc,1] = Total_Cap_total[i_loc,0,-1]/1e6
sorted_indices = np.lexsort((-list_all_city_order[:, 1], -list_all_city_order[:, 2]))
data_sorted = list_all_city_order[sorted_indices]
list_all_city_j = list(data_sorted[:,0].astype(int))
list_all_city_ii = [list_all_city_j[:7],list_all_city_j[7:22],list_all_city_j[22:35],list_all_city_j[35:]]
list_all_city_name = City_statistic.index[list_all_city_j].tolist()

cities = City_statistic.index.tolist()

for name in range(len(cities)):
    if cities[name] == 'Haerbin':
        cities[name] = 'Harbin'
    elif cities[name] == 'Huhehaote':
        cities[name] = 'Hohhot'
    elif cities[name] == 'Wulumuqi':
        cities[name] = 'Urumqi'
    elif cities[name] == 'Xian':
        cities[name] = "Xi'an"

cities_re = [cities[i] for i in list_all_city_j]
wall_actual = Total_Cap_total[list_all_city_j,0,:]/1e6
LCOE_actual = Total_Cost_total[list_all_city_j,1,:]

import matplotlib.colors as mcolors
cmap = mcolors.LinearSegmentedColormap.from_list("green_gradient", ['#E0F2E9', '#007E2E'])
colors = [cmap(i / 4) for i in range(5)]
cmap = mcolors.LinearSegmentedColormap.from_list("blue_gradient", ['#E6F0FF', '#003366'])
colors111 = [cmap(i / 4) for i in range(5)]

data_splits = [51, 51]
split_indices = np.cumsum(data_splits)
fig, axes = plt.subplots(1, 2, figsize=(40, 42))
for i, ax in enumerate(axes):
    start_idx = 0 if i == 0 else split_indices[i-1]
    end_idx = split_indices[i]
    
    cities_re_split = cities_re[start_idx:end_idx]
    wall_actual_split = wall_actual[start_idx:end_idx,:]
    LCOE_actual_split = LCOE_actual[start_idx:end_idx,:]
    
    ind = np.arange(len(cities_re_split))
    max_len = max(data_splits)
    bar_width = 1.2
    city_gap = 2
    bottom = np.zeros(51)
    for tt in range(5):
        ax.barh((max_len - ind - 1) * city_gap, wall_actual_split[:,tt]-bottom, bar_width, left = bottom,  label='Planned capacity', color=colors[tt])
        bottom = wall_actual_split[:,tt]
    if i == 0:
        ax.set_xlim(0,38)
    else:
        ax.set_xlim(0,38)
    ax.set_yticks((max_len - ind - 1) * city_gap)
    ax.set_yticklabels(cities_re_split, fontsize=s_font+20)
    ax.tick_params(axis='x', labelsize=s_font+20)
    ax.tick_params(axis='y', labelsize=s_font+20)
    ax.xaxis.tick_top()
    ax.set_xlabel('GW', fontsize=s_font+30, labelpad=20)

    ax2 = ax.twiny()
    for tt in range(5):
        ax2.scatter(LCOE_actual_split[:,tt], (max_len - ind - 1) * city_gap, color = colors111[tt], label='LCOE of planned FPV in 2050', zorder=5,s=1200)
    ax2.set_xlim(0, 0.8)
    ax2.tick_params(axis='x', labelsize=s_font + 20)
    ax2.set_xlabel('CNY/kWh', fontsize=s_font+30 , labelpad=20)

import matplotlib.patches as mpatches
lg = fig.legend(
    handles=[
        mpatches.Patch(fc=colors[-1], label='Planned capacity (2030-2050)'),
        plt.Line2D([0], [0], marker='o', color=colors111[-1],
                   markersize=37, label='LCOE (2030-2050)', linestyle='none')
    ],
    handleheight=0.7,
    handlelength=2,
    loc='lower center',
    ncol=2,
    fontsize=s_font+30,
    bbox_to_anchor=(0.5, 0.02)
)

frame = lg.get_frame()
frame.set_linewidth(0.6)
frame.set_edgecolor('black')
frame.set_facecolor('none')

for ax in axes:
    ax.set_ylim(-1, max_len * city_gap)
    ax.set_yticks(np.arange(max_len) * city_gap)
plt.subplots_adjust(wspace=0.3)
fig.savefig('Figs/Fig4a.pdf',format='pdf',dpi=600)
fig.savefig("Figs/Fig4a.png", dpi=600)
plt.show()

#%% 
from matplotlib.patches import Patch
aaCar_wo,aaCar_ww = [],[]
aaPri_woFPV,aaPri_wFPV = [],[]
for i in range(4):
    aaCar_wo.append(np.sum(Total_Carbon[list_all_city_ii[i],0,:],axis=1)/5)
    aaCar_ww.append(np.sum(Total_Carbon[list_all_city_ii[i],1,:],axis=1)/5)
    aaPri_woFPV.append(np.sum(Total_price_true[list_all_city_ii[i],0,:],axis=1)*1e10/5)
    aaPri_wFPV.append(np.sum(Total_price_true[list_all_city_ii[i],1,:],axis=1)*1e10/5)

fig, ax = plt.subplots(1,2,figsize=(20,6))
ax[0].boxplot(aaCar_wo, positions = np.arange(4) * 2.0,widths=0.4, patch_artist=True, boxprops=dict(facecolor='skyblue'))
ax[0].boxplot(aaCar_ww, positions = np.arange(4) * 2.0+0.5,widths=0.4, patch_artist=True, boxprops=dict(facecolor='seagreen'))#, showfliers = False)
ax[0].set_xlabel('City types', fontsize=s_font+4)
ax[0].set_ylabel('Unit carbon emission (ton/MWh)',fontsize=s_font+2)
ax[0].set_xticks(ticks=np.arange(4) * 2.0+0.25, labels=['Megacity','Super-large City','Laege City I','Large City II'],fontsize=s_font)
ax[0].tick_params(axis='y', labelsize=s_font)
legend_elements = [
    Patch(facecolor='skyblue', label='RS'),
    Patch(facecolor='seagreen', label='RS+F')
]
ax[0].set_ylim(0.1,0.24)
ax[0].legend(handles=legend_elements,fontsize=s_font+2,loc='upper left',ncol=2,frameon=True)

ax[1].boxplot(aaPri_woFPV, positions = np.arange(4) * 2.0,widths=0.4, patch_artist=True, boxprops=dict(facecolor='skyblue'), labels=[""] * 4, showfliers = False)
ax[1].boxplot(aaPri_wFPV, positions = np.arange(4) * 2.0+0.5,widths=0.4, patch_artist=True, boxprops=dict(facecolor='seagreen'), labels=[""] * 4, showfliers = False)
ax[1].set_xlabel('City types', fontsize=s_font+4)
ax[1].set_ylabel('Unit cost (CNY/kWh)',fontsize=s_font+2)
ax[1].tick_params(axis='y', labelsize=s_font)
ax[1].set_xticks(ticks=np.arange(4) * 2.0+0.25, labels=['Megacity','Super-large City','Laege City I','Large City II'],fontsize=s_font-0)
ax[1].set_ylim(0.36,0.65)
ax[1].legend(handles=legend_elements,fontsize=s_font+2,loc='upper left',ncol=2,frameon=True)

fig.savefig('Figs/Fig4b.pdf',format='pdf',dpi=600)
fig.savefig("Figs/Fig4b.png", dpi=600)
plt.show()

print((aaCar_wo[1]-aaCar_ww[1])/aaCar_wo[1])


#%%
F_fig1 = np.zeros((102,1))
F_table1 = pd.DataFrame(np.zeros((102,5)), columns = ['City','Planned FPV capacity (GW)','Planned FPV generation (TWh)', 'Carbon emission reduction (Million ton)','Economic cost reduction (Million CNY)'])
F_table2 = pd.DataFrame(np.zeros((102,4)), columns = ['City','Planned RPV capacity (GW)','Planned storage capacity (GW)', 'Purchased electricity (TWh)'])
F_table3 = pd.DataFrame(np.zeros((102,4)), columns = ['City','Planned RPV capacity (GW)','Planned storage capacity (GW)', 'Purchased electricity (TWh)'])
F_table23 = pd.DataFrame(np.zeros((102,4)), columns = ['City','Planned RPV capacity (GW)','Planned storage capacity (GW)', 'Purchased electricity (TWh)'])
F_price = np.zeros((102,1))
mon_d = [31,28,31,30,31,30,31,31,30,31,30,31]
for cc in range(102):  #[3,15,56,59]:  #range(3,4):
    city_name = City_statistic.index[cc]
    print(city_name)
    C1 = path_cap+'/Cap_facade_'+city_name+'.npy'
    C2 = path_cap+'/Cap_roof_'+city_name+'.npy'
    if cc in [97,92,2,6,8,69]:
        data_path = city_name+'_hybrid_n.mat'
    else:
        data_path = city_name+'_hybrid.mat'
    Data = loadmat(data_path)
    F_table1.iloc[cc,0] = city_name
    F_table2.iloc[cc,0] = city_name
    F_table3.iloc[cc,0] = city_name
    F_table23.iloc[cc,0] = city_name
    
    R_Cap_r = Data['R_Cap_r']  #np.zeros((N_grid, 2, Y))
    R_Cap_s = Data['R_Cap_s']  #np.zeros((N_grid, 2, Y))
    R_Cap_f = Data['R_Cap_f']  #np.zeros((N_grid, 3, Y))
    R_Ele = Data['R_Ele']      #np.zeros((N_grid, Y))
    R_Pow_f = Data['R_Pow_f']  #np.zeros((N_grid,Y, D, T))
    #R_Pow_ch = Data['R_Pow_ch']  #np.zeros((N_grid,2,Y, D, T))
    #R_Pow_dis = Data['R_Pow_dis']  #np.zeros((N_grid,2,Y, D, T))
    #R_Pow_G = Data['R_Pow_G']     #np.zeros((N_grid,2,Y, D, T))
    R_Pow_r = Data['R_Pow_r']  #np.zeros((N_grid, 2, Y, D, T))
    R_Pow_Buy = Data['R_Pow_Buy']  #np.zeros((N_grid,2,Y, D, T))
    #R_Pow_AB = Data['R_Pow_AB']   #np.zeros((N_grid,2,Y, D, T))
    R_Car = Data['R_Car']     #np.zeros((N_grid,2,Y))
    R_Car_t = Data['R_Car_t']  #np.zeros((N_grid,2,Y,D,T))
    R_Cost = Data['R_Cost']    #np.zeros((N_grid,2,Y))
    R_Cost_true = Data['R_Cost_true']    #np.zeros((N_grid,2,Y))

    F_table1.iloc[cc,1] = round(np.sum(R_Cap_f[:,0,-1])/1e6,2)
    F_table1.iloc[cc,2] = round(sum(5*Clu_days[cc,d]*np.sum(R_Pow_f[:,:,d,:]) for d in range(12))/1e9,2)
    F_table1.iloc[cc,3] = round(100*(np.sum(R_Car[:,1,:])-np.sum(R_Car[:,0,:]))/np.sum(R_Car[:,0,:]),2)
    F_table1.iloc[cc,4] = round(100*(np.sum(R_Cost[:,1,:])-np.sum(R_Cost[:,0,:]))/np.sum(R_Cost[:,0,:]),2)

    F_fig1[cc,0] = np.sum(R_Ele[:,:])/1e9

    F_table2.iloc[cc,1] = round(np.sum(R_Cap_r[:,0,-1])/1e6,2)
    F_table3.iloc[cc,1] = round(np.sum(R_Cap_r[:,1,-1])/1e6,2)
    F_table2.iloc[cc,2] = round(np.sum(R_Cap_s[:,0,-1])/1e6,2)
    F_table3.iloc[cc,2] = round(np.sum(R_Cap_s[:,1,-1])/1e6,2)
    F_table2.iloc[cc,3] = round(sum(5*Clu_days[cc,d]*np.sum(R_Pow_Buy[:,0,:,d,:]) for d in range(12))/1e9,2)
    F_table3.iloc[cc,3] = round(sum(5*Clu_days[cc,d]*np.sum(R_Pow_Buy[:,1,:,d,:]) for d in range(12))/1e9,2)
    for i in range(3):
        F_table23.iloc[cc,i+1] = round((F_table3.iloc[cc,i+1]-F_table2.iloc[cc,i+1])/F_table2.iloc[cc,i+1],2)
    
    _,K3_TOU_indu_i,K3_TOU_resi_i,K3_net_i,Carbon_F = Functions.TOU_period(city_name)
    F_p1 = np.vstack([np.tile(K3_TOU_indu_i[month], (days, 1)) for month, days in enumerate(mon_d)])
    F_p2 = np.vstack([np.tile(K3_TOU_resi_i[month], (days, 1)) for month, days in enumerate(mon_d)])
    F_price[cc,0] = 0.2*np.sum(F_p1) + 0.8*np.sum(F_p2)

F_table1 = F_table1.iloc[list_all_city_j]
F_table2 = F_table2.iloc[list_all_city_j]
F_table3 = F_table3.iloc[list_all_city_j]
F_table23 = F_table23.iloc[list_all_city_j]
F_fig1 = F_fig1[list_all_city_j,:]


#%% 
wall_actual0 = Total_Cap_total[list_all_city_j,0,-1]/1e6
wall_limit = Total_Cap_total[list_all_city_j,1,-1]/1e6
LCOE_actual0 = Total_Cost_total[list_all_city_j,1,-1]
LCOE_limit = Total_Cost_total[list_all_city_j,5,-1]

print(Total_Cap_total[3,:,-1])
print(np.sum(np.sum(R_Cap_r[:,1,-1],axis = 0)))


for name in range(len(cities_re)):
    if cities_re[name] == 'Haerbin':
        cities_re[name] = 'Harbin'
    elif cities_re[name] == 'Huhehaote':
        cities_re[name] = 'Hohhot'
    elif cities_re[name] == 'Wulumuqi':
        cities_re[name] = 'Urumqi'
    elif cities_re[name] == 'Xian':
        cities_re[name] = "Xi'an"

data_splits = [51, 51]
split_indices = np.cumsum(data_splits)
fig, axes = plt.subplots(1, 2, figsize=(40, 42))
for i, ax in enumerate(axes):
    start_idx = 0 if i == 0 else split_indices[i-1]
    end_idx = split_indices[i]
    
    cities_re_split = cities_re[start_idx:end_idx]
    wall_actual_split0 = wall_actual0[start_idx:end_idx]
    wall_limit_split = wall_limit[start_idx:end_idx]
    LCOE_actual_split0 = LCOE_actual0[start_idx:end_idx]
    LCOE_limit_split = LCOE_limit[start_idx:end_idx]
    
    ind = np.arange(len(cities_re_split))
    max_len = max(data_splits)
    bar_width = 1.2
    city_gap = 2
    ax.barh((max_len - ind - 1) * city_gap, wall_actual_split0, bar_width, label='Planned capacity', color='green')
    ax.barh((max_len - ind - 1) * city_gap, wall_limit_split - wall_actual_split0, bar_width, left=wall_actual_split0, label='Theoretical capacity', color='lightgreen')    
    ax.set_xlim(0,160)
    ax.set_yticks((max_len - ind - 1) * city_gap)
    ax.set_yticklabels(cities_re_split, fontsize=s_font+20)
    ax.tick_params(axis='x', labelsize=s_font+20)
    ax.tick_params(axis='y', labelsize=s_font+20)
    ax.xaxis.tick_top()
    ax.set_xlabel('GW', fontsize=s_font+30, labelpad=20)
    ax2 = ax.twiny()
    ax2.scatter(LCOE_actual_split0, (max_len - ind - 1) * city_gap, color='blue', label='LCOE of planned FPV in 2050', zorder=5,s=1200)
    ax2.scatter(LCOE_limit_split, (max_len - ind - 1) * city_gap, color='lightblue', label='LCOE of all potential FPV in 2050', zorder=5,s=1200)
    ax2.set_xlim(0, 1.5)
    ax2.tick_params(axis='x', labelsize=s_font + 20)
    ax2.set_xlabel('CNY/kWh', fontsize=s_font+30 , labelpad=20)

import matplotlib.patches as mpatches
lg = fig.legend(
    handles=[
        mpatches.Patch(fc = 'green', label='Planned capacity'),
        mpatches.Patch(fc = 'lightgreen', label='Theoretical capacity'),
        plt.Line2D([0], [0], marker='o', color='blue',
                   markersize=37, label='Planned LCOE', linestyle='none'),
        plt.Line2D([0], [0], marker='o', color='lightblue',
                   markersize=37, label='Theoretical LCOE', linestyle='none')
    ],
    handleheight=0.7,
    handlelength=2,
    loc='lower center',
    ncol=2,
    fontsize=s_font+30,
    bbox_to_anchor=(0.5, 0.0)
)

frame = lg.get_frame()
frame.set_linewidth(0.6)
frame.set_edgecolor('black')
frame.set_facecolor('none')

for ax in axes:
    ax.set_ylim(-1, max_len * city_gap)
    ax.set_yticks(np.arange(max_len) * city_gap)
plt.subplots_adjust(wspace=0.3)
fig.savefig('Figs/SFig4-3.pdf',format='pdf',dpi=600)
fig.savefig("Figs/SFig4-3.png", dpi=600)
plt.show()


#%%
import numpy as np
import matplotlib.pyplot as plt

x = F_fig1[:,0]/25
y = Total_Cap_total[list_all_city_j,0,-1]/1e6
z = F_table1.iloc[:,3]
City_order = ['red' for _ in range(1)] + ['orange' for _ in range(1)] + ['blue' for _ in range(1)] + ['green' for _ in range(6)]

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x[:7],y[:7],color = City_order[0],label = 'Megacity')
ax.scatter(x[7:22],y[7:22],color = City_order[1],label = 'Super-large City')
ax.scatter(x[22:35],y[22:35],color = City_order[2],label = 'Large city I')
ax.scatter(x[35:],y[35:],color = City_order[3],label = 'Large city II',alpha = 1.0)
ax.legend(loc = 'upper left', fontsize = s_font-2)
x0 = np.linspace(0, int(max(x)), 100)
slope, intercept = np.polyfit(x, y, 1)
fit_line = slope * x0 + intercept
ax.plot(x0, fit_line, color='grey',linestyle = '--', label='Fitted Line')
ax.set_ylabel('Planned capacity (GW)',fontsize=s_font-2)
ax.set_xlabel('Annual electricity demands (TWh)',fontsize=s_font-2)
fig.savefig('Figs/SFig4-1a.pdf',format='pdf',dpi=600)
fig.savefig("Figs/SFig4-1a.png", dpi=600)
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
Total_area_re = Total_area[list_all_city_j,0]
ax.scatter(Total_area_re[:7],y[:7],color = City_order[0],label = 'Megacity')
ax.scatter(Total_area_re[7:22],y[7:22],color = City_order[1],label = 'Super-large City')
ax.scatter(Total_area_re[22:35],y[22:35],color = City_order[2],label = 'Large city I')
ax.scatter(Total_area_re[35:],y[35:],color = City_order[3],label = 'Large city II')
ax.legend(loc = 'upper left', fontsize = s_font-2)
x0 = np.linspace(0, int(max(Total_area_re)), 100)
slope, intercept = np.polyfit(Total_area_re, y, 1)
fit_line = slope * x0 + intercept
ax.plot(x0, fit_line, color='grey',linestyle = '--', label='Fitted Line')
ax.set_ylabel('Planned capacity (GW)',fontsize=s_font-2)
ax.set_xlabel('Facade area (km2)',fontsize=s_font-2)
fig.savefig('Figs/SFig4-1b.pdf',format='pdf',dpi=600)
fig.savefig("Figs/SFig4-1b.png", dpi=600)
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y[:7],z[:7],color = City_order[0],label = 'Megacity')
ax.scatter(y[7:22],z[7:22],color = City_order[1],label = 'Super-large City')
ax.scatter(y[22:35],z[22:35],color = City_order[2],label = 'Large city I')
ax.scatter(y[35:],z[35:],color = City_order[3],label = 'Large city II')
ax.legend(loc = 'upper right', fontsize = s_font-2)
x0 = np.linspace(0, int(max(y)), 100)
slope, intercept = np.polyfit(y, z, 1)
fit_line = slope * x0 + intercept
ax.plot(x0, fit_line, color='grey',linestyle = '--', label='Fitted Line')
ax.set_ylabel('Carbon emission change rate (%)',fontsize=s_font-2)
ax.set_xlabel('Planned capacity (GW)',fontsize=s_font-2)
fig.savefig('Figs/SFig4-1c.pdf',format='pdf',dpi=600)
fig.savefig("Figs/SFig4-1c.png", dpi=600)
plt.show()

#%%
fig, ax = plt.subplots(figsize=(8, 6))
x = F_price[list_all_city_j,0]/8760
y = Total_Cost_total[list_all_city_j,1,-1]
z = Total_Cost_total[list_all_city_j,3,-1]
ax.scatter(x,y)
x0 = 0.1*np.linspace(3, int(10*max(x)+1), 100)
slope, intercept = np.polyfit(x, y, 1)
fit_line = slope * x0 + intercept
ax.plot(x0, fit_line, color='grey',linestyle = '--', label='Fitted Line')
if intercept < 0:
    equation = f'y = {slope:.2f}x - {-intercept:.2f}'
else:
    equation = f'y = {slope:.2f}x + {intercept:.2f}'

for kk in [71,76,34,87]:
    ax.scatter(F_price[kk,0]/8760, Total_Cost_total[kk,1,-1],color = 'red')
    ax.text(F_price[kk,0]/8760-0.01, Total_Cost_total[kk,1,-1]+0.01, cities[kk])
ax.set_ylabel('LCOE of FPV (CNY/kWh)',fontsize=s_font-2)
ax.set_xlabel('Electricity price (CNY/kWh)',fontsize=s_font-2)
fig.savefig('Figs/SFig4-2a.pdf',format='pdf',dpi=600)
fig.savefig("Figs/SFig4-2a.png", dpi=600)
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(z,y)
x0 = 0.1*np.linspace(2, int(10*max(z))+1, 100)
slope, intercept = np.polyfit(z, y, 1)
fit_line = slope * x0 + intercept
ax.plot(x0, fit_line, color='grey',linestyle = '--', label='Fitted Line')
if intercept < 0:
    equation = f'y = {slope:.2f}x - {-intercept:.2f}'
else:
    equation = f'y = {slope:.2f}x + {intercept:.2f}'
for kk in [71,76,34,87]:
    ax.scatter(Total_Cost_total[kk,3,-1], Total_Cost_total[kk,1,-1],color = 'red')
    if cities[kk] == 'Urumqi':
        ax.text(Total_Cost_total[kk,3,-1]+0.0, Total_Cost_total[kk,1,-1]+0.01, cities[kk])
    else:
        ax.text(Total_Cost_total[kk,3,-1]-0.01, Total_Cost_total[kk,1,-1]+0.01, cities[kk])
ax.set_ylabel('LCOE of FPV (CNY/kWh)',fontsize=s_font-2)
ax.set_xlabel('LCOE of RPV (CNY/kWh)',fontsize=s_font-2)
fig.savefig('Figs/SFig4-2b.pdf',format='pdf',dpi=600)
fig.savefig("Figs/SFig4-2b.png", dpi=600)
plt.show()

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Functions
from scipy.io import savemat,loadmat

###2025为起点，2030-2050
#30,35,40,45,50
path = '#ML_results/'
city_path = 'ALL_102_cities/'
path_type = '#ML_results/Power'
path_cap = '#ML_results/Capacity'

City_statistic = pd.read_excel(path+'City_statistic.xlsx',sheet_name = 'Class_Volume', index_col=0)
Statis_all = pd.read_excel(path+'City_statistic.xlsx',sheet_name = 'Key_information', index_col=0)

Clu_center,Clu_days = np.load('Fig_input_data/Clu_center.npy'),np.load('Fig_input_data/Clu_days.npy')

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
    if cc in [97,92,2,6,8,69]:
        data_path = '#Opt_results/'+city_name+'_hybrid_n.mat'
    else:
        data_path = '#Opt_results/'+city_name+'_hybrid.mat'
    # if not os.path.exists(data_path):
    #     continue
    Data = loadmat(data_path)
    R_Cap_r = Data['R_Cap_r']  #np.zeros((N_grid, 2, Y))
    #R_Cap_s = Data['R_Cap_s']  #np.zeros((N_grid, 2, Y))
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
    K1_pri_wall = [1e3*wall_0[i] for i in range(6)]  #屋顶价格，单位CNY/KW
    K1_pri_win  = [1e3*win_0[i] for i in range(6)]  #立面价格，单位CNY/KW
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
        for i in range(12):  #12个月
            K3_D_fa_id[case][:,i,:] = (1/1e-3)*np.load(P1)[N_gg,(Clu_center[cc,i])*24:(Clu_center[cc,i]+1)*24]
        Total_Cost_total[cc,4,case] = np.sum((K1_pri_wall[case+1]*(1-WWR[N_gg,0])+K1_pri_win[case+1]*WWR[N_gg,0])*R_Cap_f[N_gg,2,case])
        Total_Ele_f = sum(Clu_days[cc,d]*np.sum(K3_D_fa_id[case][:,d,:],axis = (0,1)) for d in range(12))
        Total_Cost_total[cc,5,case] = (Total_Cost_total[cc,4,case]+sum(0.04*Total_Cost_total[cc,4,case]/(1.08)**i for i in range(1,year_op+1)))/sum(Total_Ele_f/(1.08)**i for i in range(1,year_op+1))

    print(Total_Cost_total[cc,1,-1])
    print(Total_Cost_total[cc,3,-1])
    print(Total_Cost_total[cc,5,-1])
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
list_all_city_name = City_statistic.index[list_all_city_j].tolist()
cities = City_statistic.index.tolist()

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
        data_path = '#Opt_results/'+city_name+'_hybrid_n.mat'
    else:
        data_path = '#Opt_results/'+city_name+'_hybrid.mat'
    Data = loadmat(data_path)
    F_table1.iloc[cc,0] = city_name
    F_table2.iloc[cc,0] = city_name
    F_table3.iloc[cc,0] = city_name
    F_table23.iloc[cc,0] = city_name
    
    R_Cap_r = Data['R_Cap_r']  #np.zeros((N_grid, 2, Y))
    R_Cap_s = Data['R_Cap_s']  #np.zeros((N_grid, 2, Y))
    R_Cap_f = Data['R_Cap_f']  #np.zeros((N_grid, 3, Y))
    R_Ele = Data['R_Ele']      #np.zeros((N_grid, Y))
    #R_Pow = Data['R_Pow']      #np.zeros((N_grid, Y, D, T))
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
s_font,s_title = 16,14

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
    #ax.barh((max_len - ind - 1) * city_gap, wall_limit_split, bar_width, label='Planned capacity', color='green')
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
        mpatches.Patch(fc = 'lightgreen', label='Ideal capacity'),
        plt.Line2D([0], [0], marker='o', color='blue',
                   markersize=37, label='Planned LCOE', linestyle='none'),
        plt.Line2D([0], [0], marker='o', color='lightblue',
                   markersize=37, label='Ideal LCOE', linestyle='none')
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
fig.savefig('Figs_new_supp/sFig3_ideal_comp.pdf', format='pdf', dpi=600, bbox_inches='tight')
fig.savefig("Figs_new_supp/sFig3_ideal_comp.png", dpi=600, bbox_inches='tight')
plt.show()

#%%  Capa_relation
import numpy as np
import matplotlib.pyplot as plt

x = F_fig1[:,0]/25
y = Total_Cap_total[list_all_city_j,0,-1]/1e6
z = F_table1.iloc[:,3]
City_order = ['red' for _ in range(1)] + ['orange' for _ in range(1)] + ['blue' for _ in range(1)] + ['green' for _ in range(6)]

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x[:7],y[:7],color = City_order[0],label = 'SLC')
ax.scatter(x[7:22],y[7:22],color = City_order[1],label = 'VLC')
ax.scatter(x[22:35],y[22:35],color = City_order[2],label = 'LC-I')
ax.scatter(x[35:],y[35:],color = City_order[3],label = 'LC-II',alpha = 1.0)
ax.legend(loc = 'upper left', fontsize = s_font+2)
x0 = np.linspace(0, int(max(x)), 100)
slope, intercept = np.polyfit(x, y, 1)
fit_line = slope * x0 + intercept
ax.plot(x0, fit_line, color='grey',linestyle = '--', label='Fitted Line')
ax.set_ylabel('Planned capacity (GW)',fontsize=s_font+2)
ax.set_xlabel('Annual electricity demands (TWh)',fontsize=s_font+2)
ax.xaxis.set_tick_params(labelsize=s_font)
ax.yaxis.set_tick_params(labelsize=s_font)
fig.savefig('Figs_new_supp/sFig3_capa_rela_ele.pdf', format='pdf', dpi=600, bbox_inches='tight')
fig.savefig("Figs_new_supp/sFig3_capa_rela_ele.png", dpi=600, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
Total_area_re = Total_area[list_all_city_j,0]
ax.scatter(Total_area_re[:7],y[:7],color = City_order[0],label = 'SLC')
ax.scatter(Total_area_re[7:22],y[7:22],color = City_order[1],label = 'VLC')
ax.scatter(Total_area_re[22:35],y[22:35],color = City_order[2],label = 'LC-I')
ax.scatter(Total_area_re[35:],y[35:],color = City_order[3],label = 'LC-II')
ax.legend(loc = 'upper left', fontsize = s_font+2)
x0 = np.linspace(0, int(max(Total_area_re)), 100)
slope, intercept = np.polyfit(Total_area_re, y, 1)
fit_line = slope * x0 + intercept
ax.plot(x0, fit_line, color='grey',linestyle = '--', label='Fitted Line')
#ax.scatter(Total_area[35], y[35],color = 'red')
ax.set_ylabel('Planned capacity (GW)',fontsize=s_font+2)
ax.set_xlabel('Facade area (km2)',fontsize=s_font+2)
ax.xaxis.set_tick_params(labelsize=s_font)
ax.yaxis.set_tick_params(labelsize=s_font)
fig.savefig('Figs_new_supp/sFig3_capa_rela_area.pdf', format='pdf', dpi=600, bbox_inches='tight')
fig.savefig("Figs_new_supp/sFig3_capa_rela_area.png", dpi=600, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y[:7],z[:7],color = City_order[0],label = 'SLC')
ax.scatter(y[7:22],z[7:22],color = City_order[1],label = 'VLC')
ax.scatter(y[22:35],z[22:35],color = City_order[2],label = 'LC-I')
ax.scatter(y[35:],z[35:],color = City_order[3],label = 'LC-II')
ax.legend(loc = 'upper right', fontsize = s_font+2)
x0 = np.linspace(0, int(max(y)), 100)
slope, intercept = np.polyfit(y, z, 1)
fit_line = slope * x0 + intercept
ax.plot(x0, fit_line, color='grey',linestyle = '--', label='Fitted Line')
ax.set_ylabel('Carbon reduction rate (%)',fontsize=s_font+2)
ax.set_xlabel('Planned capacity (GW)',fontsize=s_font+2)
ax.xaxis.set_tick_params(labelsize=s_font)
ax.yaxis.set_tick_params(labelsize=s_font)
fig.savefig('Figs_new_supp/sFig3_capa_rela_car.pdf', format='pdf', dpi=600, bbox_inches='tight')
fig.savefig("Figs_new_supp/sFig3_capa_rela_car.png", dpi=600, bbox_inches='tight')
plt.show()

#%% LCOE_relation
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
    ax.text(F_price[kk,0]/8760-0.01, Total_Cost_total[kk,1,-1]+0.01, cities[kk],fontsize=s_font)
ax.set_ylabel('LCOE of FPV (CNY/kWh)',fontsize=s_font+2)
ax.set_xlabel('Electricity price (CNY/kWh)',fontsize=s_font+2)
ax.xaxis.set_tick_params(labelsize=s_font)
ax.yaxis.set_tick_params(labelsize=s_font)
fig.savefig('Figs_new_supp/sFig3_LCOE_rela_price.pdf', format='pdf', dpi=600, bbox_inches='tight')
fig.savefig("Figs_new_supp/sFig3_LCOE_rela_price.png", dpi=600, bbox_inches='tight')
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
        ax.text(Total_Cost_total[kk,3,-1]+0.005, Total_Cost_total[kk,1,-1]-0.007, cities[kk],fontsize=s_font)
    else:
        ax.text(Total_Cost_total[kk,3,-1]-0.01, Total_Cost_total[kk,1,-1]+0.01, cities[kk],fontsize=s_font)
ax.set_ylabel('LCOE of FPV (CNY/kWh)',fontsize=s_font+2)
ax.set_xlabel('LCOE of RPV (CNY/kWh)',fontsize=s_font+2)
ax.xaxis.set_tick_params(labelsize=s_font)
ax.yaxis.set_tick_params(labelsize=s_font)
fig.savefig('Figs_new_supp/sFig3_LCOE_rela_RPV.pdf', format='pdf', dpi=600, bbox_inches='tight')
fig.savefig("Figs_new_supp/sFig3_LCOE_rela_RPV.png", dpi=600, bbox_inches='tight')
plt.show()




#%%   #############ZONE
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, shape, MultiPolygon
from tqdm import tqdm
import numpy as np
from matplotlib.patches import Patch
from geopy.distance import geodesic
from scipy.io import savemat,loadmat

path = '#ML_results/'
city_path = 'ALL_102_cities/'
path_type = '#ML_results/Power'
path_cap = '#ML_results/Capacity'

City_statistic = pd.read_excel(path+'City_statistic.xlsx',sheet_name = 'Class_Volume', index_col=0)
Statis_all = pd.read_excel(path+'City_statistic.xlsx',sheet_name = 'Key_information', index_col=0)

#3: Beijing; 74: Wuhan; 63: Suzhou; 92：Changchun; 56: Shanghai； 15: Guangzhou；59：shenzhen；60：shenyang
for cc in [3,74,59]:
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

    Grid = pd.DataFrame(np.zeros((len(N_gg),5)),columns = ['Long.','Lat.','Type','Capacity-2030(KW)','Capacity-2050(KW)'])
    Forms = ['HnD','HnS','MnD','MnS']
    for i in range(4):
        Grid.iloc[list_form[i],:2] = G_type[N_gg,:][list_form[i],6:8]
        Grid.iloc[list_form[i],2] = Forms[i]
        Grid.iloc[list_form[i],3] = R_Cap_f[N_gg,:,:][list_form[i],0,0]  #/G_type[N_gg,:][list_form[i],4]
        Grid.iloc[list_form[i],4] = R_Cap_f[N_gg,:,:][list_form[i],0,-1] #/G_type[N_gg,:][list_form[i],4]

    s_font,s_title = 16+5,14+5

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
            Uni_Cap[:,1] = R_Cap_f[list_form[i],0,-1]
            Uni_Cap[:,2] = np.sum(R_Pow_f[list_form[i],:,:,:], axis = (1,2,3))
            Uni_Cap[:,3] = np.sum(R_Car[list_form[i],0,:], axis = 1)-np.sum(R_Car[list_form[i],1,:],axis = 1)
            
            x,y = Uni_Cap[:,0]/1e6, Uni_Cap[:,1]/1e3
            Cap_form.append(np.sum(R_Cap_f[list_form[i],0,-1]))
            Cap_carbon.append(np.sum(Uni_Cap[:,3]))
            
            ax1 = fig.add_subplot(gs[i1, i2])
            #ax1.scatter(np.arange(len(x)), y/x, c = colors[i],alpha = 0.5)
            ax1.scatter(x,y,c = colors[i],alpha = 0.5)
            ax1.set_title(types[i],fontsize = s_font, y=0.85)
            slope1, slope2, intercept = np.polyfit(x, y, 2)
            x0 = np.linspace(0, 6, 100)
            fit_line = slope1 * x0**2 + slope2 * x0 + intercept
            ax1.plot(x0, fit_line, color = 'grey', linestyle = '--', label='Fitted Line')
            if i1 == 1:
                ax1.set_xlabel('Facade area (km2)',fontsize = s_font)
            if i2 == 0:
                x1,y1 = 5,160
                ax1.set_ylabel('Capacity (MW)',fontsize = s_font)
            if i2 == 1:
                x1,y1 = 0.5,30
            ax1.set_xlim(0,x1)
            ax1.set_ylim(0,y1)
            ax1.xaxis.set_tick_params(labelsize=s_font-2)  # x轴刻度字体
            ax1.yaxis.set_tick_params(labelsize=s_font-2)  # y轴刻度字体

    fig.suptitle(city_name, 
                fontsize=s_font, 
                weight='bold', 
                y=0.95)

    fig.savefig('Figs_new_supp/sFig5_zone_area_capa_'+city_name+'.pdf', format='pdf', dpi=600, bbox_inches='tight')
    fig.savefig('Figs_new_supp/sFig5_zone_area_capa_'+city_name+'.png', dpi=600, bbox_inches='tight')
    plt.show()

    print(Cap_form[0]/sum(Cap_form))
    print(Cap_carbon[0]/sum(Cap_carbon))
    
    ##### total capa
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
        Feas_read_info = np.load(city_path+city_name+'_ALL_Featuers_supplementary.npy')
    else:
        Feas_read_info = np.load(city_path+city_name+'_ALL_Featuers.npy')[:,range(17,29)]

    R_area = Feas_read_info[indices_non_zero,3:4]

    Clu_center,Clu_days = np.load('Fig_input_data/Clu_center.npy'),np.load('Fig_input_data/Clu_days.npy')
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
        #ax.set_xticks(np.arange(0.5, 6.5, 1))
        #ax.set_xticklabels(['2025', '2030', '2035', '2040', '2045', '2050'])
        ax.legend(custom_lines, Forms,loc='upper left', fontsize = 10, ncol=1)
        #ax.set_ylim(0,160)
        #ax.set_title(['Others','Residential'][jj],fontsize = s_title)
        ax.xaxis.set_tick_params(labelsize=s_font-2)
        ax.yaxis.set_tick_params(labelsize=s_font-2)
        ax.set_title(city_name,fontsize = s_font,weight='bold',y=0.9)
        ax.set_xlabel('Stages',fontsize = s_font)
        ax.set_ylabel(['Planned Capacity (GW)','Unit Capacity (kW/m2)'][jj],fontsize = s_font)
        fig.savefig('Figs_new_supp/sFig5_zone_total_capa_'+city_name+'.pdf', format='pdf', dpi=600, bbox_inches='tight')
        fig.savefig('Figs_new_supp/sFig5_zone_total_capa_'+city_name+'.png', dpi=600, bbox_inches='tight')
        plt.show()

    print(Cap_indu[1,0]/np.sum(Cap_indu[1,:]))
    print(np.sum(Cap_indu[1,1:3])/np.sum(Cap_indu[1,:]))

    print(Ele_FRL/Ele_FRL[2:3,:])

    ###### total_carbon_RS
    Car_cum = np.cumsum(Cap_car_ori, axis=1)
    if cc == 3:
        y_lim = 820
    elif cc == 74:
        y_lim = 540
    elif cc == 59:
        y_lim = 400
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
    ax.set_xlabel('Stages',fontsize = s_font)
    ax.set_ylabel('Carbon emission (Million ton)',fontsize = s_font)
    ax.set_title(city_name+' (RS)',fontsize = s_font,weight='bold',y=0.9)
    ax.xaxis.set_tick_params(labelsize=s_font-2)
    ax.yaxis.set_tick_params(labelsize=s_font-2)
    fig.savefig('Figs_new_supp/sFig5_zone_total_capa_RS_'+city_name+'.pdf', format='pdf', dpi=600, bbox_inches='tight')
    fig.savefig('Figs_new_supp/sFig5_zone_total_capa_RS_'+city_name+'.png', dpi=600, bbox_inches='tight')
    plt.show()

    print((np.cumsum(Cap_car_ori, axis=1)[1,-1]-np.cumsum(Cap_car, axis=1)[1,-1])/np.cumsum(Cap_car_ori, axis=1)[1,-1])


    ### total_carbon_RS+F
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
    ax.set_xlabel('Stages',fontsize = s_font)
    ax.set_ylabel('Carbon emission (Million ton)',fontsize = s_font)
    ax.set_title(city_name+' (RS+F)',fontsize = s_font,weight='bold',y=0.9)
    ax.xaxis.set_tick_params(labelsize=s_font-2)
    ax.yaxis.set_tick_params(labelsize=s_font-2)
    fig.savefig('Figs_new_supp/sFig5_zone_total_capa_RSF_'+city_name+'.pdf', format='pdf', dpi=600, bbox_inches='tight')
    fig.savefig('Figs_new_supp/sFig5_zone_total_capa_RSF_'+city_name+'.png', dpi=600, bbox_inches='tight')
    plt.show()

    ###### Unit_Capa
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
    plt.title(city_name,fontsize = s_font,weight='bold',y=0.9)
    plt.xticks(xticks, xtick_labels, fontsize=s_font-2)
    plt.yticks(fontsize=s_font-2)
    plt.ylabel('Unit capacity (W/m2)',fontsize = s_font)
    plt.xlabel('Stages',fontsize = s_font)
    plt.ylim(-15,)
    custom_lines = [plt.Line2D([0], [0], color=colors[i], lw=4) for i in range(4)]
    plt.legend(custom_lines, Forms,loc='lower left', fontsize = s_font-4, ncol=4)
    fig.savefig('Figs_new_supp/sFig5_zone_unit_capa_'+city_name+'.pdf', format='pdf', dpi=600, bbox_inches='tight')
    fig.savefig('Figs_new_supp/sFig5_zone_unit_capa_'+city_name+'.png', dpi=600, bbox_inches='tight')
    plt.show()

    #### Unit_hour
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
    plt.title(city_name,fontsize = s_font,weight='bold',y=0.9)
    plt.xticks(xticks, xtick_labels, fontsize=s_font-2)
    plt.yticks(fontsize=s_font-2)
    plt.ylabel('Annual utilization hours',fontsize = s_font)
    plt.xlabel('Stages',fontsize = s_font)
    plt.ylim(500,)
    if cc == 3 or cc == 74:
        plt.ylim(500,2000)
    custom_lines = [plt.Line2D([0], [0], color=colors[i], lw=4) for i in range(4)]
    plt.legend(custom_lines, Forms,loc='lower left', fontsize = s_font-4, ncol=4)
    fig.savefig('Figs_new_supp/sFig5_zone_unit_hour_'+city_name+'.pdf', format='pdf', dpi=600, bbox_inches='tight')
    fig.savefig('Figs_new_supp/sFig5_zone_unit_hour_'+city_name+'.png', dpi=600, bbox_inches='tight')

    plt.show()


    ##### unit_carbon_RS
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
    plt.title(city_name+' (RS)',fontsize = s_font,weight='bold',y=0.9)
    plt.xticks(xticks, xtick_labels, fontsize=s_font-2)
    plt.yticks(fontsize=s_font-2)
    plt.ylabel('Unit carbon emission (kg/kWh)',fontsize = s_font)
    plt.xlabel('Stages',fontsize = s_font)
    plt.ylim(-0.05,yy_lim)
    custom_lines = [plt.Line2D([0], [0], color=colors[i], lw=4) for i in range(4)]
    plt.legend(custom_lines, Forms,loc='lower left', fontsize = s_font-4, ncol=4)
    fig.savefig('Figs_new_supp/sFig5_zone_unit_carbon_RS_'+city_name+'.pdf', format='pdf', dpi=600, bbox_inches='tight')
    fig.savefig('Figs_new_supp/sFig5_zone_unit_carbon_RS_'+city_name+'.png', dpi=600, bbox_inches='tight')
    plt.show()


    #%## unit_carbon_RS+F
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
    plt.title(city_name+' (RS+F)',fontsize = s_font,weight='bold',y=0.9)
    plt.xticks(xticks, xtick_labels, fontsize=s_font-2)
    plt.yticks(fontsize=s_font-2)
    plt.ylabel('Unit carbon emission (kg/kWh)',fontsize = s_font)
    plt.xlabel('Stages',fontsize = s_font)
    plt.ylim(-0.05,yy_lim)
    custom_lines = [plt.Line2D([0], [0], color=colors[i], lw=4) for i in range(4)]
    plt.legend(custom_lines, Forms,loc='lower left', fontsize = s_font-4, ncol=4)
    fig.savefig('Figs_new_supp/sFig5_zone_unit_carbon_RSF_'+city_name+'.pdf', format='pdf', dpi=600, bbox_inches='tight')
    fig.savefig('Figs_new_supp/sFig5_zone_unit_carbon_RSF_'+city_name+'.png', dpi=600, bbox_inches='tight')
    plt.show()

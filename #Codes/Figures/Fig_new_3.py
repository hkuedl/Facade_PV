#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat,loadmat
import os

path = '#ML_results/'
city_path = 'ALL_102_cities/'
path_type = '#ML_results/Power'
path_cap = '#ML_results/Capacity'
City_statistic = pd.read_excel(path+'City_statistic.xlsx',sheet_name = 'Class_Volume', index_col=0)
Statis_all = pd.read_excel(path+'City_statistic.xlsx',sheet_name = 'Key_information', index_col=0)

# Clu_center,Clu_days = np.load('Fig_input_data/Clu_center.npy'),np.load('Fig_input_data/Clu_days.npy')
# Total_Cap_total = np.zeros((102,2,5))
# Total_Ele = np.zeros((102,5))
# Total_Cost_total = np.zeros((102,6,5))
# Total_price = np.zeros((102,2,5))
# Total_price_true = np.zeros((102,2,5))
# Total_Carbon = np.zeros((102,2,5))
# Total_Carbon_00 = np.zeros((102,2,5))
# Total_price_00 = np.zeros((102,2,5))
# Total_price_00_true = np.zeros((102,2,5))
# Total_area = np.zeros((102,1))
# for cc in range(102):  #[3,15,56,59]:
#     city_name = City_statistic.index[cc]
#     print(city_name)
#     C1 = path_cap+'/Cap_facade_'+city_name+'.npy'
#     C2 = path_cap+'/Cap_roof_'+city_name+'.npy'
#     G_type = np.load('#ML_results/Grid_type/'+'Grid_type_'+city_name+'.npy')
#     Feas_read_sta = np.load(city_path+city_name+'_ALL_Featuers.npy')[:,[i for i in range(14)]+[15,16]]
#     indices_non_zero = np.where(Feas_read_sta[:,11] != 0)[0]
#     WWR = np.load(city_path+city_name+'_ALL_Featuers.npy')[indices_non_zero,13:14]
#     N_grid = G_type.shape[0]
#     N_gg = np.where(G_type[:,0] != 888)[0]
#     Total_area[cc,0] = np.sum(G_type[:,4])/1e6 #(km2)
#     if cc in [97,92,2,6,8,69]:
#         data_path = '#Opt_results/'+city_name+'_hybrid_n.mat'
#     else:
#         data_path = '#Opt_results/'+city_name+'_hybrid.mat'
#     Data = loadmat(data_path)
#     R_Cap_r = Data['R_Cap_r']  #np.zeros((N_grid, 2, Y))
#     R_Cap_f = Data['R_Cap_f']  #np.zeros((N_grid, 3, Y))
#     R_Ele = Data['R_Ele']      #np.zeros((N_grid, Y))
#     R_Pow_f = Data['R_Pow_f']  #np.zeros((N_grid,Y, D, T))
#     R_Pow_r = Data['R_Pow_r']  #np.zeros((N_grid, 2, Y, D, T))
#     R_Car = Data['R_Car']     #np.zeros((N_grid,2,Y))
#     R_Cost = Data['R_Cost']    #np.zeros((N_grid,2,Y))
#     R_Cost_true = Data['R_Cost_true']    #np.zeros((N_grid,2,Y))
#     Total_Cap_total[cc,:,:] = np.sum(R_Cap_f[:,[0,2],:],axis = 0)
#     Total_Carbon[cc,:,:] = 1e3*np.sum(R_Car[:,:,:],axis = 0)/np.sum(R_Ele[:,:],axis = 0)
#     Total_price[cc,:,:] = np.sum(R_Cost[:,:,:],axis = 0)/np.sum(R_Ele[:,:],axis = 0)
#     Total_price_true[cc,:,:] = np.sum(R_Cost_true[:,:,:],axis = 0)/np.sum(R_Ele[:,:],axis = 0)
#     Total_Carbon_00[cc,:,:] = np.sum(R_Car[:,:,:],axis = 0)
#     Total_price_00[cc,:,:] = np.sum(R_Cost[:,:,:],axis = 0)
#     Total_price_00_true[cc,:,:] = np.sum(R_Cost_true[:,:,:],axis = 0)
#     wall_0 = [3.18]+[2.98,2.75,2.70,2.65,2.60]
#     win_0 = [4.80]+[4.50,4.15,4.08,4.00,3.92]
#     K1_pri_wall = [1e3*wall_0[i] for i in range(6)]
#     K1_pri_win  = [1e3*win_0[i] for i in range(6)]
#     year_op = 25
#     K3_D_fa_id = [np.zeros((len(N_gg),12,24)) for _ in range(5)]
#     for case in range(5):
#         Total_Ele[cc,case] = sum(Clu_days[cc,d]*np.sum(R_Pow_f[:,case,d,:],axis = (0,1)) for d in range(12))
#         Total_Cost_total[cc,0,case] = np.sum((K1_pri_wall[case+1]*(1-WWR[:,0])+K1_pri_win[1+case]*WWR[:,0])*R_Cap_f[:,0,case])
#         Total_Cost_total[cc,1,case] = (Total_Cost_total[cc,0,case]+sum(0.04*Total_Cost_total[cc,0,case]/(1.08)**i for i in range(1,year_op+1)))/sum(Total_Ele[cc,case]/(1.08)**i for i in range(1,year_op+1))
#         Total_Cost_total[cc,2,case] = np.sum(K1_pri_wall[case+1]*R_Cap_r[:,1,case])
#         Total_Ele_r = sum(Clu_days[cc,d]*np.sum(R_Pow_r[:,1,case,d,:],axis = (0,1)) for d in range(12))
#         Total_Cost_total[cc,3,case] = (Total_Cost_total[cc,2,case]+sum(0.02*Total_Cost_total[cc,2,case]/(1.08)**i for i in range(1,year_op+1)))/sum(Total_Ele_r/(1.08)**i for i in range(1,year_op+1))
#         P1 = path_type+str(case+2)+'/N_P_facade_ideal_'+str(case+2)+'_'+city_name+'.npy'
#         for i in range(12):
#             K3_D_fa_id[case][:,i,:] = (1/1e-3)*np.load(P1)[N_gg,(Clu_center[cc,i])*24:(Clu_center[cc,i]+1)*24]
#         Total_Cost_total[cc,4,case] = np.sum((K1_pri_wall[case+1]*(1-WWR[N_gg,0])+K1_pri_win[case+1]*WWR[N_gg,0])*R_Cap_f[N_gg,2,case])
#         Total_Ele_f = sum(Clu_days[cc,d]*np.sum(K3_D_fa_id[case][:,d,:],axis = (0,1)) for d in range(12))
#         Total_Cost_total[cc,5,case] = (Total_Cost_total[cc,4,case]+sum(0.04*Total_Cost_total[cc,4,case]/(1.08)**i for i in range(1,year_op+1)))/sum(Total_Ele_f/(1.08)**i for i in range(1,year_op+1))
# list_all_city_j = []
# list_all_city_order = np.zeros((102,3))
# list_all_city_order[:,0] = np.arange(102)
# for i in range(4):
#     indices = Statis_all.index[Statis_all.iloc[:,-1] == ['超大城市','特大城市','I型大城市','II型大城市'][i]]
#     for index in indices:
#         i_loc = Statis_all.index.get_loc(index)
#         list_all_city_order[i_loc,2] = 4-i
#         list_all_city_order[i_loc,1] = Total_Cap_total[i_loc,0,-1]/1e6
# sorted_indices = np.lexsort((-list_all_city_order[:, 1], -list_all_city_order[:, 2]))
# data_sorted = list_all_city_order[sorted_indices]
# list_all_city_j = list(data_sorted[:,0].astype(int))
# wall_actual = Total_Cap_total[list_all_city_j,0,:]/1e6
# LCOE_actual = Total_Cost_total[list_all_city_j,1,:]
# np.save('Fig_input_data/Fig3_list_city.npy',data_sorted[:,0].astype(int))
# np.save('Fig_input_data/Fig3_wall_actual.npy',wall_actual)
# np.save('Fig_input_data/Fig3_LCOE_actual.npy',LCOE_actual)



list_all_city_j = list(np.load('Fig_input_data/Fig3_list_city.npy'))
wall_actual = np.load('Fig_input_data/Fig3_wall_actual.npy')
LCOE_actual = np.load('Fig_input_data/Fig3_LCOE_actual.npy')

#%%
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

import matplotlib.colors as mcolors
cmap = mcolors.LinearSegmentedColormap.from_list("green_gradient", ['#E0F2E9', '#007E2E'])
colors = [cmap(i / 4) for i in range(5)]
cmap111 = mcolors.LinearSegmentedColormap.from_list("blue_gradient", ['#E6F0FF', '#003366'])
colors111 = [cmap111(i / 4) for i in range(5)]

lw_axis = 0.8
s_font,s_title = 16,14

cities_re = [cities[i] for i in list_all_city_j]
cities_re_split1 = [cities_re[0:51],cities_re[51:]]
wall_actual_split1 = [wall_actual[0:51,:],wall_actual[51:,:]]
LCOE_actual_split1 = [LCOE_actual[0:51,:],LCOE_actual[51:,:]]

ratio_plot = 3

figwidth =  8.5 * ratio_plot
fs = 8 * ratio_plot
lw = 0.4 * ratio_plot
lw2 = 0.75 * ratio_plot
lw3 = 1 * ratio_plot
# ----------------- Create Canvas and Subplots -----------------
font_options = {'size': fs-12}
plt.rc('font', **font_options)

fig, axes = plt.subplots(
    1, 2,
    figsize=np.array([figwidth*1.2, figwidth * 1.3]) / 2.54,
)

for i_ax in range(2):
    ax = axes[i_ax]
    wall_actual_split = wall_actual_split1[i_ax]
    LCOE_actual_split = LCOE_actual_split1[i_ax]
    cities_re_split = cities_re_split1[i_ax]

    ind = np.arange(51)
    max_len = 51
    bar_width = 1.5
    city_gap = 2
    bottom = np.zeros(51)
    for tt in range(5):
        ax.barh((max_len - ind - 1) * city_gap, wall_actual_split[:,tt]-bottom, bar_width, left = bottom,  label='Planned capacity', color=colors[tt])
        bottom = wall_actual_split[:,tt]

    ax.set_xlim(0,38)
    ax.set_yticks((max_len - ind - 1) * city_gap)
    ax.set_yticklabels(cities_re_split, fontsize=fs-9)
    ax.tick_params(axis='x', labelsize=fs-6)
    ax.tick_params(axis='y', labelsize=fs-8)
    ax.xaxis.tick_top()
    ax.set_xlabel('GW', fontsize=fs-5, labelpad=5)

    # temp_part = df_plot[['city_level']].copy()
    # temp_part.reset_index(inplace=True)       # Turn city names into a column
    # temp_part['row_idx'] = temp_part.index    # Row index for y-axis in barh
    if i_ax == 0:
        i_rows = 4
        min_row = [44*2+0,29*2+0,16*2+0,0] #[0,7*2,22*2,35*2]
        max_row = [50*2,44*2-2,29*2-2,16*2-2] #[7*2,22*2,35*2,51*2]
        City_name = ['SLC','VLC','LC-I','LC-II']
    else:
        i_rows = 1
        min_row = [0] #[0,7*2,22*2,35*2]
        max_row = [50*2] #[7*2,22*2,35*2,51*2]
        City_name = ['LC-II']
    for i_row in range(i_rows):
        min_idx = min_row[i_row]
        max_idx = max_row[i_row]
        mid_idx = 0.5 * (min_idx + max_idx)  # Vertical middle

        x_pos = ax.get_xlim()[-1] * 0.83
        arrow_len = 0.5
        h_len = ax.get_xlim()[-1] * 0.04

        # (1) Top arrow
        ax.annotate(
            '',
            xy=(x_pos, min_idx),
            xytext=(x_pos, min_idx - arrow_len),
            arrowprops=dict(arrowstyle='<-', lw=lw_axis, color='black', mutation_scale=20),  # Larger value makes the arrow bigger
            annotation_clip=False,
            transform=ax.transData,
        )
        ax.plot(
            [x_pos - h_len, x_pos + h_len],  # Left and right endpoints
            [min_idx-0.3, min_idx-0.3],              # Same y-value
            color='black',
            lw=lw_axis,
            transform=ax.transData,
            clip_on=False
        )

        # (2) Vertical line
        ax.plot(
            [x_pos, x_pos], 
            [min_idx, max_idx],
            color='black', lw=lw_axis,
            transform=ax.transData,
            clip_on=False
        )

        # (3) Bottom arrow
        ax.annotate(
            '',
            xy=(x_pos, max_idx),
            xytext=(x_pos, max_idx + arrow_len),
            arrowprops=dict(arrowstyle='<-', lw=lw_axis, color='black', mutation_scale=20),
            annotation_clip=False,
            transform=ax.transData
        )
        ax.plot(
            [x_pos - h_len, x_pos + h_len],
            [max_idx+0.3, max_idx+0.3],
            color='black',
            lw=lw_axis,
            transform=ax.transData,
            clip_on=False
        )
        # (4) Add text at the midpoint of the vertical line
        ax.text(
            x_pos, mid_idx,
            City_name[i_row],
            ha='center', va='center',
            fontsize=fs-9,
            transform=ax.transData,
            bbox=dict(
                facecolor='white',         # Background color
                edgecolor='none'           # Remove border
            )
        )
    ax2 = ax.twiny()
    for tt in range(5):
        ax2.scatter(LCOE_actual_split[:,tt], (max_len - ind - 1) * city_gap, color = colors111[tt], label='LCOE of planned FPV in 2050', zorder=5,s=100)
    ax2.set_xlim(0, 0.8)
    ax2.tick_params(axis='x', labelsize=fs-6)
    ax2.set_xlabel('CNY/kWh', fontsize=fs-5,labelpad=5)
    ax2.set_ylim(ax.get_ylim())

cax = fig.add_axes([0.35, 0.005, 0.4, 0.02])
norm = mcolors.BoundaryNorm(np.linspace(0, 1, 6), cmap.N)
cb = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm),
                 cax=cax,
                 orientation='horizontal',
                 ticks=np.linspace(0.1, 0.9, 5))
cb.set_ticklabels(['2030', '2035', '2040', '2045', '2050'])
cb.ax.tick_params(labelsize=fs-5, length=0, pad = 10)
cb.outline.set_visible(False)

x_centers = np.linspace(0.39, 0.71, 5)
y_circle = 0.037
for i, (x, color) in enumerate(zip(x_centers, colors111)):
    circle = plt.Circle((x, y_circle), 0.01,
                       color=color,
                       transform=fig.transFigure,  
                       zorder=10)
    fig.add_artist(circle)

lcoe_text = ax.text(0.28, 0.037, "LCOE",
                   ha='center', va='center',
                   fontsize=fs-5, fontweight='bold',
                   transform=fig.transFigure)

lcoe_text = ax.text(0.28, 0.012, "Capacity",
                   ha='center', va='center',
                   fontsize=fs-5, fontweight='bold',
                   transform=fig.transFigure)

for ax in axes:
    ax.set_ylim(-1, max_len * city_gap)
    ax.set_yticks(np.arange(max_len) * city_gap)

plt.subplots_adjust(wspace=0.5)

fig.savefig('Figs_new/Fig3.pdf',format='pdf',dpi=600,bbox_inches='tight')
fig.savefig("Figs_new/Fig3.png", dpi=600,bbox_inches='tight')
plt.show()
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

    C1 = path_cap+'/Cap_facade_'+city_name+'.npy'
    C2 = path_cap+'/Cap_roof_'+city_name+'.npy'

    if cc in [97,92,2,6,8,69]:
        data_path = '#Opt_results/'+city_name+'_hybrid_n.mat'
    else:
        data_path = '#Opt_results/'+city_name+'_hybrid.mat'

    G_type = np.load('#ML_results/Grid_type/'+'Grid_type_'+city_name+'.npy')
    Feas_read_sta = np.load(city_path+city_name+'_ALL_Featuers.npy')[:,[i for i in range(14)]+[15,16]]
    indices_non_zero = np.where(Feas_read_sta[:,11] != 0)[0]
    WWR = np.load(city_path+city_name+'_ALL_Featuers.npy')[indices_non_zero,13:14]
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

    # R_Cap_f_uni = R_Cap_f[N_gg,0,-1]#/G_type[N_gg,4]
    # non_zero = np.where(R_Cap_f_uni != 0)[0]
    # a1,a2 = np.quantile(R_Cap_f_uni[non_zero],0.33),np.quantile(R_Cap_f_uni[non_zero],0.67)
    # list_form = [list(np.where((R_Cap_f_uni<=0)&(R_Cap_f_uni>=0))[0]),\
    #        list(np.where((R_Cap_f_uni<a1)&(R_Cap_f_uni>0))[0]),\
    #         list(np.where((R_Cap_f_uni<=a2)&(R_Cap_f_uni>a1))[0]),\
    #             list(np.where((R_Cap_f_uni<=np.max(R_Cap_f_uni))&(R_Cap_f_uni>a2))[0])]
    # print([len(list_form[i]) for i in range(4)])
    # # np.save('list_form_'+city_name+'.npy',list_form[0])

    th_hh = 18
    th_std = [0,0]  #[np.percentile(G_type[N_gg,2], 50),np.percentile(G_type[N_gg,2], 50)]     #4  # 3.6#4[3.6,3.6]  # 
    th_aa = [200,200]  # [np.percentile(G_type[N_gg,3], 50),np.percentile(G_type[N_gg,3], 50)]   #200 #120  # 55#120[120,120]  #  
    list_form = [list(np.where((G_type[N_gg,1]>=th_hh)&(G_type[N_gg,2]>=th_std[1])&(G_type[N_gg,3]>=th_aa[1]))[0]),\
        list(np.where((G_type[N_gg,1]>=th_hh)&(G_type[N_gg,2]>=th_std[1])&(G_type[N_gg,3]<th_aa[0]))[0]),\
        list(np.where((G_type[N_gg,1]<th_hh)&(G_type[N_gg,2]>=th_std[0])&(G_type[N_gg,3]>=th_aa[1]))[0]),\
        list(np.where((G_type[N_gg,1]<th_hh)&(G_type[N_gg,2]>=th_std[0])&(G_type[N_gg,3]<th_aa[0]))[0])]

    print([len(list_form[i]) for i in range(4)])

    s_font = 30

    list_form = np.load('Fig_input_data/list_form_'+city_name+'.npy')

    hours = np.arange(0, 24*12)
    months = np.arange(0, 4)

    for i_index in [0,1]:
        PU_ele = np.max(np.sum(R_Pow[list_form,:,:,:],axis = 0))
        load_curve = (np.sum(R_Pow[list_form,:,:,:],axis = (0,1))/(5)/PU_ele)[[3,4,5,6,7,8,9,10,11,0,1,2]].ravel()
        RPV_output = (np.sum(R_Pow_r[list_form,i_index,:,:,:],axis = (0,1))/(5)/PU_ele)[[3,4,5,6,7,8,9,10,11,0,1,2]].ravel()
        FPV_output = (i_index*np.sum(R_Pow_f[list_form,:,:,:],axis = (0,1))/(5)/PU_ele)[[3,4,5,6,7,8,9,10,11,0,1,2]].ravel()
        St_ch = (np.sum(R_Pow_ch[list_form,i_index,:,:,:],axis = (0,1))/(5)/PU_ele)[[3,4,5,6,7,8,9,10,11,0,1,2]].ravel()
        St_dis = (np.sum(R_Pow_dis[list_form,i_index,:,:,:],axis = (0,1))/(5)/PU_ele)[[3,4,5,6,7,8,9,10,11,0,1,2]].ravel()
        grid_purchase = (np.sum(R_Pow_Buy[list_form,i_index,:,:,:],axis = (0,1))/(5)/PU_ele)[[3,4,5,6,7,8,9,10,11,0,1,2]].ravel()
        grid_discard = (np.sum(R_Pow_AB[list_form,i_index,:,:,:],axis = (0,1))/(5)/PU_ele)[[3,4,5,6,7,8,9,10,11,0,1,2]].ravel()
        grid_sell = (np.sum(R_Pow_G[list_form,i_index,:,:,:],axis = (0,1))/(5)/PU_ele)[[3,4,5,6,7,8,9,10,11,0,1,2]].ravel()

        fig, ax = plt.subplots(figsize=(18, 6))
        background = [0,24*3,24*6,24*9,24*12]
        backgroundcolor = ['whitesmoke','lightgrey','whitesmoke','lightgrey']
        for i in range(4):
            ax.axvspan(background[i], background[i+1], facecolor=backgroundcolor[i], alpha=0.5)  # alpha控制透明度

        colors = ['skyblue', 'mediumseagreen', 'moccasin', 'tomato', 'darksalmon', 'darkorange', 'orange']
        ax.plot(hours, load_curve, label='Load', color='black', linestyle = '--' ,linewidth=2)
        ax.stackplot(hours, RPV_output, FPV_output, grid_purchase, St_dis,St_ch,grid_discard,grid_sell,
                    labels=['RPV', 'FPV', 'Purchase', 'Discharge','Charge', 'Discard', 'Sell'], colors=colors)
        ax.set_ylabel('p.u.', fontsize=s_font+5)
        if i_index == 0:
            ax.set_title('Load balance under RS', fontsize=s_font, fontweight='bold',y=0.9)
            legend = ax.legend(loc='upper center',bbox_to_anchor=(0.5, 1.3), ncol=4,frameon=False,fontsize=s_font-5,labelspacing=0.8)
        else:
            ax.set_title('Load balance under RS+F', fontsize=s_font, fontweight='bold',y=0.9)
        ax.set_xticks([12*3,36*3,60*3,84*3])
        ax.set_xticklabels([])
        ax.tick_params(axis='y', labelsize=s_font-5)
        ax.set_ylim(0, 1)
        ax.set_xlim(-2,24*12+2)

        #ax.vlines(x=[24*3, 24*6, 24*9], ymin=0, ymax=1, colors='black', linestyles=':', linewidth=5)

        if i_index == 0:
            if cc == 3:
                fig.savefig('Figs_new/Fig6a'+city_name+'.pdf',format='pdf',dpi=600,bbox_inches='tight')
                fig.savefig('Figs_new/Fig6a'+city_name+'.png', dpi=600,bbox_inches='tight')
            else:
                fig.savefig('Figs_new_supp/sFig6a'+city_name+'.pdf',format='pdf',dpi=600,bbox_inches='tight')
                fig.savefig('Figs_new_supp/sFig6a'+city_name+'.png', dpi=600,bbox_inches='tight')
        elif i_index == 1:
            if cc == 3:
                fig.savefig('Figs_new/Fig6b'+city_name+'.pdf',format='pdf',dpi=600,bbox_inches='tight')
                fig.savefig('Figs_new/Fig6b'+city_name+'.png', dpi=600,bbox_inches='tight')
            else:
                fig.savefig('Figs_new_supp/sFig6b'+city_name+'.pdf',format='pdf',dpi=600,bbox_inches='tight')
                fig.savefig('Figs_new_supp/sFig6b'+city_name+'.png', dpi=600,bbox_inches='tight')
        plt.show()


    #%
    ###Note: here begins at winter
    PU_car = np.max(np.sum(R_Car_t[list_form,0,:,:,:],axis = (0)))
    values10 = (np.sum(R_Car_t[list_form,0,:,:,:],axis = (0,1))/5/PU_car)[[3,4,5,6,7,8,9,10,11,0,1,2]]
    values20 = (np.sum(R_Car_t[list_form,1,:,:,:],axis = (0,1))/5/PU_car)[[3,4,5,6,7,8,9,10,11,0,1,2]]
    values1 = values10.ravel()
    values2 = values20.ravel()

    x = np.arange(0, 24*12, 1)
    fig, ax = plt.subplots(figsize=(18, 6))
    background = [0,24*3,24*6,24*9,24*12]
    backgroundcolor = ['whitesmoke','lightgrey','whitesmoke','lightgrey']
    for i in range(4):
        ax.axvspan(background[i], background[i+1], facecolor=backgroundcolor[i], alpha=0.5) 
    ax.plot(x, values1, label='RS', color='blue', linestyle='-', linewidth=2.0, alpha=1)
    ax.plot(x, values2, label='RS+F', color='green', linestyle='-', linewidth=2.0, alpha=1)
    #ax.vlines(x=[24*3, 24*6, 24*9], ymin=0, ymax=0.6, colors='black', linestyles=':', linewidth=5)
    ax.set_title('Carbon emission', fontsize=s_font+2, fontweight='bold',y=0.9)
    #ax.set_xlabel('Time', fontsize=s_font)
    ax.set_ylabel('p.u.', fontsize=s_font+7)
    ax.set_xlim(-2,24*12+2)
    ax.set_ylim(0, 0.6)
    #ax.set_xticks(np.arange(0, 25, 6))
    ax.set_xticks([12*3,36*3,60*3,84*3])
    ax.set_xticklabels(['Spring', 'Summer', 'Autumn', 'Winter'],fontsize = s_font+2)
    ax.tick_params(axis='y', labelsize=s_font-3)
    #ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', fontsize=s_font)
    plt.tight_layout()

    if cc == 3:
        fig.savefig('Figs_new/Fig6c'+city_name+'.pdf',format='pdf',dpi=600,bbox_inches='tight')
        fig.savefig('Figs_new/Fig6c'+city_name+'.png', dpi=600,bbox_inches='tight')
    else:
        fig.savefig('Figs_new_supp/sFig6c'+city_name+'.pdf',format='pdf',dpi=600,bbox_inches='tight')
        fig.savefig('Figs_new_supp/sFig6c'+city_name+'.png', dpi=600,bbox_inches='tight')
    plt.show()

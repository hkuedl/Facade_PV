#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat,loadmat
import os
import matplotlib.font_manager as fm
from matplotlib import rcParams
font_path = 'arial.ttf'
custom_font = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)
rcParams['font.family'] = custom_font.get_name()

path = '#ML_results/'
city_path = 'ALL_102_cities/'
path_type = 'Power'
path_cap = 'Capacity'
City_statistic = pd.read_excel(path+'City_statistic.xlsx',sheet_name = 'Class_Volume', index_col=0)
Statis_all = pd.read_excel(path+'City_statistic.xlsx',sheet_name = 'Key_information', index_col=0)

list_all_city_j = list(np.load('Fig_input_data/Fig3_list_city.npy'))
wall_actual = np.load('Fig_input_data/Fig3_wall_actual.npy')
LCOE_actual = np.load('Fig_input_data/Fig3_LCOE_actual.npy')


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
    circle = plt.Circle((x, y_circle), 0.01,  # 圆心坐标和半径
                       color=color,
                       transform=fig.transFigure,  # 使用画布坐标系
                       zorder=10)  # 确保圆形在顶层
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
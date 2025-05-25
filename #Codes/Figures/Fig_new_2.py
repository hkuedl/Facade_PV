#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import Functions
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

# Merge area data and annual generation data
df_plot = area_df.iloc[:, :2].copy()  # km^2

roof_power = roof_data_1  # TWh
facade_power = ideal_facade_data_1  # TWh
df_plot['roof_power'] = roof_power
df_plot['facade_power'] = facade_power

df_plot['facade_carbon'] = facade_power
df_plot['roof_carbon'] = roof_power
for i in range(len(df_plot)):
    city_name = df_plot.index[i]
    _,K3_TOU_indu_i,K3_TOU_resi_i,K3_net_i,Carbon_F = Functions.TOU_period(city_name)
    df_plot.iloc[i,-2] = facade_power[i]*Carbon_F   #Million ton
    df_plot.iloc[i,-1] = roof_power[i]*Carbon_F   #Million ton

# Merge average building height data
df_plot = df_plot.merge(average_building_height, left_index=True, right_index=True)

# Calculate the ratio of facade to roof area and generation
df_plot['facade_roof_ratio_area'] = df_plot['Area_facade-0(km2)'] / df_plot['Area_roof-0(km2)']
df_plot['facade_roof_ratio_generation'] = df_plot['facade_power'] / df_plot['roof_power']

# Merge population data
df_plot = df_plot.merge(population_df, left_index=True, right_index=True)

df_plot


#%%import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#colors = ['#87CEEB', '#2E8B57']  # Blue, green corresponding to roof and facade (not used here)
colors = ['skyblue', 'mediumseagreen']
color_ratio = '#003366'
lw_axis = 0.8

# ----------------- Data Preparation -----------------
# df_plot index is city name, including:
#   city_level, Area_facade-0(km2), Area_roof-0(km2)
df_vis = df_plot.copy()
df_vis = df_vis.rename(index={
    'Haerbin': 'Harbin',
    'Huhehaote': 'Hohhot',
    'Wulumuqi': 'Urumqi',
    'Xian': "Xi'an"
})

# 1) Sort by city_level (descending) and facade_power (descending)
df_vis.sort_values(
    by=['city_level', 'facade_carbon'],
    ascending=[False, False],
    inplace=True
)

# 2) Split the cities into two parts
n = len(df_vis)
mid = n // 2
df_vis_part1 = df_vis.iloc[:mid]
df_vis_part2 = df_vis.iloc[mid:]

print(df_vis.loc[df_vis['city_level'] == 3, 'facade_carbon'].mean())
print(df_vis.loc[df_vis['city_level'] == 3, 'roof_carbon'].mean())

print(df_vis.loc[df_vis['city_level'] == 3, 'facade_carbon']/df_vis.loc[df_vis['city_level'] == 3, 'roof_carbon'])

print((df_vis.loc[df_vis['city_level'] == 0, 'facade_carbon']/df_vis.loc[df_vis['city_level'] == 0, 'roof_carbon']).max())
print(df_vis.loc[df_vis['city_level'] == 0, 'roof_carbon'].mean())

# Define a mapping dictionary for city levels (example, can be modified/expanded)
level_labels = {
    3: 'SLC',
    2: 'VLC',
    1: 'LC-I',
    0: 'LC-II'
}

# ----------------- Create Canvas and Subplots -----------------
font_options = {'size': fs-12}
plt.rc('font', **font_options)

fig, axes = plt.subplots(
    1, 2,
    figsize=np.array([figwidth, figwidth * 1.2]) / 2.54,
)

parts = [df_vis_part1, df_vis_part2]

for i_ax, df_part in enumerate(parts):
    ax = axes[i_ax]

    # Construct a temporary table temp_part, keeping city_level for later labeling
    temp_part = df_part[['city_level', 'facade_carbon', 'roof_carbon', 'facade_roof_ratio_generation']].copy()
    # Only two columns (facade power and roof power) for actual plotting
    df_plot_part = temp_part[['facade_carbon', 'roof_carbon']]

    # Plot horizontal bar chart
    df_plot_part.plot(
        kind='barh',
        ax=ax,
        stacked=False,
        width=0.8,
        edgecolor='none',
        linewidth=lw_axis,
        alpha=0.8,
        color=colors[::-1],  # Facade, Roof
        legend=False,
    )

    # Reverse y-axis for ascending order from top to bottom
    ax.invert_yaxis()

    ax.set_xlim(0, 90)
    ax.set_xticks(np.arange(0, 100, 15))
    ax.yaxis.set_tick_params(length=0)
    ax.set_xlabel("Carbon mitigation potential (Million ton)", fontsize=fs-9)
    ax.set_ylabel(None)

    # -------------- Labeling City Level --------------
    # Reset index to match row_idx with the y-axis order of barh
    temp_part.reset_index(inplace=True)       # Turn city names into a column
    temp_part['row_idx'] = temp_part.index    # Row index for y-axis in barh

    # Group by city_level, find the min and max row indices for each group on the y-axis
    grouped = temp_part.groupby('city_level')['row_idx']
    for lvl, rows in grouped:
        min_idx = rows.min()
        max_idx = rows.max()
        mid_idx = 0.5 * (min_idx + max_idx)  # Vertical middle

        # x_pos: offset from the far-right of the largest bar, used for drawing arrows and text
        x_pos = ax.get_xlim()[-1] * 0.83
        # Arrow length
        arrow_len = 0.5
        h_len = ax.get_xlim()[-1] * 0.04  # Half-length of horizontal line (example value)

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
        text_label = level_labels.get(lvl, f'Level {lvl}')
        ax.text(
            x_pos, mid_idx,
            text_label,
            ha='center', va='center',
            fontsize=fs-9,
            transform=ax.transData,
            bbox=dict(
                facecolor='white',         # Background color
                edgecolor='none'           # Remove border
            )
        )

    # -------------- Plot Ratio Scatter on the Same Subplot with ax.twiny() --------------
    # Create a new x-axis with a shared y-axis
    ax2 = ax.twiny()
    
    # y-values correspond to barh, need to use row_idx
    y_vals = temp_part['row_idx']
    # x-values are the ratio of FPV to RPV
    x_vals = temp_part['facade_roof_ratio_generation']
    
    # Plot scatter plot
    ax2.scatter(x_vals, y_vals, color=color_ratio, s=50,
            alpha=0.8, marker='o')
    
    # Set x-axis labels, limits, ticks, etc.
    ax2.set_xlim(0, 3)
    ax2.set_xticks(np.arange(0, 3.5, 0.5))
    ax2.set_xlabel("Ratio of FPV to RPV", fontsize=fs-9)
    
    # Ensure y-axis range matches ax
    ax2.set_ylim(ax.get_ylim())

# ----------------- Add Unified Legend -----------------
lg = fig.legend(
    handles=[
        mpatches.Patch(fc=colors[1], label='FPV carbon mitigation potential'),  # Green
        mpatches.Patch(fc=colors[0], label='RPV carbon mitigation potential'),    # Blue
        plt.Line2D([0], [0], marker='o', color='none', markerfacecolor=color_ratio,
                   markersize=7, label='Ratio of FPV to RPV')  # Dark blue circle
    ],
    handleheight=0.7,  # Default value 0.7
    handlelength=1.0,  # Default value 2
    loc='lower center',
    ncol=3,
    fontsize=fs - 9,
    bbox_to_anchor=(0.5, 0.00)  # Legend placed below the entire figure
)
frame = lg.get_frame()
frame.set_linewidth(0.6)
frame.set_edgecolor('black')
frame.set_facecolor('none')

# ----------------- Adjust Layout and Save -----------------
# Adjust subplot layout
plt.subplots_adjust(left=0.10, right=0.98, bottom=0.1, top=0.95,
		wspace=0.4, hspace=0
        )

fig.savefig('Figs_new/Fig2a.pdf',format='pdf',dpi=600,bbox_inches='tight')
fig.savefig("Figs_new/Fig2a.png", dpi=600,bbox_inches='tight')
plt.show()


#%%
## Facade area vs Roof area
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
cmap = 'RdPu'
vmin, vmax = (population_df['城区人口'].min())/1e2, (population_df['城区人口'].max())/1e2  # (4.99..., 13.52...)
norm = plt.Normalize(vmin=1, vmax=10)

color_fit1 = ["#c240402b","#bb24246e","#bb2c2cbd","#831717"]

font_options = {'size': fs-3}
plt.rc('font', **font_options)

fig, ax = plt.subplots(figsize=np.array([figwidth * 2/3, figwidth * 2/3 * 0.7]) / 2.54)
#color_fit1_list = [color_fit1[df_plot['city_level'][i]] for i in range(len(df_plot))]
# sc = ax.scatter(df_plot['Area_roof-0(km2)'], df_plot['Area_facade-0(km2)'],
#     c=color_fit1_list,
#     alpha=0.8, s=70, edgecolors='black', linewidths=lw)

sc = ax.scatter(df_plot['Area_roof-0(km2)'], df_plot['Area_facade-0(km2)'],
    c=population_df['城区人口']/1e2,cmap=cmap,norm=norm,
    alpha=0.8, s=70, edgecolors='black', linewidths=lw)

# Plot fitted line
x = df_plot['Area_roof-0(km2)'].values.reshape(-1, 1)
y = df_plot['Area_facade-0(km2)'].values

model = LinearRegression(fit_intercept=False)
model.fit(x, y)
k = model.coef_[0]
R2 = model.score(x, y)

x_fit = np.linspace(x.min(), x.max(), 100)
y_fit = k * x_fit
ax.plot(x_fit, y_fit, '-', color=color_fit1[-1], linewidth=lw2, 
        label=f'$y={k:.2f}x$')  #  R²={R2:.2f}
ax.text(
    0.4, 0.64,           # Position near the end of the fitted line
    f'$R^2={R2:.2f}$',  # Annotation text
    fontsize=fs - 6,
    color='red',
    transform=ax.transAxes,
    ha='right', va='bottom',
)

# Set axis labels
ax.set_xlim(0, 2000)
ax.set_ylim(0, 2500)
ax.set_xlabel('Roof', fontsize=fs-0)
ax.set_ylabel('Facade', fontsize=fs-0)
ax.set_title("Area (km$^2$)", fontsize=fs-5, y=0.88, fontweight='bold')
# Add y = x reference line
x_limits = ax.get_xlim()
x_line = np.linspace(x_limits[0], x_limits[1], 100)
ax.plot(x_line, x_line, linestyle='-', color='gray', linewidth=lw, label='$y=x$')

# Add grid and legend
ax.grid(alpha=0.5)

lg = ax.legend(loc='lower right', fontsize=fs - 6)
frame = lg.get_frame()
frame.set_linewidth(0.6)
frame.set_edgecolor('black')
frame.set_facecolor('none')

# Add colorbar to display the third variable (average building height)
cbar = plt.colorbar(sc, ax=ax)
#cbar.set_label('Urban population (Million)', fontsize=fs - 5)
cbar.set_ticks([1,3,5,10])  # 5 evenly spaced ticks, adjust if needed
cbar.ax.tick_params(labelsize=fs - 5)
for tt in range(4):
    labelsss = ['SLC','VLC','LC-I','LC-II']
    labelyyy = [0.98,0.7,0.35,0.15]
    cbar.ax.text(4.5,
                labelyyy[tt],
                labelsss[tt],
                ha='center', va='center',
                rotation=90,
                fontsize=fs - 5,
                color='black',
                transform=cbar.ax.transAxes)

# Adjust layout
plt.subplots_adjust(left=0.19, right=0.96, bottom=0.18, top=0.96)

fig.savefig('Figs_new/Fig2b.pdf',format='pdf',dpi=600,bbox_inches='tight')
fig.savefig("Figs_new/Fig2b.png", dpi=600,bbox_inches='tight')

plt.show()


#%%

solar_f = np.sum(np.load('Fig_input_data/solar_facade.npy'),axis = 1)/1e6  #TWh
solar_r = np.sum(np.load('Fig_input_data/solar_roof.npy'),axis = 1)/1e6  #TWh

# # Parameter settings
cmap = 'OrRd'
vmin, vmax = (population_df['城区人口'].min())/1e2, (population_df['城区人口'].max())/1e2  # (4.99..., 13.52...)
norm = plt.Normalize(vmin=1, vmax=10)

color_fit2 = "#521783"
color_fit1 ="#831717"
font_options = {'size': fs-3}
plt.rc('font', **font_options)

fig, ax = plt.subplots(figsize=np.array([figwidth * 2/3, figwidth * 2/3 * 0.7]) / 2.54)
# ax.spines[:].set_linewidth(lw)
# ax.tick_params(width=lw)  # Default: axes.linewidth and xtick.major.width are 0.8

# Scatter plot: the third variable controls the color
# sc = ax.scatter(solar_r,solar_f,
#     c=color_fit2,
#     alpha=0.8, s=70, edgecolors='black', linewidths=lw)

sc = ax.scatter(solar_r,solar_f,
    c=population_df['城区人口']/1e2,cmap=cmap,norm=norm,
    alpha=0.8, s=70, edgecolors='black', linewidths=lw)


# Plot fitted line
x = solar_r.reshape(-1,1)
y = solar_f.reshape(-1,1)

model = LinearRegression(fit_intercept=False)
model.fit(x, y)
k = model.coef_[0]
R2 = model.score(x, y)

x_fit = np.linspace(x.min(), x.max(), 100)
y_fit = k * x_fit
ax.plot(x_fit, y_fit, '-', color=color_fit2, linewidth=lw2, 
        label=f'$y={k[0]:.2f}x$')  #  R²={R2:.2f}
ax.text(
    0.95, 0.5,           # Position near the end of the fitted line
    f'$R^2={R2:.2f}$',  # Annotation text
    fontsize=fs - 6,
    color='red',
    transform=ax.transAxes,
    ha='right', va='bottom',
)

# Set axis labels
ax.set_xlim(0,600)
ax.set_ylim(0, 500)
ax.set_xlabel('Roof', fontsize=fs-0)
ax.set_ylabel('Facade', fontsize=fs-0)
ax.set_title("Annual solar radiation\n(TWh)", fontsize=fs-5, y=0.8, fontweight='bold')
# Add y = x reference line
x_limits = ax.get_xlim()
x_line = np.linspace(x_limits[0], x_limits[1], 100)
ax.plot(x_line, x_line, linestyle='-', color='gray', linewidth=lw, label='$y=x$')
ax.set_xticks(np.arange(0, 601, 100))
# Add grid and legend
ax.grid(alpha=0.5)

lg = ax.legend(loc='lower right', fontsize=fs - 6)
frame = lg.get_frame()
frame.set_linewidth(0.6)
frame.set_edgecolor('black')
frame.set_facecolor('none')


# Add colorbar to display the third variable (average building height)
cbar = plt.colorbar(sc, ax=ax)
#cbar.set_label('Urban population (Million)', fontsize=fs - 5)
cbar.set_ticks([1,3,5,10])  # 5 evenly spaced ticks, adjust if needed
cbar.ax.tick_params(labelsize=fs - 5)
for tt in range(4):
    labelsss = ['SLC','VLC','LC-I','LC-II']
    labelyyy = [0.98,0.7,0.35,0.15]
    cbar.ax.text(4.5,
                labelyyy[tt],
                labelsss[tt],
                ha='center', va='center',
                rotation=90,
                fontsize=fs - 5,
                color='black',
                transform=cbar.ax.transAxes)


# Adjust layout
plt.subplots_adjust(left=0.19, right=0.96, bottom=0.18, top=0.96)

fig.savefig('Figs_new/Fig2c.pdf',format='pdf',dpi=600,bbox_inches='tight')
fig.savefig("Figs_new/Fig2c.png", dpi=600,bbox_inches='tight')

plt.show()

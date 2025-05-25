#%%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap

## ------------------------ Plotting Parameters Setup ------------------------
ratio_plot = 3  # Scaling factor for plot dimensions. Larger plots ensure clarity even when converted to bitmap, hence enlarged.

figwidth =  8.5 * ratio_plot  # In centimeters, but the default unit is inches. 1 inch = 2.54 cm.
fs = 8 * ratio_plot  # Minimum font size for axis labels (6~14 pt).
lw = 0.4 * ratio_plot  # Line width for axes (0.25~1.5 pt).
lw2 = 0.75 * ratio_plot  # Line width for curves.
lw3 = 1 * ratio_plot  # Line width for bold curves.

grid_alpha = 0.5

import os
import numpy as np
import pandas as pd

# Specify the HDF file path for storing results
result_folder = r'Fig_input_data/' 
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
hdf_file = os.path.join(result_folder, 'Cap_facade_ideal_all_df.h5')

# If the HDF file already exists, read the data; otherwise, process the data and save it
Cap_facade_ideal_all_df = pd.read_hdf(hdf_file, key='df')
print("Data read from the existing HDF file")

Cap_facade_ideal_all_df

# Specify the HDF file path for storing results
result_folder = r'Fig_input_data/'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
hdf_file = os.path.join(result_folder, 'Cap_roof_ideal_all_df.h5')

# If the HDF file already exists, read the data
Cap_roof_ideal_all_df = pd.read_hdf(hdf_file, key='df')
print("Data read from the HDF file.")

Cap_roof_ideal_all_df


# Specify the HDF file path for storing results
result_folder = r'Fig_input_data/'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
hdf_file = os.path.join(result_folder, 'Power_facade_ideal_1_sum_all_df.h5')

# If the HDF file already exists, read the data; otherwise, process the data and save the results
Power_facade_ideal_1_sum_all_df = pd.read_hdf(hdf_file, key='df')
print("Data read from the HDF file.")

aaa = list(Power_facade_ideal_1_sum_all_df.index.get_level_values(0))

#########Notice: here the City's order is not the same with G_type and Others!!!!!!!!!!


# Specify the HDF file path for storing results
result_folder = r'Fig_input_data/'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
hdf_file = os.path.join(result_folder, 'Power_roof_ideal_1_sum_all_df.h5')

# If the HDF file already exists, read the data; otherwise, process the data and save the results
Power_roof_ideal_1_sum_all_df = pd.read_hdf(hdf_file, key='df')
print("Data read from the HDF file.")

Power_roof_ideal_1_sum_all_df



# Specify the HDF file path for storing results
result_folder = r'Fig_input_data/'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
hdf_file = os.path.join(result_folder, 'Grid_type_all_df.h5')

# If the HDF file already exists, read the data; otherwise, process the data and save the results
Grid_type_all_df = pd.read_hdf(hdf_file, key='df')
print("Data read from the HDF file.")

Grid_type_all_df


# Specify the HDF file path for storing results
result_folder = r'Fig_input_data/'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
hdf_file = os.path.join(result_folder, 'Grid_feas_static_all_df.h5')

# If the HDF file already exists, read the data; otherwise, process the data and save the results
Grid_feas_static_all_df = pd.read_hdf(hdf_file, key='df')
print("Data read from the HDF file.")

Grid_feas_static_all_df


# Specify the HDF file path for storing results
result_folder = r'Fig_input_data/'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
hdf_file = os.path.join(result_folder, 'Grid_feas_wea_all_df.h5')

# If the HDF file already exists, read the data; otherwise, process the data and save the results
Grid_feas_wea_all_df = pd.read_hdf(hdf_file, key='df')
print("Data read from the HDF file.")

Grid_feas_wea_all_df


import pandas as pd

# skew: skewness index - the skewness of the mean compared to the median;
# WWR: Window-to-Wall Ratio, consistent within the grid
# can_avg: other - mean height of vegetation canopy above ground;
# can_std: other - standard deviation of vegetation canopy height above ground
Grid_feature_label = pd.concat([Grid_feas_static_all_df.iloc[:, :-4],  # Excluding the last 4 columns: ['skew','WWR','can_avg','can_std']
        Grid_feas_wea_all_df['Global'],
        Cap_facade_ideal_all_df, Power_facade_ideal_1_sum_all_df],
        axis=1)

Grid_feature_label['util_hour'] = Grid_feature_label['Power_facade_ideal_1_sum'] / Grid_feature_label['Cap_facade_ideal'] / 1e3  # MWh/GW, multiplied by 1e3 for conversion

Grid_feature_label['generation_per_area'] = \
    Grid_feature_label['Power_facade_ideal_1_sum'] / Grid_type_all_df['facade_area'] * 1e3  # MWh/m2, multiplied by 1e3 to convert to kWh/m2

Grid_feature_label


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ------------------------- Read and prepare data -------------------------
df = Grid_feature_label.copy()

all_cols = df.columns.tolist()
label_col = all_cols[-3]    # The last column is the label: total annual generation. The second last column is installed capacity
feature_cols = all_cols[:-4]  # All columns except the last 2 are features

X = df[feature_cols]
y = df[label_col] / 1000  # Convert to GWh

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, 
            test_size=0.2, 
            random_state=42)

# ------------------------- Train the XGBoost model -------------------------
model = xgb.XGBRegressor(
    n_estimators=100,  # 'n_estimators': [50, 100, 200],  # Default value: 100 (number of weak learners)
    max_depth=6,  # 'max_depth': [3, 6, 9],  # Default value: 6 (maximum depth of the tree)
    learning_rate=0.2,  # 'learning_rate': [0.01, 0.1, 0.2],  # Default value: 0.3 (learning rate)
    random_state=42)
model.fit(X_train, y_train)

# ------------------------- Evaluate prediction performance -------------------------
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Take the square root of mean squared error
y_mean = np.mean(y_test)  # Calculate the mean of y_test
cv_rmse = (rmse / y_mean)

print(f"Test R^2: {r2:.3f}")
print(f"Test RMSE: {rmse:.3f}")
print(f"Test CV-RMSE: {cv_rmse*100:.2f}%")  # Display as percentage


import shap
import numpy as np

# ----------------- Define SHAP result storage path -----------------

shap_hdf_path = "Fig_input_data/annual_generation_per_grid_shap.hdf"  # HDF5 storage path

# ----------------- Check if SHAP results already exist, if so, load them -----------------
if os.path.exists(shap_hdf_path):
    print("Detected SHAP result file, loading...")
    shap_df_grid = pd.read_hdf(shap_hdf_path, key='shap_values')
    print("SHAP results loaded.")
else:
    print("No SHAP result file detected, calculating SHAP values...")
    
    # Build Explainer and calculate SHAP values
    explainer = shap.TreeExplainer(model, X_train)  # Inform SHAP about the training data
    shap_values_array = explainer.shap_values(X_test)  # Calculate SHAP values
    
    # **Convert SHAP results into DataFrame**
    shap_df_grid = pd.DataFrame(shap_values_array, columns=X_test.columns)  # Columns corresponding to features
    
    # Store SHAP results to HDF5
    os.makedirs(os.path.dirname(shap_hdf_path), exist_ok=True)
    shap_df_grid.to_hdf(shap_hdf_path, key='shap_values', mode='w')
    
    print(f"SHAP results have been calculated and stored in {shap_hdf_path}")

shap_values = shap_df_grid.values

X_test.columns



# Step 1: Compute SHAP importance and convert to Series (including feature names)
shap_importance = np.abs(shap_values).mean(axis=0)  # Calculate the mean absolute SHAP values for each feature
feature_names = X_test.columns  # Get the feature names from the test set

# Convert the SHAP importance values into a pandas Series with feature names as index
shap_series = pd.Series(shap_importance, index=feature_names)

# Step 2: Replace feature names with full names
name_map = {
    'lon': 'Longitude',
    'lat': 'Latitude',
    'density': 'Building density',
    'hei_avg': 'Mean building height',
    'hei_std': 'Std. of building height',
    'area_std': 'Std. of building footprint area',
    'complex': 'Complexity',
    'compact': 'Compactness',
    'volume': 'Number of building volumes',
    'outdoor': 'Mean outdoor distance',
    '12ratio': '12-ratio',
    'Global': 'Annual gross radiation'
}

# Map the feature names to their full descriptions
shap_series.index = shap_series.index.map(name_map)

# Step 3: Drop the '12-ratio' feature, as it's not needed
shap_series.drop('12-ratio', inplace=True)

# Step 4: Calculate the relative importance (normalize by the sum of all SHAP values)
shap_series /= shap_series.sum()

shap_series


#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

colors = ['mediumseagreen']  # Blue, Green, corresponding to Roof, Facade
lw_axis = 0.8

# Classification features
first_class = [
    'Building density',
    'Mean building height',
    'Std. of building height',
    'Std. of building footprint area',
    'Complexity',
    'Compactness',
    'Number of building volumes',
    'Mean outdoor distance'
]

second_class = [
    'Longitude',
    'Latitude',
    'Annual gross radiation'
]
all_class = first_class + second_class
# Separate and sort by values in descending order
shap_all = shap_series[all_class].sort_values(ascending=False)

# Set color
colors_bar = [colors[0]] * len(shap_all)

# Set font style
font_options = {'size': fs - 3}
plt.rc('font', **font_options)

# ---------- Create the figure and axis ----------
fig, ax = plt.subplots(
    figsize=np.array([figwidth * 2/3, figwidth * 2/3 * 1.4]) / 2.54
)

# Create positions for y-axis manually
positions = np.arange(len(shap_all))

# Plot horizontal bar chart (using positions for y-axis)
ax.barh(positions, shap_all.values, color=colors_bar, height=0.5, alpha=0.6)

# Hide default y-axis ticks
ax.set_yticks([])
ax.yaxis.tick_right()  # Keep the y-axis on the right if needed, remove if not required

# Add horizontal dashed line between two categories
#n1 = len(shap_all)
#ax.axhline(y=n1 - 0.5, color='grey', linestyle='--', linewidth=lw_axis)

# ---------- Set axis labels and others ----------
# Use log scale for x-axis
# Set the x-axis range (ensure all values are greater than 0)
x_min, x_max = 1e-3, 1  # Adjust based on data range
ax.set_xscale('log')
ax.set_xlim(x_min, x_max)
ax.set_ylim(-0.5, len(shap_all)-0.5)

# Reverse y-axis so that index 0 is at the top
ax.invert_yaxis()

ax.set_xlabel("Importance", fontsize=fs)
ax.set_ylabel("Feature", fontsize=fs)
ax.tick_params(axis='y', length=0)  # Hide y-axis ticks if needed

# ---------- Annotate each feature ----------
# Choose a suitable x_offset
x_offset = 0.0012
for i, feature_name in enumerate(shap_all.index):
    ax.text(x_offset, i, feature_name,
            ha='left', va='center',  # Align text to the right, center vertically
            fontsize=fs - 3)

plt.subplots_adjust(left=0.08, right=0.96, bottom=0.10, top=0.98)

fig.savefig('Figs_new/Fig4a_1.pdf',format='pdf',dpi=600,bbox_inches='tight')
fig.savefig("Figs_new/Fig4a_1.png", dpi=600,bbox_inches='tight')

plt.show()


#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

colors = ['#1515FF', 'mediumseagreen']  # Blue, Green, corresponding to Roof, Facade

# Data preprocessing: Extract x_col and y_col, removing NaN values
df_plot = Grid_feature_label.copy()
x_col, y_col = 'complex', 'Power_facade_ideal_1_sum'

mask = ~df_plot[x_col].isna() & ~df_plot[y_col].isna()
x_raw = df[x_col][mask].values
y_raw = df[y_col][mask].values / 1e3  # Convert to GWh

np.random.seed(42)  # Set random seed, 42 is a commonly used example value, can be replaced with any integer
sample_size = min(20000, len(x_raw))  # Avoid data size smaller than 20000
sample_idx = np.random.choice(len(x_raw), size=sample_size, replace=False, )
x = x_raw[sample_idx]
y = y_raw[sample_idx]

font_options = { 'size': fs-3}
plt.rc('font', **font_options)
fig, ax = plt.subplots(figsize=np.array([figwidth * 2/3, figwidth * 2/3 * 0.7]) / 2.54)

# Sort the data
sorted_indices = np.argsort(x)
x_sorted = x[sorted_indices]
y_sorted = y[sorted_indices]

# Scatter plot
sc = ax.scatter(x_sorted, y_sorted,  color=colors[-1],
        s=20, alpha=0.8, edgecolors='none')  # Default size 20

# Set axis labels, grid, and legend
ax.set_xlim(0, 1.2)
ax.set_xticks(np.arange(0, 1.2+0.1, 0.3))
ax.set_ylim(0, 180)
ax.set_yticks(np.arange(0, 181, 45))

ax.set_xlabel('Complexity (\u2013)', fontsize=fs-3)
ax.set_ylabel('Annual generation (GWh)', fontsize=fs-3)
ax.grid(alpha=0.5)
# ax.legend(loc='lower right', fontsize=fs-2)

# Adjust layout
plt.subplots_adjust(left=0.19, right=0.96, bottom=0.18, top=0.96,
		# wspace=0.25, hspace=0.21
        )

fig.savefig('Figs_new/Fig4a_2.pdf',format='pdf',dpi=600,bbox_inches='tight')
fig.savefig("Figs_new/Fig4a_2.png", dpi=600,bbox_inches='tight')
plt.show()

#%%
import Functions
from scipy.io import savemat,loadmat
Clu_center,Clu_days = np.load('Fig_input_data/Clu_center.npy'),np.load('Fig_input_data/Clu_days.npy')
path = '#ML_results/'
City_statistic = pd.read_excel(path+'City_statistic.xlsx',sheet_name = 'Class_Volume', index_col=0)
Building_number = []
Building_avghei = []
Building_y = []
Building_count = 0
Yuan_num_area_ele_car = np.zeros((4,4))
Yuan_s = 0
for cc in range(102):
    city_name = City_statistic.index[cc]
    print(city_name)
    _,K3_TOU_indu_i,K3_TOU_resi_i,K3_net_i,Carbon_F = Functions.TOU_period(city_name)
    G_type = np.load('#ML_results/Grid_type/'+'Grid_type_'+city_name+'.npy')
    N_gg = np.where(G_type[:,0] != 888)[0]
    P1 = '#ML_results/Power1/N_P_facade_ideal_1_'+city_name+'.npy'
    Power = np.sum(np.load(P1),axis = 1)
    Building_number.append(G_type[N_gg,3])
    Building_count += len(G_type[N_gg,3])
    Building_avghei.append(G_type[N_gg,1])
    Building_y.append(Power[N_gg]/1e3)
    

    if cc in [97,92,2,6,8,69]:
        data_path = '#Opt_results/'+city_name+'_hybrid_n.mat'
    else:
        data_path = '#Opt_results/'+city_name+'_hybrid.mat'
    Data = loadmat(data_path)
    R_Pow_f = Data['R_Pow_f']
    R_Car = Data['R_Car']     #np.zeros((N_grid,2,Y))
    Power_plan = sum(Clu_days[cc,d]*np.sum(R_Pow_f[:,:,d,:],axis = (1,2)) for d in range(12))
    Carbon_plan = np.sum(R_Car[:,0,:],axis = 1) - np.sum(R_Car[:,1,:],axis = 1)
    th_hh = 18
    th_std = [0,0]  #[np.percentile(G_type[N_gg,2], 50),np.percentile(G_type[N_gg,2], 50)]     #4  # 3.6#4[3.6,3.6]  # 
    th_aa = [200,200]  # [np.percentile(G_type[N_gg,3], 50),np.percentile(G_type[N_gg,3], 50)]   #200 #120  # 55#120[120,120]  #  
    list_form = [list(np.where((G_type[N_gg,1]>=th_hh)&(G_type[N_gg,2]>=th_std[1])&(G_type[N_gg,3]>=th_aa[1]))[0]),\
           list(np.where((G_type[N_gg,1]>=th_hh)&(G_type[N_gg,2]>=th_std[1])&(G_type[N_gg,3]<th_aa[0]))[0]),\
           list(np.where((G_type[N_gg,1]<th_hh)&(G_type[N_gg,2]>=th_std[0])&(G_type[N_gg,3]>=th_aa[1]))[0]),\
           list(np.where((G_type[N_gg,1]<th_hh)&(G_type[N_gg,2]>=th_std[0])&(G_type[N_gg,3]<th_aa[0]))[0])]
    for ii in range(4): #HnD, HnS, MnD, MnS
        Yuan_num_area_ele_car[ii,0] += len(list_form[ii])  #First column: number of buildings
        Yuan_num_area_ele_car[ii,1] += np.sum(G_type[list_form[ii],4])    #Facade ares
        Yuan_num_area_ele_car[ii,2] += np.sum(Power_plan[list_form[ii]])
        #Yuan_num_area_ele_car[ii,3] += Carbon_F*np.sum(Power_plan[list_form[ii]])
        Yuan_num_area_ele_car[ii,3] += np.sum(Carbon_plan[list_form[ii]])
    Yuan_s += len(N_gg)

Building_number = np.concatenate(Building_number)[mask][sample_idx]
Building_avghei = np.concatenate(Building_avghei)[mask][sample_idx]
Building_y = np.concatenate(Building_y)[mask][sample_idx]
#Building_avghei = df['hei_avg'][mask].values[sample_idx]
#Building_y = df['Power_facade_ideal_1_sum'][mask].values[sample_idx]/1e3  #MWh to GWh

print(np.min(Building_y), np.max(Building_y))

#%%
Building_y_normalized = (Building_y - Building_y.min()) / (Building_y.max() - Building_y.min()) * 100
fig = plt.figure(figsize=np.array([figwidth * 2/3, figwidth * 2/3 * 0.7]) / 2.54)
scatter = plt.scatter(
    Building_number, Building_avghei,
    s=Building_y*1,
    c=Building_y,
    cmap='viridis',
    alpha=0.6,
    edgecolors='black',
    linewidth=0.5
)
plt.xlim(-10, 4000)
plt.ylim(10, 40)
cbar = plt.colorbar(scatter, label='Point Size')
cbar.set_label('Annual generation (GWh)', fontsize=fs-3)
cbar.set_ticks([0,45,90,135,180])

plt.xlabel('Building number', fontsize=fs-3)
plt.ylabel('Mean height (m)', fontsize=fs-3)

plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()

fig.savefig('Figs_new/Fig4a_3.pdf',format='pdf',dpi=600,bbox_inches='tight')
fig.savefig("Figs_new/Fig4a_3.png", dpi=600,bbox_inches='tight')
plt.show()



#%%
Yuan_num_area_ele_car_ratio = 100*np.round(Yuan_num_area_ele_car / np.sum(Yuan_num_area_ele_car, axis=0),3)

#colors = ['#66b3ff', "#ffbe99", "#99ff99", "#b499ff"]
#colors = [ "#8ad6ab",'#fdf07c', "#7cb6bb", "#8a96b8"]

colors = [ "#69abd6","#7cf9fddc", "#7cbb9e", "#8a99b8"]

product_labels = ['HnD', 'HnS', 'MnD', 'MnS']
pie_titles = ['Cell number', 'Facade area', 'Power generation', 'Carbon mitigation']
fig, axes = plt.subplots(2, 2, figsize=(6,8))
axes = axes.ravel()
explode = [0.1, 0, 0, 0]  # 0.1表示向外偏移10%
for i in range(4):
    ax = axes[i]
    wedges, texts, autotexts = ax.pie(
        Yuan_num_area_ele_car_ratio[:, i], 
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': fs-10},
        pctdistance=0.6,
        radius = 1.2,
        shadow = False,
        explode=explode,
    )
    ax.set_title(pie_titles[i], fontsize=fs-7, fontweight='bold')

    wedges[0].set_edgecolor('blue')
    wedges[0].set_linewidth(1.2)
    
    for w in wedges[1:]:
        w.set_alpha(0.8)

plt.subplots_adjust(wspace=0.1, hspace=-0.2)

fig.legend(
    wedges,
    product_labels,
    loc='lower center',
    bbox_to_anchor=(0.5, 0.05),
    ncol=2,
    fontsize=fs-8
)

fig.savefig('Figs_new/Fig4b.pdf',format='pdf',dpi=600,bbox_inches='tight')
fig.savefig("Figs_new/Fig4b.png", dpi=600,bbox_inches='tight')
plt.show()
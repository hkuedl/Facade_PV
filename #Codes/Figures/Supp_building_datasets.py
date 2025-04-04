# %%
import geopandas as gpd
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
import pyproj
from shapely.ops import transform
from functools import partial

# Read data files
shp1 = gpd.read_file(r"Beijing_new_with_height\beijing.shp")

shp2 = gpd.read_file(r"Beijing_new_without_height\Beijing.shp")

shp3 = gpd.read_file(r"Beijing_90city\Beijing.shp")

shp4 = gpd.read_file(r"Beijing_major\Beijing.shp")

# Define projection transformation function
proj_wgs84 = pyproj.CRS('EPSG:4326')  # WGS84 coordinate system
proj_utm = pyproj.CRS('EPSG:32650')  # UTM coordinate system, UTM 50N for Beijing region

project = partial(
    pyproj.Transformer.from_crs(proj_wgs84, proj_utm, always_xy=True).transform
)

# Filter out None values and transform coordinates to calculate area
area_shp1 = [transform(project, geom).area for geom in shp1['geometry'] if geom is not None]
area_shp2 = [transform(project, geom).area for geom in shp2['geometry'] if geom is not None]
area_shp3 = shp3['Shape_Area'].tolist()
area_shp4 = [transform(project, geom).area for geom in shp4['geometry'] if geom is not None]

# Calculate height
height_shp1 = shp1['Height'].tolist()
height_shp4 = shp4['height'].tolist()

# Calculate volume
volume_shp1 = [(transform(project, geom).area * height) for geom, height in zip(shp1['geometry'], shp1['Height']) if geom is not None]
volume_shp4 = [(transform(project, geom).area * height) for geom, height in zip(shp4['geometry'], shp4['height']) if geom is not None]

# Filter out NaN values
def filter_nan(data):
    return [x for x in data if not np.isnan(x)]

# Custom ScalarFormatter to set font size for scientific notation
class FixedScalarFormatter(ScalarFormatter):
    def __init__(self, useMathText=True, **kwargs):
        super().__init__(useMathText=useMathText, **kwargs)
        self.set_powerlimits((0, 0))
        self.set_useOffset(False)
        self.set_useMathText(False)

    def _set_format(self):
        self.format = "%1.1f"
        self._useMathText = False

    def _set_offset(self):
        self.offset = ""

# Ensure the target folder exists
output_dir = r"picture"
os.makedirs(output_dir, exist_ok=True)

# Plot and save each histogram
def save_histogram(data, bins, color, xlabel, title, filename_prefix, max_x, max_value=None):
    data = filter_nan(data)
    if max_value is not None:
        data = [x for x in data if x <= max_value]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.hist(data, bins=bins, color=color, alpha=0.7)
    ax.set_xlabel(xlabel, fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.set_xlim(0, max_x)
    ax.tick_params(axis='y', labelsize=24)  # Retain y-axis ticks
    ax.set_yticks(ax.get_yticks())  # Retain y-axis ticks
    ax.set_yticklabels([])  # Hide y-axis labels
    ax.set_title(title, fontsize=24)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{filename_prefix}.pdf'), format='pdf', dpi=600)
    fig.savefig(os.path.join(output_dir, f'{filename_prefix}.png'), dpi=600)
    plt.close(fig)

# Building area distributions
save_histogram(area_shp1, bins=20, color='skyblue', xlabel='Area (m²)', title='Dataset-1', filename_prefix='Fig1a', max_x=2000, max_value=np.percentile(area_shp1, 99))
save_histogram(area_shp2, bins=20, color='green', xlabel='Area (m²)', title='Dataset-2', filename_prefix='Fig1b', max_x=2000, max_value=np.percentile(area_shp2, 99))
save_histogram(area_shp3, bins=20, color='red', xlabel='Area (m²)', title='Dataset-3', filename_prefix='Fig1c', max_x=2000, max_value=np.percentile(area_shp3, 99))
save_histogram(area_shp4, bins=20, color='purple', xlabel='Area (m²)', title='Dataset-4', filename_prefix='Fig1d', max_x=2000, max_value=np.percentile(area_shp4, 99))

# Height distributions
save_histogram(height_shp1, bins=20, color='skyblue', xlabel='Height (m)', title='Dataset-1', filename_prefix='Fig2a', max_x=30, max_value=np.percentile(height_shp1, 99))
save_histogram(height_shp4, bins=20, color='purple', xlabel='Height (m)', title='Dataset-4', filename_prefix='Fig2b', max_x=30, max_value=np.percentile(height_shp4, 99))

# Volume distributions
save_histogram(volume_shp1, bins=20, color='skyblue', xlabel='Volume (m³)', title='Dataset-1', filename_prefix='Fig3a', max_x=25000, max_value=np.percentile(volume_shp1, 99))
save_histogram(volume_shp4, bins=20, color='purple', xlabel='Volume (m³)', title='Dataset-4', filename_prefix='Fig3b', max_x=25000, max_value=np.percentile(volume_shp4, 99))
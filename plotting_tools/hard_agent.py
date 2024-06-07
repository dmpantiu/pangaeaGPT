import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import streamlit as st
import cartopy.feature as cfeature
import pandas as pd
import numpy as np
import os
import cartopy.io.shapereader as shpreader
import xarray as xr
import time
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Path to local shapefiles and bathymetry file
base_dir = os.path.join('plotting_data', 'shape_files')
bathymetry_file = os.path.join('plotting_data', 'bathymetry', 'etopo', 'ETOPO2v2c_f4.nc')
output_file = os.path.join('plotting_tools', 'temp_files', 'plot.png')

# Define the base color palette and levels for sampling stations
color_dict_sampling = {
    '0-50': '#D3C2B3', '50-100': '#D6CCC0', '100-250': '#DCD6CB', '250-500': '#D8DDCD',
    '500-750': '#D4DDCD', '750-1000': '#C5E7CF', '1000-1250': '#B9EDD3', '1250-1500': '#AEF1D6',
    '1500-2000': '#A8EEE7', '2000-2500': '#A5E0F3', '2500-3000': '#A0D0FC', '3000-3500': '#98C3FA',
    '3500-4000': '#8EB3FB', '4000-4500': '#879FFF', '4500-5000': '#8597FE', '5000-5500': '#838BFE',
    '5500-6000': '#8384FF', '6000-6500': '#8380FD', '6500-7000': '#837AFD'
}

# Define the base color palette and levels for master track map
color_dict_master_track = {
    '0-50': '#E0F7FF', '50-100': '#D4F1FF', '100-250': '#C6EBFF', '250-500': '#B9E5FF',
    '500-750': '#ACE0FF', '750-1000': '#9FD8FF', '1000-1250': '#93D2FF', '1250-1500': '#86CCFF',
    '1500-2000': '#79C6FF', '2000-2500': '#6DBFFF', '2500-3000': '#60B9FF', '3000-3500': '#53B2FF',
    '3500-4000': '#47ABFF', '4000-4500': '#3AA5FF', '4500-5000': '#2D9EFF', '5000-5500': '#2098FF',
    '5500-6000': '#1491FF', '6000-6500': '#078BFF', '6500-7000': '#007FFF'
}


def create_colormap(min_depth, max_depth, color_dict):
    start_time = time.time()
    colors = []
    levels = []
    for key, color in reversed(color_dict.items()):  # Reverse the order of colors
        depth_range = key.split('-')
        start_depth = 0
        end_depth = -int(depth_range[0])
        if start_depth >= min_depth and end_depth <= max_depth:
            levels.extend([start_depth, end_depth])
            colors.append(color)
    levels = np.linspace(min_depth, max_depth, len(colors) + 1)
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=len(levels) - 1)
    end_time = time.time()
    print(f"Colormap creation took {end_time - start_time:.2f} seconds")
    return levels, cmap


def plot_sampling_stations(main_title):
    total_start_time = time.time()

    step_start_time = time.time()
    dataset_path = os.path.join('current_data', 'dataset.csv')
    dataset = pd.read_csv(dataset_path)
    step_end_time = time.time()
    print(f"Loading dataset took {step_end_time - step_start_time:.2f} seconds")

    step_start_time = time.time()
    # Ensure the longitude and latitude columns are numeric
    dataset['Longitude'] = pd.to_numeric(dataset['Longitude'], errors='coerce')
    dataset['Latitude'] = pd.to_numeric(dataset['Latitude'], errors='coerce')

    # Drop rows with invalid longitude or latitude
    dataset = dataset.dropna(subset=['Longitude', 'Latitude'])

    # Find the event column(s)
    event_columns = [col for col in dataset.columns if 'Event' in col]
    if event_columns:
        # Use the first event column found
        event_col = event_columns[0]
    else:
        print("No event column found.")
        return

    # Drop duplicate events
    dataset = dataset.drop_duplicates(subset=[event_col])
    step_end_time = time.time()
    print(f"Data cleaning took {step_end_time - step_start_time:.2f} seconds")

    step_start_time = time.time()
    # Calculate the extent with padding
    min_lon = dataset['Longitude'].min() - 7
    max_lon = dataset['Longitude'].max() + 7
    min_lat = dataset['Latitude'].min() - 7
    max_lat = dataset['Latitude'].max() + 7

    # Print debug information
    print(f"Min Lon: {min_lon}, Max Lon: {max_lon}, Min Lat: {min_lat}, Max Lat: {max_lat}")

    # Ensure the extent is within valid bounds
    min_lon = max(min_lon, -180)
    max_lon = min(max_lon, 180)
    min_lat = max(min_lat, -90)
    max_lat = min(max_lat, 90)

    # Print debug information after bounds check
    print(
        f"Adjusted Min Lon: {min_lon}, Adjusted Max Lon: {max_lon}, Adjusted Min Lat: {min_lat}, Adjusted Max Lat: {max_lat}")

    # Calculate aspect ratio
    lon_range = max_lon - min_lon
    lat_range = max_lat - min_lat
    aspect_ratio = lon_range / lat_range

    # Dynamically set figure size based on aspect ratio
    width = 15
    height = width / aspect_ratio
    step_end_time = time.time()
    print(f"Extent calculation took {step_end_time - step_start_time:.2f} seconds")

    step_start_time = time.time()
    # Load bathymetry data within the extent
    ds = xr.open_dataset(bathymetry_file)
    bathymetry = ds['z'].sel(x=slice(min_lon, max_lon), y=slice(min_lat, max_lat))

    # Filter to include only depths (negative values)
    bathymetry = bathymetry.where(bathymetry < 0, drop=True)

    # Get the min and max elevation values in the bathymetry data
    min_depth = bathymetry.min().item()
    max_depth = bathymetry.max().item()

    # Ensure colormap includes 0
    if min_depth > -1:
        min_depth = -1

    # Create the colormap and levels
    levels, custom_cmap = create_colormap(min_depth, max_depth, color_dict_sampling)
    step_end_time = time.time()
    print(f"Bathymetry data loading and colormap creation took {step_end_time - step_start_time:.2f} seconds")

    step_start_time = time.time()
    fig, ax = plt.subplots(figsize=(width, height), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

    # Plot bathymetry data with the custom gradient
    bathy_plot = ax.contourf(bathymetry.x, bathymetry.y, bathymetry, levels=levels, cmap=custom_cmap,
                             transform=ccrs.PlateCarree())

    # Adding features from local shapefiles
    ocean_shp = shpreader.Reader(os.path.join(base_dir, 'ne_10m_ocean', 'ne_10m_ocean.shp'))
    land_shp = shpreader.Reader(os.path.join(base_dir, 'ne_10m_land', 'ne_10m_land.shp'))
    coastline_shp = shpreader.Reader(os.path.join(base_dir, 'ne_10m_coastline', 'ne_10m_coastline.shp'))

    ax.add_geometries(ocean_shp.geometries(), ccrs.PlateCarree(), facecolor='none', edgecolor='black', zorder=0)
    ax.add_geometries(land_shp.geometries(), ccrs.PlateCarree(), facecolor='lightgray', edgecolor='black', zorder=1)
    ax.add_geometries(coastline_shp.geometries(), ccrs.PlateCarree(), facecolor='none', edgecolor='black', zorder=2)

    ax.gridlines(draw_labels=True)

    # Plotting the sampling stations
    stations = dataset[['Longitude', 'Latitude', event_col]]
    ax.scatter(stations['Longitude'], stations['Latitude'], color='red', s=20, zorder=3, transform=ccrs.PlateCarree())

    # Calculate offsets as a percentage of the longitude and latitude range
    lon_offset = (max_lon - min_lon) * 0.015  # 1.5% of the longitude range
    lat_offset = (max_lat - min_lat) * 0.015  # 1.5% of the latitude range

    # Annotate each point with the event label
    for i, row in stations.iterrows():
        ax.text(row['Longitude'] + lon_offset, row['Latitude'] + lat_offset, row[event_col],
                transform=ccrs.PlateCarree(),
                fontsize=12, ha='left', color='black', weight='bold',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))

    plt.title(f'{main_title} - Sampling Stations', y=1.05, fontsize=15, weight='bold')

    # Create an axis on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.75, axes_class=plt.Axes)

    # Create the colorbar
    cbar = plt.colorbar(bathy_plot, cax=cax, orientation='vertical', label='Depth (m)')

    # Ensure temp_files directory exists
    plot_dir = os.path.join('plotting_tools', 'temp_files')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Save the plot as a PNG file
    plt.savefig(output_file, format='png')
    step_end_time = time.time()
    print(f"Plotting and saving the figure took {step_end_time - step_start_time:.2f} seconds")

    total_end_time = time.time()
    print(f"Total time for plot_sampling_stations: {total_end_time - total_start_time:.2f} seconds")

    if os.path.exists(output_file):
        st.session_state.saved_plot_path = output_file
        print(f"Plot saved to {output_file}")


def plot_master_track_map(main_title):
    total_start_time = time.time()

    step_start_time = time.time()
    dataset_path = os.path.join('current_data', 'dataset.csv')
    dataset = pd.read_csv(dataset_path)
    step_end_time = time.time()
    print(f"Loading dataset took {step_end_time - step_start_time:.2f} seconds")

    step_start_time = time.time()
    # Ensure the longitude and latitude columns are numeric
    dataset['Longitude'] = pd.to_numeric(dataset['Longitude'], errors='coerce')
    dataset['Latitude'] = pd.to_numeric(dataset['Latitude'], errors='coerce')

    # Drop rows with invalid longitude or latitude
    dataset = dataset.dropna(subset=['Longitude', 'Latitude'])

    # Find the event column(s)
    event_columns = [col for col in dataset.columns if 'Event' in col]
    if event_columns:
        # Use the first event column found
        event_col = event_columns[0]
    else:
        print("No event column found.")
        return

    step_end_time = time.time()
    print(f"Data cleaning took {step_end_time - step_start_time:.2f} seconds")

    step_start_time = time.time()
    # Calculate the extent with padding
    min_lon = dataset['Longitude'].min() - 5
    max_lon = dataset['Longitude'].max() + 5
    min_lat = dataset['Latitude'].min() - 5
    max_lat = dataset['Latitude'].max() + 5

    # Print debug information
    print(f"Min Lon: {min_lon}, Max Lon: {max_lon}, Min Lat: {min_lat}, Max Lat: {max_lat}")

    # Ensure the extent is within valid bounds
    min_lon = max(min_lon, -180)
    max_lon = min(max_lon, 180)
    min_lat = max(min_lat, -90)
    max_lat = min(max_lat, 90)

    # Print debug information after bounds check
    print(
        f"Adjusted Min Lon: {min_lon}, Adjusted Max Lon: {max_lon}, Adjusted Min Lat: {min_lat}, Adjusted Max Lat: {max_lat}")

    # Calculate aspect ratio
    lon_range = max_lon - min_lon
    lat_range = max_lat - min_lat
    aspect_ratio = lon_range / lat_range

    # Dynamically set figure size based on aspect ratio
    width = 15
    height = width / aspect_ratio
    step_end_time = time.time()
    print(f"Extent calculation took {step_end_time - step_start_time:.2f} seconds")

    step_start_time = time.time()
    # Load bathymetry data within the extent
    ds = xr.open_dataset(bathymetry_file)
    bathymetry = ds['z'].sel(x=slice(min_lon, max_lon), y=slice(min_lat, max_lat))

    # Filter to include only depths (negative values)
    bathymetry = bathymetry.where(bathymetry < 0, drop=True)

    # Get the min and max elevation values in the bathymetry data
    min_depth = bathymetry.min().item()
    max_depth = bathymetry.max().item()

    # Ensure colormap includes 0
    if min_depth > -1:
        min_depth = -1

    # Create the colormap and levels
    levels, custom_cmap = create_colormap(min_depth, max_depth, color_dict_master_track)
    step_end_time = time.time()
    print(f"Bathymetry data loading and colormap creation took {step_end_time - step_start_time:.2f} seconds")

    step_start_time = time.time()
    fig, ax = plt.subplots(figsize=(width, height), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

    # Plot bathymetry data with the custom gradient
    bathy_plot = ax.contourf(bathymetry.x, bathymetry.y, bathymetry, levels=levels, cmap=custom_cmap,
                             transform=ccrs.PlateCarree())

    # Adding features from local shapefiles
    ocean_shp = shpreader.Reader(os.path.join(base_dir, 'ne_10m_ocean', 'ne_10m_ocean.shp'))
    land_shp = shpreader.Reader(os.path.join(base_dir, 'ne_10m_land', 'ne_10m_land.shp'))
    coastline_shp = shpreader.Reader(os.path.join(base_dir, 'ne_10m_coastline', 'ne_10m_coastline.shp'))

    ax.add_geometries(ocean_shp.geometries(), ccrs.PlateCarree(), facecolor='none', edgecolor='black', zorder=0)
    ax.add_geometries(land_shp.geometries(), ccrs.PlateCarree(), facecolor='lightgray', edgecolor='black', zorder=1)
    ax.add_geometries(coastline_shp.geometries(), ccrs.PlateCarree(), facecolor='none', edgecolor='black', zorder=2)

    ax.gridlines(draw_labels=True)

    # Plotting the master track map (assuming master track data is provided in a similar CSV format)
    master_track_data = dataset[['Longitude', 'Latitude', event_col]]  # Modify as needed for master track data
    ax.plot(master_track_data['Longitude'], master_track_data['Latitude'], color='red', linestyle='-', linewidth=1,
            transform=ccrs.PlateCarree())

    plt.title(f'{main_title} - Master Track Map', y=1.05, fontsize=25, weight='bold')

    # Create an axis on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.75, axes_class=plt.Axes)

    # Create the colorbar
    cbar = plt.colorbar(bathy_plot, cax=cax, orientation='vertical', label='Depth (m)')

    # Ensure temp_files directory exists
    plot_dir = os.path.join('plotting_tools', 'temp_files')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Save the plot as a PNG file
    plt.savefig(output_file, format='png')
    step_end_time = time.time()
    print(f"Plotting and saving the figure took {step_end_time - step_start_time:.2f} seconds")

    total_end_time = time.time()
    print(f"Total time for plot_master_track_map: {total_end_time - total_start_time:.2f} seconds")

    if os.path.exists(output_file):
        st.session_state.saved_plot_path = output_file
        print(f"Plot saved to {output_file}")


def main():
    start_time = time.time()
    dataset_path = os.path.join('current_data', 'dataset.csv')
    data = pd.read_csv(dataset_path)

    # Print the first few rows of the dataframe for debugging
    print(data.head())

    # Plot the sampling stations
    plot_sampling_stations()
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
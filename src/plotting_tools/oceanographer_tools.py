#oceanographer_tools.py
import matplotlib.pyplot as plt
import os
import pandas as pd
import streamlit as st
import numpy as np
import gsw
from matplotlib.ticker import MaxNLocator
import logging

#output_file = os.path.join('tmp', 'figs', 'plot.png')


import uuid


# Add this function to generate unique image paths
def generate_unique_image_path():
    figs_dir = os.path.join('tmp', 'figs')
    os.makedirs(figs_dir, exist_ok=True)
    unique_path = os.path.join(figs_dir, f'fig_{uuid.uuid4()}.png')
    logging.debug(f"Generated unique image path: {unique_path}")
    return unique_path

# Define the CTD Plotting function
def plot_ctd_profiles(main_title, pressure_col, temperature_col, salinity_col, dataset_df):

    """
    Plots CTD profiles from the provided DataFrame.

    Parameters:
    - main_title: Title for the plot.
    - pressure_col: Column name for pressure data.
    - temperature_col: Column name for temperature data.
    - salinity_col: Column name for salinity data.
    """
    #dataset_path = os.path.join('data', 'current_data', 'dataset.csv')
    df = dataset_df

    fig, ax1 = plt.subplots(figsize=(14, 10))

    if temperature_col in df.columns:
        ax1.plot(df[temperature_col], df[pressure_col], 'r-', label='Temperature')
        ax1.set_xlabel('Temperature (°C)')
        ax1.set_ylabel('Depth (m)')
        ax1.invert_yaxis()
        ax1.set_title(main_title)
        ax1.legend(loc='upper left')
        ax1.grid(True)

    if salinity_col in df.columns:
        ax2 = ax1.twiny()
        ax2.plot(df[salinity_col], df[pressure_col], 'b-', label='Salinity')
        ax2.set_xlabel('Salinity (PSU)')
        ax2.legend(loc='upper right')
        ax2.grid(True)

    fig.tight_layout()
    plt.show()

    # Save the plot as a PNG file
    plot_path = generate_unique_image_path()
    plt.savefig(plot_path)

    if os.path.exists(plot_path):
        st.session_state.new_plot_path = plot_path
        print(f"Plot saved to {plot_path}")
        return {"result": "CTD Plot generated successfully."}
    else:
        print("Failed to generate CTD Plot.")
        return {"result": "Failed to generate CTD Plot."}


# Define the TS Diagram Plotting function
def plot_ts_diagram(main_title, temperature_col, salinity_col, dataset_df):
    """
    Plots a TS (Temperature-Salinity) diagram from the provided DataFrame.

    Parameters:
    - main_title: Title for the plot.
    - temperature_col: Column name for temperature data.
    - salinity_col: Column name for salinity data.
    """
    #dataset_path = os.path.join('data', 'current_data', 'dataset.csv')
    df = dataset_df

    # Find the minimum and maximum values of temperature and salinity
    mint, maxt = df[temperature_col].min(), df[temperature_col].max()
    mins, maxs = df[salinity_col].min(), df[salinity_col].max()

    # Generate temperature and salinity ranges
    tempL = np.linspace(mint - 0.5, maxt + 0.5, 156)
    salL = np.linspace(mins - 0.5, maxs + 0.5, 156)

    # Create a meshgrid of temperature and salinity
    Tg, Sg = np.meshgrid(tempL, salL)
    # Calculate seawater density
    sigma_theta = gsw.sigma0(Sg, Tg)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot isopycnals (lines of constant density)
    cs = ax.contour(Sg, Tg, sigma_theta, colors='lightgray', linewidths=0.5, zorder=1)
    cl = ax.clabel(cs, fontsize=8, inline=True, fmt='%.1f')

    # Scatter plot with depth as the color if available, otherwise use density
    if 'Depth [m]' in df.columns:
        depth_col = 'Depth [m]'
    elif 'Depth water [m]' in df.columns:
        depth_col = 'Depth water [m]'
    else:
        depth_col = None

    if depth_col:
        sc = ax.scatter(df[salinity_col], df[temperature_col], c=df[depth_col],
                        cmap='viridis', s=5, alpha=0.7)
        cb = plt.colorbar(sc)
        cb.set_label('Depth [m]', rotation=270, labelpad=15)
    else:
        density = gsw.sigma0(df[salinity_col].values, df[temperature_col].values)
        sc = ax.scatter(df[salinity_col], df[temperature_col], c=density,
                        cmap='viridis', s=5, alpha=0.7)
        cb = plt.colorbar(sc)
        cb.set_label('Density (kg m$^{-3}$)', rotation=270, labelpad=15)

    ax.set_xlabel('Salinity [PSU]')
    ax.set_ylabel('Potential Temperature θ [°C]')
    ax.set_title(main_title, fontsize=14, fontweight='bold')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.tick_params(direction='out')
    cb.ax.tick_params(direction='out')

    # Add sigma_theta label
    ax.text(0.02, 0.98, '$σ_θ$', transform=ax.transAxes, fontsize=12, va='top')

    plt.tight_layout()
    # Save the plot as a PNG file
    output_file = generate_unique_image_path()
    plt.savefig(output_file, format='png', dpi=300, transparent=False)

    if os.path.exists(output_file):
        st.session_state.new_plot_path = output_file
        print(f"Plot saved to {output_file}")
        return {"result": "TS Diagram generated successfully."}
    else:
        print("Failed to generate TS Diagram.")
        return {"result": "Failed to generate TS Diagram."}
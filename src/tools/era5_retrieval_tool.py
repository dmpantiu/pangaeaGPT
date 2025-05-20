"""
src/tools/era5_retrieval_tool.py

ERA5 data retrieval tool for use in the visualization agent.
Retrieves climate data from the Google Cloud ARCO-ERA5 dataset.
"""

import os
import logging
import uuid
import xarray as xr
import pandas as pd
import numpy as np
import streamlit as st
from pydantic import BaseModel, Field
from typing import Optional, Literal
from langchain_core.tools import StructuredTool

# A) Import-guard for gcsfs
try:
    import gcsfs
except ImportError as e:
    raise ImportError("gcsfs is required for ERA5 retrieval. Install with 'pip install gcsfs'") from e

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Public ARCO-ERA5 Zarr store
ARCO_ERA5_MAIN_ZARR_STORE = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'


class ERA5RetrievalArgs(BaseModel):
    variable_id: Literal[
        "sea_surface_temperature",
        "surface_pressure",
        "total_cloud_cover",
        "total_precipitation",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_temperature",
        "2m_dewpoint_temperature",
        "geopotential",
        "specific_humidity",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "vertical_velocity"
    ] = Field(description="ERA5 variable to retrieve (must match Zarr store names).")
    start_date: str = Field(description="Start date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS).")
    end_date: str = Field(description="End date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS).")
    min_latitude: float = Field(-90.0, description="Minimum latitude (–90 to 90).")
    max_latitude: float = Field(90.0, description="Maximum latitude (–90 to 90).")
    min_longitude: float = Field(0.0, description="Minimum longitude (0–360 or –180 to 360).")
    max_longitude: float = Field(359.75, description="Maximum longitude (0–360 or –180 to 360).")
    pressure_level: Optional[int] = Field(None, description="Pressure level in hPa for 3D variables.")


def retrieve_era5_data(
    variable_id: str,
    start_date: str,
    end_date: str,
    min_latitude: float = -90.0,
    max_latitude: float = 90.0,
    min_longitude: float = 0.0,
    max_longitude: float = 359.75,
    pressure_level: Optional[int] = None
) -> dict:
    """
    Retrieves a subset of the public ARCO-ERA5 Zarr dataset and saves it locally
    as a Zarr store.
    """
    try:
        logging.info(
            f"ERA5 retrieval: var={variable_id}, "
            f"time={start_date}→{end_date}, "
            f"lat=[{min_latitude},{max_latitude}], "
            f"lon=[{min_longitude},{max_longitude}], "
            f"level={pressure_level}"
        )

        # 1) Determine or create a sandbox directory  
        main_dir = None
        if "active_datasets" in st.session_state and st.session_state["active_datasets"]:
            doi = next(iter(st.session_state["active_datasets"]))
            cached = st.session_state["datasets_cache"].get(doi)
            if cached:
                path = cached[0]
                if isinstance(path, str) and os.path.isdir(path):
                    main_dir = os.path.dirname(os.path.abspath(path))
        if not main_dir:
            main_dir = os.path.join("tmp", "sandbox", uuid.uuid4().hex)
            logging.info(f"Created new sandbox: {main_dir}")
        os.makedirs(main_dir, exist_ok=True)

        era5_dir = os.path.join(main_dir, "era5_data")
        os.makedirs(era5_dir, exist_ok=True)
        logging.info(f"ERA5 output directory: {era5_dir}")
        
        # 2) Use the simple approach that works in the test script
        logging.info(f"Opening ERA5 dataset: {ARCO_ERA5_MAIN_ZARR_STORE}")
        ds = xr.open_zarr(
            ARCO_ERA5_MAIN_ZARR_STORE, 
            consolidated=True, 
            storage_options={'token': 'anon'}
        )
        
        # 3) Select the desired variable
        logging.info(f"Selecting variable: {variable_id}")
        if variable_id not in ds:
            available_vars = list(ds.data_vars)
            raise ValueError(f"Variable '{variable_id}' not found. Available variables: {available_vars}")
            
        var_data = ds[variable_id]
        
        # Log the coordinate ranges for debugging
        logging.info(f"Longitude range in dataset: {ds.longitude.values.min()} to {ds.longitude.values.max()}")
        logging.info(f"Latitude range in dataset: {ds.latitude.values.min()} to {ds.latitude.values.max()}")
        
        # 4) Apply time and spatial subsetting
        logging.info("Applying time and spatial subsetting")
        
        # Parse dates to datetime objects for consistent handling
        start_datetime = pd.to_datetime(start_date)
        end_datetime = pd.to_datetime(end_date)
        
        # Select data within the time range
        logging.info(f"Selecting time range: {start_datetime} to {end_datetime}")
        time_filtered = var_data.sel(time=slice(start_datetime, end_datetime))
        
        # Select data within the spatial bounds
        logging.info(f"Selecting spatial bounds: lat [{min_latitude}, {max_latitude}], lon [{min_longitude}, {max_longitude}]")
        
        # Note: in some datasets, latitude is stored in descending order
        lat_slice = slice(max_latitude, min_latitude) 
        
        # Convert longitude values to 0-360 range
        lon_min_360 = min_longitude % 360
        lon_max_360 = max_longitude % 360
        
        logging.info(f"Original longitude range: [{min_longitude}, {max_longitude}]")
        logging.info(f"Converted longitude range: [{lon_min_360}, {lon_max_360}]")
        
        # Check if we're crossing the 0/360 meridian
        if lon_min_360 > lon_max_360:
            logging.info("Crossing the 0/360 meridian, handling special case")
            
            # Create a mask for longitudes that are either >= lon_min_360 OR <= lon_max_360
            lon_mask = (ds.longitude >= lon_min_360) | (ds.longitude <= lon_max_360)
            
            # Split the selection into two parts and concatenate them
            part1 = time_filtered.sel(latitude=lat_slice, longitude=slice(lon_min_360, 360))
            part2 = time_filtered.sel(latitude=lat_slice, longitude=slice(0, lon_max_360))
            
            # Check if both parts have data
            if part1.sizes['longitude'] > 0 and part2.sizes['longitude'] > 0:
                logging.info(f"Part 1 longitude count: {part1.sizes['longitude']}")
                logging.info(f"Part 2 longitude count: {part2.sizes['longitude']}")
                space_filtered = xr.concat([part1, part2], dim='longitude')
            elif part1.sizes['longitude'] > 0:
                logging.info("Only part 1 has data")
                space_filtered = part1
            elif part2.sizes['longitude'] > 0:
                logging.info("Only part 2 has data")
                space_filtered = part2
            else:
                logging.warning("No data found for the specified longitude range")
                space_filtered = time_filtered.sel(latitude=lat_slice)
        else:
            # Normal case (no wrapping)
            logging.info("Standard longitude selection (no wrapping)")
            space_filtered = time_filtered.sel(
                latitude=lat_slice,
                longitude=slice(lon_min_360, lon_max_360)
            )
        
        # Add pressure level selection if applicable
        if pressure_level is not None and "level" in space_filtered.coords:
            logging.info(f"Selecting pressure level: {pressure_level}")
            space_filtered = space_filtered.sel(level=pressure_level)
        
        # 5) Check if we have data
        logging.info(f"Data shape after selection: latitude={space_filtered.sizes.get('latitude', 0)}, longitude={space_filtered.sizes.get('longitude', 0)}, time={space_filtered.sizes.get('time', 0)}")
        
        if (space_filtered.sizes.get('longitude', 0) == 0 or 
            space_filtered.sizes.get('latitude', 0) == 0):
            msg = "Selected region contains no data points. Please check your coordinates."
            logging.warning(msg)
            return {
                "success": False,
                "error": msg,
                "message": f"Failed to retrieve ERA5 data: {msg}"
            }
        
        # 5) Load the data into memory
        logging.info("Loading data into memory")
        loaded_data = space_filtered.load()
        
        # 6) Create an xarray Dataset with just this variable
        subset_ds = xr.Dataset({variable_id: loaded_data})
        
        # Add metadata
        subset_ds.attrs['title'] = f"ERA5 {variable_id} data"
        subset_ds.attrs['description'] = f"Subset of ERA5 {variable_id} for {start_date} to {end_date}"
        subset_ds.attrs['source'] = ARCO_ERA5_MAIN_ZARR_STORE
        
        # 7) Define Zarr store path
        uid = uuid.uuid4().hex
        zarr_filename = f"{variable_id}_{uid}.zarr"
        zarr_path = os.path.join(era5_dir, zarr_filename)

        # 8) Save to Zarr store
        logging.info(f"Saving to Zarr store: {zarr_path}")
        subset_ds.to_zarr(zarr_path, mode='w') # Use mode='w' to overwrite if exists

        # 9) Return success with Zarr file path
        return {
            "success": True,
            "output_path_zarr": zarr_path,
            "variable": variable_id,
            "message": f"ERA5 {variable_id} data retrieved and saved as Zarr successfully to {zarr_path}"
        }

    except Exception as e:
        logging.error(f"Error in ERA5 retrieval: {e}", exc_info=True)
        error_msg = str(e)
        if "Expected a BytesBytesCodec" in error_msg:
            error_msg += " (This is likely due to a version incompatibility with zarr/numcodecs libraries)"
            
        return {
            "success": False,
            "error": error_msg,
            "message": f"Failed to retrieve ERA5 data: {error_msg}"
        }


# Register the tool for the agent system
era5_retrieval_tool = StructuredTool.from_function(
    func=retrieve_era5_data,
    name="retrieve_era5_data",
    description=(
        "Retrieves a subset of the ARCO-ERA5 Zarr climate reanalysis dataset "
        "for a given variable, time range, spatial bounds, and optional pressure level. "
        "Saves the data as a Zarr store and returns the path to it."
    ),
    args_schema=ERA5RetrievalArgs
)
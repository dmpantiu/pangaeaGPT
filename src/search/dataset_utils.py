#src/search/dataset_utils.py
import os
import shutil
import pandas as pd
import logging
import streamlit as st
import pangaeapy.pandataset as pdataset

# Function to fetch dataset based on DOI
#@st.cache_data(ttl=3600)
def fetch_dataset(doi):
    if doi in st.session_state.datasets_cache:
        logging.debug("Dataset for DOI %s already in cache.", doi)
        dataset, name = st.session_state.datasets_cache[doi]
        st.session_state.dataset_dfs[doi] = dataset
        st.session_state.dataset_names[doi] = name
        return dataset, name

    dataset_id = doi.split('.')[-1].strip(')')
    try:
        logging.debug("Fetching dataset for DOI %s with ID %s", doi, dataset_id)
        ds = pdataset.PanDataSet(int(dataset_id))
        logging.debug("Dataset fetched with title: %s", ds.title)

        # Removed code that saves dataset to disk

        st.session_state.datasets_cache[doi] = (ds.data, ds.title)
        st.session_state.dataset_dfs[doi] = ds.data
        st.session_state.dataset_names[doi] = ds.title
        return ds.data, ds.title
    except Exception as e:
        logging.error("Error fetching dataset for DOI %s: %s", doi, e)
        return None, None


# Function to fetch dataset details using pangaeapy
def fetch_dataset_details(doi):
    try:
        dataset = pdataset.PanDataSet(id=doi)
        dataset.setMetadata()
        abstract = getattr(dataset, 'abstract', "No description available") or "No description available"
        param_dict = dataset.getParamDict()
        short_names = param_dict.get('shortName', [])
        parameters = ', '.join(short_names) + "..." if len(short_names) > 10 else ', '.join(short_names)

        return abstract, parameters

    except Exception as e:
        logging.error(f"Error fetching dataset details for DOI {doi}: {e}")
        return "No description available", "No parameters available"

# Conversion function
def convert_df_to_csv(df):
    logging.debug("Converting DataFrame to CSV")
    return df.to_csv().encode('utf-8')

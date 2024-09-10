import os
import shutil
import pandas as pd
import logging
import streamlit as st
import pangaeapy.pandataset as pdataset

# Function to fetch dataset based on DOI
#@st.cache_data(ttl=3600)
def fetch_dataset(doi):
    if st.session_state.dataset_df is not None and st.session_state.active_dataset == doi:
        logging.debug("Dataset for DOI %s already loaded.", doi)
        return st.session_state.dataset_df, st.session_state.dataset_name

    dataset_folder = os.path.join(os.getcwd(), 'current_data')
    dataset_csv_path = os.path.join(dataset_folder, 'dataset.csv')

    if doi in st.session_state.datasets_cache:
        logging.debug("Dataset for DOI %s already in cache.", doi)
        dataset, name = st.session_state.datasets_cache[doi]
        st.session_state.dataset_df = dataset
        st.session_state.dataset_name = name
        st.session_state.active_dataset = doi
        return dataset, name

    dataset_id = doi.split('.')[-1].strip(')')
    try:
        logging.debug("Fetching dataset for DOI %s with ID %s", doi, dataset_id)
        ds = pdataset.PanDataSet(int(dataset_id))
        logging.debug("Dataset fetched with title: %s", ds.title)

        if os.path.exists(dataset_folder):
            shutil.rmtree(dataset_folder)
        os.makedirs(dataset_folder)

        ds.data.to_csv(dataset_csv_path, index=False)

        st.session_state.datasets_cache[doi] = (ds.data, ds.title)
        st.session_state.dataset_df = ds.data
        st.session_state.dataset_name = ds.title
        st.session_state.active_dataset = doi
        return ds.data, ds.title
    except Exception as e:
        logging.error("Error fetching dataset for DOI %s: %s", doi, e)
        return None, None

# Conversion function
def convert_df_to_csv(df):
    logging.debug("Converting DataFrame to CSV")
    return df.to_csv().encode('utf-8')

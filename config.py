# config.py

import logging
import os
from datetime import datetime
import streamlit as st


# Create logs directory if it doesn't exist
logs_dir = os.path.join(os.getcwd(), 'logs')
os.makedirs(logs_dir, exist_ok=True)

# Configure logging
log_filename = f'app_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
log_filepath = os.path.join(logs_dir, log_filename)
logging.basicConfig(
    filename=log_filepath,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define constants
API_KEY = st.secrets["general"]["openai_api_key"]
LANGCHAIN_API_KEY = st.secrets["general"]["langchain_api_key"]
LANGCHAIN_PROJECT_NAME = st.secrets["general"]["langchain_project_name"]


#DATASET_FOLDER = os.path.join(os.getcwd(), 'data', 'current_data')
#DATASET_CSV_PATH = os.path.join(DATASET_FOLDER, 'dataset.csv')
PLOT_OUTPUT_DIR = os.path.join('src', 'plotting_tools', 'temp_files')


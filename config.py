# config.py

import logging
import os
from datetime import datetime

# Removed import of streamlit as st

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

# Define constants without API keys
PLOT_OUTPUT_DIR = os.path.join('tmp', 'figs')

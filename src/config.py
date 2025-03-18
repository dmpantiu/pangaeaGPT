# src/config.py

import os
import streamlit as st
import yaml
from datetime import datetime
import logging

# --- Load Central Configuration ---
CONFIG_FILE = os.path.join(os.getcwd(), "config.yaml")
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r") as f:
        app_config = yaml.safe_load(f)
else:
    app_config = {}

# Export deployment mode for use in other modules
DEPLOYMENT_MODE = app_config.get("deployment_mode", "huggingface")  # default to local if not specified

# --- Set API Keys based on Deployment Mode ---
if DEPLOYMENT_MODE == "local":
    API_KEY = st.secrets["general"]["openai_api_key"]
    LANGCHAIN_API_KEY = st.secrets["general"]["langchain_api_key"]
    # Use the environment variable if it exists; otherwise, fall back to st.secrets
    LANGCHAIN_PROJECT_NAME = os.environ.get("LANGCHAIN_PROJECT_NAME", st.secrets["general"]["langchain_project_name"])
else:
    API_KEY = os.environ.get("OPENAI_API_KEY", "")
    LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY", "")
    LANGCHAIN_PROJECT_NAME = os.environ.get("LANGCHAIN_PROJECT_NAME", "")

# --- Logging Setup (unchanged) ---
logs_dir = os.path.join(os.getcwd(), 'logs')
os.makedirs(logs_dir, exist_ok=True)

log_filename = f'app_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
log_filepath = os.path.join(logs_dir, log_filename)
logging.basicConfig(
    filename=log_filepath,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

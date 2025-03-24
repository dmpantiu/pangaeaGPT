#app.py

import streamlit as st
import os
import importlib
import logging
from io import StringIO           
import pandas as pd
import xarray as xr                
import json                       
import uuid
import base64
from pathlib import Path


import src.config as config
from src.config import DEPLOYMENT_MODE
# ------------------------------------------------------------------
# Set Page Config first! (This must be the very first Streamlit command)
st.set_page_config(
    page_title="PANGAEA GPT",
    page_icon="img/pangaea-logo.png",
    layout="wide"
)
# ------------------------------------------------------------------

# === Step 1: Force OpenAI API Key ===
if "openai_api_key" not in st.session_state:
    if DEPLOYMENT_MODE == "local":
        try:
            st.session_state["openai_api_key"] = st.secrets["general"]["openai_api_key"]
        except KeyError:
            st.sidebar.warning("⚠️ OpenAI API key not found in .streamlit/secrets.toml. Please enter it below.")
            user_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
            if user_api_key:
                st.session_state["openai_api_key"] = user_api_key
                st.rerun()
            st.stop()
    else:  # Hugging Face mode
        st.sidebar.warning("⚠️ Please enter your OpenAI API key below to enable full functionality.")
        user_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
        if user_api_key:
            st.session_state["openai_api_key"] = user_api_key
            st.rerun()
        st.stop()

API_KEY = st.session_state["openai_api_key"]
os.environ["OPENAI_API_KEY"] = API_KEY

# === Step 2: Display LangSmith Fields in the Sidebar ===
def img_to_base64(img_path):
    with open(img_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

with st.sidebar:
    # Using the path we know works
    logo_path = "img/pangaea-logo.png"
    
    # Display the logo centered using base64 encoding
    if Path(logo_path).exists():
        img_base64 = img_to_base64(logo_path)
        st.markdown(
            f"""
            <div style="display: flex; justify-content: center;">
                <img src="data:image/png;base64,{img_base64}" width="120">
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.title("Configuration")
    # Rest of the sidebar code...

    model_name = st.selectbox(
        "Select Model",
        ["gpt-4o", "gpt-4o-mini", "o1-mini", "o3-mini"],
        index=0,
        key="model_name"
    )
# Determine initial values for LangSmith inputs based on deployment mode
    if DEPLOYMENT_MODE == "local":
        try:
            initial_langsmith_api_key = st.secrets["general"]["langchain_api_key"]
        except KeyError:
            initial_langsmith_api_key = ""
        try:
            initial_langsmith_project_name = st.secrets["general"]["langchain_project_name"]
        except KeyError:
            initial_langsmith_project_name = ""
    else:
        initial_langsmith_api_key = ""
        initial_langsmith_project_name = ""

    # LangSmith API key input (pre-filled in local mode if available)
    langsmith_api_key = st.text_input(
        "LangSmith API Key (optional)",
        type="password",
        value=initial_langsmith_api_key,
        key="langsmith_api_key"
    )
    # LangSmith project name input (pre-filled in local mode if available)
    langsmith_project_name = st.text_input(
        "LangSmith Project Name (optional)",
        value=initial_langsmith_project_name,
        key="langsmith_project_name"
    )

# === Step 3: Update Environment Variables for LangSmith Keys ===
langchain_api_key = st.session_state.get("langsmith_api_key") or ""
langchain_project_name = st.session_state.get("langsmith_project_name") or ""

# Add region support
if DEPLOYMENT_MODE == "local":
    try:
        langchain_region = st.secrets["general"].get("langchain_region", "us")
    except KeyError:
        langchain_region = "us"  # Default to US region
else:
    langchain_region = "us"  # Default for non-local deployment

# Set the endpoint based on region
if langchain_region.lower() == "eu":
    langchain_endpoint = "https://eu.api.smith.langchain.com"
else:
    langchain_endpoint = "https://api.smith.langchain.com"

if langchain_api_key and langchain_project_name:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
    os.environ["LANGCHAIN_PROJECT_NAME"] = langchain_project_name
    os.environ["LANGCHAIN_ENDPOINT"] = langchain_endpoint
    os.environ["LANGCHAIN_REGION"] = langchain_region

# === Step 4: Reload the Config Module so It Picks Up the Updated Environment Variables ===
import src.config as config
importlib.reload(config)

# === Step 5: Import Configuration Values, Styles, Utilities, and Agent Functions ===
from src.config import API_KEY, LANGCHAIN_API_KEY, LANGCHAIN_PROJECT_NAME, DEPLOYMENT_MODE
from src.ui.styles import CUSTOM_UI
from src.utils import log_history_event
from langchain_core.messages import HumanMessage, AIMessage
from src.memory import CustomMemorySaver


from PIL import Image
SYSTEM_ICON = "img/11111111.png"
USER_ICON = "img/2222222.png"

# Import logic functions from main.py
from main import (
    initialize_session_state,
    get_search_agent,
    process_search_query,
    add_user_message_to_search,
    add_assistant_message_to_search,
    load_selected_datasets_into_cache,
    set_active_datasets_from_selection,
    get_datasets_info_for_active_datasets,
    add_user_message_to_data_agent,
    add_assistant_message_to_data_agent,
    create_and_invoke_supervisor_agent,
    convert_dataset_to_csv,
    has_new_plot,
    reset_new_plot_flag,
    get_dataset_csv_name,
    set_current_page,
    set_selected_text,
    set_show_dataset,
    set_dataset_for_data_agent,
    ensure_memory,
    ensure_thread_id,
)
from langchain_community.chat_message_histories import ChatMessageHistory

# === Step 6: Initialize Session State ===
initialize_session_state(st.session_state)

# === Step 7: Load Custom UI Styles (for example, CSS) ===
st.markdown(CUSTOM_UI, unsafe_allow_html=True)

# === Step 8: Create the Search Agent (which now uses the updated keys) ===
search_agent = get_search_agent(
    datasets_info=st.session_state.datasets_info,
    model_name=st.session_state["model_name"],
    api_key=API_KEY
)

# -------------------------
# SEARCH PAGE (Dataset Explorer)
# -------------------------
if st.session_state.current_page == "search":
    st.markdown("## Dataset Explorer")
    chat_placeholder = st.container()
    message_placeholder = st.empty()
    chat_input_container = st.container()

    predefined_queries = [
        "Search for data on gelatinous zooplankton in the Fram Strait?",
        "Continuous records of the atmospheric greenhouse gases",
        "Find datasets about renewable energy sources.",
        "Search for prokaryote abundance data on Hakon Mosby volcano",
        "Global distributions of coccolithophores abundance and biomass",
        "Shipboard acoustic doppler current profiling during POSEIDON cruise P414 (POS414)", 
        "Processed data of CTD buoys 2019O1 to 2019O8 MOSAiC"
    ]
    selected_query = st.selectbox(
        "Select an example or write down your query:",
        [""] + predefined_queries,
        index=0,
        key="selected_query",
    )
    if selected_query != "":
        set_selected_text(st.session_state, selected_query)
    else:
        set_selected_text(st.session_state, st.session_state.get("selected_text", ""))

    # Display chat messages (including any table with initial search results)
    with chat_placeholder:
        print(f"DEBUG: USER_ICON type: {type(USER_ICON)}, value: {USER_ICON}")
        for i, message in enumerate(st.session_state.messages_search):
            if message["role"] == "system":
                with st.chat_message(message["role"], avatar=SYSTEM_ICON):
                    st.markdown(message["content"])
            elif message["role"] == "user":
                with st.chat_message(message["role"], avatar=USER_ICON):
                    st.markdown(message["content"])
            else:  # assistant messages
                with st.chat_message(message["role"], avatar=SYSTEM_ICON):
                    st.markdown(message["content"])
            if "table" in message:
                df = pd.read_json(StringIO(message["table"]), orient="split")
                for index, row in df.iterrows():
                    cols = st.columns([1, 2, 2, 4, 2, 1])
                    cols[0].write(f"**#{row['Number']}**")
                    cols[1].write(f"**Name:** {row['Name']}")
                    cols[2].write(f"**DOI:** [{row['DOI']}]({row['DOI']})")
                    with cols[3].expander("See full description"):
                        st.write(row['Description'])
                    parameters_list = row['Parameters'].split(", ")
                    if len(parameters_list) > 7:
                        parameters_list = parameters_list[:7] + ["..."]
                    cols[4].write(f"**Parameters:** {', '.join(parameters_list)}")
                    checkbox_key = f"select-{i}-{index}"
                    with cols[5]:
                        selected = st.checkbox("Select", key=checkbox_key)
                    if selected:
                        st.session_state.selected_datasets.add(row['DOI'])
                    else:
                        st.session_state.selected_datasets.discard(row['DOI'])

    # --- Single search input form ---
    with chat_input_container:
        st.markdown("<br>" * 2, unsafe_allow_html=True)
        st.markdown("""
            <style>
            .stFormSubmitButton > button {
                background-color: rgba(67, 163, 151, 0.6) !important; 
                color: white !important;
                border: none !important;
                border-radius: 6px !important;
                padding: 8px 14px !important;
                cursor: pointer !important;
                font-size: 14px !important;
                box-shadow: 0 0 15px rgba(67, 163, 151, 0.3) !important;
            }
            .stFormSubmitButton > button:hover {
                background-color: white !important;
                color: rgb(67, 163, 151) !important;
                border: 1px solid rgb(67, 163, 151) !important;
                box-shadow: 0 0 20px rgba(67, 163, 151, 0.5) !important;
            }
            .stFormSubmitButton > button:active {
                transform: translateY(1px) !important;
            }
            </style>
        """, unsafe_allow_html=True)
        with st.form(key='chat_form', clear_on_submit=True):
            user_input = st.text_input(
                "Enter your query:",
                value=st.session_state.get("selected_text", ""),
                key="chat_input",
            )
            submit_button = st.form_submit_button(label='Send')
        if submit_button and user_input:
            st.session_state.selected_text = ""
            st.session_state.messages_search.append({"role": "user", "content": user_input})
            logging.debug("User input: %s", user_input)
            log_history_event(
                st.session_state, 
                "user_message", 
                {"page": "search", "content": user_input}
            )
            with message_placeholder:
                st.info("Work in progress...")
            ai_message = process_search_query(user_input, search_agent, st.session_state)
            st.session_state.messages_search.append({"role": "assistant", "content": ai_message})
            log_history_event(
                st.session_state, 
                "assistant_message", 
                {"page": "search", "content": ai_message}
            )
            # Save search tracking info for "Load More"
            st.session_state.search_query = user_input
            st.session_state.search_offset = 10  # initial load of 10
            if st.session_state.datasets_info is not None:
                st.session_state.search_total = st.session_state.datasets_info.attrs.get('total', 0)
            set_show_dataset(st.session_state, False)
            st.rerun()

    # --- Load More button ---
    if st.session_state.datasets_info is not None:
        total_hits = st.session_state.get("search_total", st.session_state.datasets_info.attrs.get('total', 0))
        current_count = st.session_state.datasets_info.shape[0]
        if current_count < total_hits:
            if st.button("Load More Datasets", key="load_more_button"):
                offset = st.session_state.get("search_offset", current_count)
                query = st.session_state.get("search_query", st.session_state.get("selected_text", ""))
                from src.search.search_pg_default import pg_search_default
                new_df = pg_search_default(query, count=10, from_idx=offset)
                if new_df.empty:
                    st.info("No more datasets available.")
                else:
                    st.session_state.datasets_info = pd.concat(
                        [st.session_state.datasets_info, new_df], ignore_index=True
                    )
                    st.session_state.search_offset = offset + new_df.shape[0]
                    st.rerun()

    # --- Send Selected Datasets button ---
    button_placeholder = st.empty()
    if len(st.session_state.selected_datasets) > 0:
        with button_placeholder:
            if st.button('Send Selected Datasets to Data Agent', key='send_datasets_button'):
                load_selected_datasets_into_cache(st.session_state.selected_datasets, st.session_state)
                set_active_datasets_from_selection(st.session_state)
                set_current_page(st.session_state, "data_agent")
                st.rerun()
        st.markdown("""
            <style>
            div[data-testid="stButton"] > button {
                position: fixed !important;
                bottom: 80px !important;
                right: 20px !important;
                z-index: 9999 !important;
                background-color: rgba(125, 209, 231, 0.8);
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px 24px;
                cursor: pointer;
                font-size: 14px;
                transition: all 0.3s ease;
                box-shadow: 0 0 15px rgba(125, 209, 231, 0.3);
                animation: subtle-glow 2s infinite;
            }
            div[data-testid="stButton"] > button:hover {
                transform: translateY(-2px);
                background-color: rgba(125, 209, 231, 1);
                box-shadow: 0 0 20px rgba(125, 209, 231, 0.5);
            }
            @keyframes subtle-glow {
                0% { box-shadow: 0 0 15px rgba(125, 209, 231, 0.3); }
                50% { box-shadow: 0 0 20px rgba(125, 209, 231, 0.5); }
                100% { box-shadow: 0 0 15px rgba(125, 209, 231, 0.3); }
            }
            </style>
        """, unsafe_allow_html=True)
    else:
        button_placeholder.empty()

    # Display active datasets (if any) and allow downloads or sending one dataset to Data Agent
    if st.session_state.active_datasets:
        for doi in st.session_state.active_datasets:
            dataset, name = st.session_state.datasets_cache.get(doi, (None, None))
            if dataset is not None:
                csv_data = convert_dataset_to_csv(dataset)
                with st.expander(f"Current Dataset: {doi}", expanded=st.session_state.show_dataset):
                    if isinstance(dataset, pd.DataFrame):
                        st.dataframe(dataset)
                    elif isinstance(dataset, xr.Dataset):
                        st.write(dataset)
                    else:
                        st.write("Unsupported data type")
                    col1, col2, col3 = st.columns([1, 3, 1])
                    with col1:
                        if csv_data:
                            st.download_button(
                                label="Download data as CSV",
                                data=csv_data,
                                file_name=get_dataset_csv_name(doi),
                                mime='text/csv',
                                key=f"download-{doi.split('/')[-1]}",
                                use_container_width=True
                            )
                        else:
                            st.write("Download not available for this dataset type.")
                    with col3:
                        if st.button(f"Send {doi} to Data Agent", use_container_width=True):
                            set_dataset_for_data_agent(st.session_state, doi, csv_data, dataset, name)
    message_placeholder = st.empty()

# -------------------------
# DATA AGENT PAGE
# -------------------------
elif st.session_state.current_page == "data_agent":
    st.markdown("## Data Agent")
    message_placeholder = st.empty()
    # Add processing status display container
    thinking_container = st.container()
    thinking_status = st.empty()

    # Initialize thinking log in session state if not exists
    if "thinking_log" not in st.session_state:
        st.session_state.thinking_log = []
        
    with st.sidebar:
        if "thinking_log" in st.session_state and st.session_state.thinking_log:
            with st.expander("🕒 Agent Performance Stats"):
                # Create stats dictionary to hold all counters
                stats = {
                    "total_steps": 0,
                    "agent_times": {},
                    "tool_counts": {}
                }
                
                # Process the thinking log to extract stats
                def process_log_stats(entries, stats_dict):
                    for entry in entries:
                        stats_dict["total_steps"] += 1
                        
                        # Track agent execution times
                        if entry.get("agent_name") and entry.get("duration"):
                            agent = entry["agent_name"]
                            if agent not in stats_dict["agent_times"]:
                                stats_dict["agent_times"][agent] = []
                            stats_dict["agent_times"][agent].append(entry["duration"])
                        
                        # Track tool usage
                        if entry.get("tool_name"):
                            tool = entry["tool_name"]
                            stats_dict["tool_counts"][tool] = stats_dict["tool_counts"].get(tool, 0) + 1
                        
                        # Process children recursively
                        if entry.get("children"):
                            process_log_stats(entry["children"], stats_dict)
                
                # Call the function with our stats dictionary
                process_log_stats(st.session_state.thinking_log, stats)
                
                # Display statistics
                st.write(f"**Total Steps:** {stats['total_steps']}")
                
                st.write("**Agent Execution Times:**")
                for agent, times in stats["agent_times"].items():
                    avg_time = sum(times) / len(times) if times else 0
                    st.write(f"- {agent}: {avg_time:.2f}s avg")
                
                st.write("**Tool Usage:**")
                for tool, count in sorted(stats["tool_counts"].items(), key=lambda x: x[1], reverse=True):
                    st.write(f"- {tool}: {count} calls")
        
    logging.info("Entered Data Agent page")

    # Save history button
    if st.button("Export Session History to JSON"):
        history_data = st.session_state.get("execution_history", [])
        with open("pangaea_session_history.json", "w", encoding="utf-8") as f:
            json.dump(history_data, f, indent=4, ensure_ascii=False)
        st.success("Exported session history to pangaea_session_history.json")
        logging.info("Exported session history to JSON")

    # Clear History button
    if st.button("Clear History", key="clear_history_data_agent"):
        st.session_state.messages_data_agent = []
        st.session_state.intermediate_steps = []
        logging.info("Cleared Data Agent history")
        st.rerun()

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 4, 1])
    with col3:
        if st.button("Back to Search", use_container_width=True, type="secondary"):
            set_current_page(st.session_state, "search")
            logging.info("Navigated back to Search page")
            st.rerun()
    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    datasets_info = get_datasets_info_for_active_datasets(st.session_state)
    logging.info(f"Loaded {len(datasets_info)} datasets for Data Agent")
    ensure_memory(st.session_state)
    ensure_thread_id(st.session_state)
    user_input = st.chat_input("Enter your query:")
    if user_input:
        # Store status container reference in session state
        st.session_state.thinking_status = thinking_status
        
        # Clear previous thinking log
        st.session_state.thinking_log = []
        st.session_state.processing = True
        
        # Display the thinking log expander
        with thinking_status:
            with st.expander("**🧠 Agent Thinking Process**", expanded=True):
                st.markdown("_Initializing agent processing..._")
        
        st.session_state.messages_data_agent.append({"role": "user", "content": f"{user_input}"})
        logging.info(f"User input in Data Agent: {user_input}")
        log_history_event(
            st.session_state, 
            "user_message", 
            {"page": "data_agent", "content": user_input}
        )
        user_query = user_input
        
        # Show "processing" message
        with message_placeholder:
            st.info("Your request is being processed. You can see the progress below in the Agent Thinking Process panel.")
        
        response = create_and_invoke_supervisor_agent(user_query, datasets_info, st.session_state["memory"],
                                                     st.session_state)
        
        if response:
            try:
                new_content = response['messages'][-1].content
                plot_images = response.get("plot_images", [])
                visualization_used = response.get("visualization_agent_used", False)
                st.session_state.messages_data_agent.append({
                    "role": "assistant",
                    "content": new_content,
                    "plot_images": plot_images,
                    "visualization_agent_used": visualization_used
                })
                log_history_event(
                    st.session_state, 
                    "assistant_message", 
                    {"page": "data_agent", "content": new_content}
                )
                logging.info(f"Processed assistant response in Data Agent, content: {new_content}")
                message_placeholder.empty()
                st.rerun()
            except Exception as e:
                logging.error(f"Error invoking graph: {e}")
                st.error(f"An error occurred: {e}")
            message_placeholder.empty()
        for message in st.session_state.messages_data_agent:
            logging.info(f"Displaying message in Data Agent: {message['role']}")
            if message["role"] == "system":
                with st.chat_message(message["role"], avatar=SYSTEM_ICON):
                    st.markdown(message["content"])
            elif message["role"] == "user":
                with st.chat_message(message["role"], avatar=USER_ICON):
                    st.markdown(message["content"])
            else:
                with st.chat_message(message["role"], avatar=SYSTEM_ICON):
                    st.markdown(message["content"])
                    if "plot_images" in message:
                        for plot_info in message["plot_images"]:
                            if isinstance(plot_info, tuple):
                                plot_path, code_path = plot_info
                                if os.path.exists(plot_path):
                                    col1, col2, col3 = st.columns([1, 2, 1])
                                    with col2:
                                        st.image(plot_path, caption='Generated Plot', use_container_width=True)
                                    if os.path.exists(code_path):
                                        with open(code_path, 'r') as f:
                                            code = f.read()
                                        st.code(code, language='python')
                            else:
                                if os.path.exists(plot_info):
                                    col1, col2, col3 = st.columns([1, 2, 1])
                                    with col2:
                                        st.image(plot_info, caption='Generated Plot', use_container_width=True)
                st.session_state.visualization_agent_used = False
        for info in datasets_info:
            logging.info(f"Displaying dataset in Data Agent: DOI {info['doi']}, type: {info['data_type']}")
            with st.expander(f"Dataset: {info['name']}", expanded=False):
                # Add clickable DOI link
                st.markdown(f"**DOI:** [{info['doi']}]({info['doi']})")
                st.write(f"**Name:** {info['name']}")
                st.write(f"**Type:** {info['data_type']}")
                if info['data_type'] == "pandas DataFrame":
                    st.dataframe(info['dataset'])
                    logging.info(f"Displayed DataFrame for DOI {info['doi']}")
                elif info['data_type'] == "xarray Dataset":
                    st.write(info['dataset'])  # Display string representation
                    st.write("**Variables:**")
                    st.write(list(info['dataset'].data_vars))
                    st.write("**Attributes:**")
                    st.write(info['dataset'].attrs)
                    logging.info(f"Displayed xarray Dataset for DOI {info['doi']}")
                elif info['data_type'] == "file folder":
                    st.write(f"Files in folder: {info['df_head']}")
                    logging.info(f"Displayed file folder for DOI {info['doi']}")
                else:
                    st.write("Unsupported data type")
                    logging.warning(f"Unsupported data type displayed for DOI {info['doi']}")
        if has_new_plot(st.session_state):
            reset_new_plot_flag(st.session_state)
    else:
        for message in st.session_state.messages_data_agent:
            logging.info(f"Displaying message in Data Agent: {message['role']}")
            if message["role"] == "system":
                with st.chat_message(message["role"], avatar=SYSTEM_ICON):
                    st.markdown(message["content"])
            elif message["role"] == "user":
                with st.chat_message(message["role"], avatar=USER_ICON):
                    st.markdown(message["content"])
            else:
                with st.chat_message(message["role"], avatar=SYSTEM_ICON):
                    st.markdown(message["content"])
                    if "plot_images" in message:
                        for plot_info in message["plot_images"]:
                            if isinstance(plot_info, tuple):
                                plot_path, code_path = plot_info
                                if os.path.exists(plot_path):
                                    col1, col2, col3 = st.columns([1, 2, 1])
                                    with col2:
                                        st.image(plot_path, caption='Generated Plot', use_container_width=True)
                                    if os.path.exists(code_path):
                                        with open(code_path, 'r') as f:
                                            code = f.read()
                                        st.code(code, language='python')
                            else:
                                if os.path.exists(plot_info):
                                    col1, col2, col3 = st.columns([1, 2, 1])
                                    with col2:
                                        st.image(plot_info, caption='Generated Plot', use_container_width=True)
                st.session_state.visualization_agent_used = False
        for info in datasets_info:
            logging.info(f"Displaying dataset in Data Agent: DOI {info['doi']}, type: {info['data_type']}")
            with st.expander(f"Dataset: {info['name']}", expanded=False):
                # Add clickable DOI link
                st.markdown(f"**DOI:** [{info['doi']}]({info['doi']})")
                st.write(f"**Name:** {info['name']}")
                st.write(f"**Type:** {info['data_type']}")
                if info['data_type'] == "pandas DataFrame":
                    st.dataframe(info['dataset'])
                    logging.info(f"Displayed DataFrame for DOI {info['doi']}")
                elif info['data_type'] == "xarray Dataset":
                    st.write(info['dataset'])  # Display string representation
                    st.write("**Variables:**")
                    st.write(list(info['dataset'].data_vars))
                    st.write("**Attributes:**")
                    st.write(info['dataset'].attrs)
                    logging.info(f"Displayed xarray Dataset for DOI {info['doi']}")
                elif info['data_type'] == "file folder":
                    st.write(f"Files in folder: {info['df_head']}")
                    logging.info(f"Displayed file folder for DOI {info['doi']}")
                else:
                    st.write("Unsupported data type")
                    logging.warning(f"Unsupported data type displayed for DOI {info['doi']}")
        if has_new_plot(st.session_state):
            reset_new_plot_flag(st.session_state)
else:
    if len(st.session_state.active_datasets) == 0 and st.session_state.current_page == "data_agent":
        st.warning("No datasets loaded. Please load a dataset first.")
        logging.warning("No datasets loaded in Data Agent")
# ui_main.py

import streamlit as st
import logging
from io import StringIO
import pandas as pd
import os
import json

from src.config import API_KEY, LANGCHAIN_API_KEY, LANGCHAIN_PROJECT_NAME
from src.ui.styles import CUSTOM_UI, SYSTEM_ICON, USER_ICON
from src.utils import log_history_event
from langchain_core.messages import HumanMessage, AIMessage
from src.memory import CustomMemorySaver

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

# Initialize session state
initialize_session_state(st.session_state)

from langchain_community.chat_message_histories import ChatMessageHistory

# Streamlit page config and UI setup
st.set_page_config(
    page_title="PANGAEA GPT",
    page_icon="img/pangaea-logo.png",
    layout="wide"
)

st.markdown(CUSTOM_UI, unsafe_allow_html=True)

# Load secrets
openai_api_key = API_KEY
langchain_api_key = LANGCHAIN_API_KEY
langchain_project_name = LANGCHAIN_PROJECT_NAME

# Sidebar UI
with st.sidebar:
    st.markdown("""
        <style>
        [data-testid="stImage"] {
            padding: 0 !important;
            margin: 0 !important;
            width: 100% !important;
            display: block !important;
        }
        [data-testid="stImage"] > img {
            margin: 0 auto !important;
            display: block !important;
            max-width: 60px !important; 
        }
        .element-container {
            padding: 0 !important;
            margin: 0 !important;
        }
        </style>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 3, 2])
    with col2:
        st.image("img/pangaea-logo.png", width=100)
    st.title("Configuration")
    model_name = st.selectbox("Select Model", ["gpt-4o", "gpt-4o-mini", "o1-mini", "o3-mini"], index=0,
                              key="model_name")


if langchain_api_key and langchain_project_name:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = langchain_project_name
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key

# Create the search agent now that model_name is set
search_agent = get_search_agent(
    datasets_info=st.session_state.datasets_info,
    model_name=st.session_state["model_name"],
    api_key=openai_api_key
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
        "Search for prokaryote abundance data on Hakon Mosby volcano"
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
                    st.experimental_rerun()

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
                position: fixed;
                bottom: 80px;
                right: 20px;
                z-index: 9999;
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
                    st.dataframe(dataset)
                    col1, col2, col3 = st.columns([1, 3, 1])
                    with col1:
                        st.download_button(
                            label="Download data as CSV",
                            data=csv_data,
                            file_name=get_dataset_csv_name(doi),
                            mime='text/csv',
                            key=f"download-{doi.split('/')[-1]}",
                            use_container_width=True
                        )
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
    # Save history button
    if st.button("Export Session History to JSON"):
        history_data = st.session_state.get("execution_history", [])
        with open("pangaea_session_history.json", "w", encoding="utf-8") as f:
            json.dump(history_data, f, indent=4, ensure_ascii=False)
        st.success("Exported session history to pangaea_session_history.json")

    # NEW: Clear History button added here on the Data Agent page
    if st.button("Clear History", key="clear_history_data_agent"):
        st.session_state.messages_data_agent = []
        st.session_state.intermediate_steps = []
        st.rerun()

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 4, 1])
    with col3:
        if st.button("Back to Search", use_container_width=True, type="secondary"):
            set_current_page(st.session_state, "search")
            st.rerun()
    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    datasets_info = get_datasets_info_for_active_datasets(st.session_state)
    ensure_memory(st.session_state)
    ensure_thread_id(st.session_state)
    user_input = st.chat_input("Enter your query:")
    if user_input:
        st.session_state.messages_data_agent.append({"role": "user", "content": f"{user_input}"})
        logging.debug("User input: %s", user_input)
        log_history_event(
            st.session_state, 
            "user_message", 
            {"page": "data_agent", "content": user_input}
        )
        user_query = user_input
        response = create_and_invoke_supervisor_agent(user_query, datasets_info, st.session_state["memory"],
                                                      st.session_state)
        if response:
            with message_placeholder:
                st.info("Work in progress...")
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
                message_placeholder.empty()
                st.rerun()
            except Exception as e:
                logging.error(f"Error invoking graph: {e}")
                st.error(f"An error occurred: {e}")
            message_placeholder.empty()
        for message in st.session_state.messages_data_agent:
            logging.info(f"Displaying message: {message['role']}")
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
            with st.expander(f"Dataset: {info['doi']}", expanded=False):
                st.write(f"**Name:** {info['name']}")
                st.dataframe(info['dataset'])
        if has_new_plot(st.session_state):
            reset_new_plot_flag(st.session_state)
    else:
        for message in st.session_state.messages_data_agent:
            logging.info(f"Displaying message: {message['role']}")
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
            with st.expander(f"Dataset: {info['doi']}", expanded=False):
                st.write(f"**Name:** {info['name']}")
                st.dataframe(info['dataset'])
        if has_new_plot(st.session_state):
            reset_new_plot_flag(st.session_state)
else:
    if len(st.session_state.active_datasets) == 0 and st.session_state.current_page == "data_agent":
        st.warning("No datasets loaded. Please load a dataset first.")

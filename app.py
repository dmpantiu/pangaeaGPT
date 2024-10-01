# app.py

# 1. Imports
import streamlit as st
import pandas as pd
import warnings
import logging
from datetime import datetime
import shutil
from langchain.schema import HumanMessage
from io import StringIO
from src.search.dataset_utils import fetch_dataset, convert_df_to_csv
import os
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from config import PLOT_OUTPUT_DIR  # Only import what's necessary

# Import agent creation functions
from src.agents import create_supervisor_agent, create_search_agent

# 1. Streamlit app setup
st.set_page_config(
    page_title="PANGAEA GPT",
    page_icon="img/pangaea-logo.png",
    layout="wide"
)

# Main title and styles (omitted for brevity)
st.markdown(
    """
    <style>
        h1 {
            text-align: center; 
            font-weight: bold; 
            font-family: 'Open Sans', 'Helvetica Neue', sans-serif; 
            padding-top: 1rem;
            padding-bottom: 1rem;
            font-size: 31.5px;
        }
        h2 {
            text-align: center; 
            font-weight: bold; 
            font-family: 'Open Sans', 'Helvetica Neue', sans-serif; 
            padding-top: 0.5rem;
            font-size: 24px;
        }
        [data-testid=stSidebar] [data-testid=stImage] {
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: auto;
            max-width: 99px;
        }
        /* Hide fullscreen button */
        button[title="View fullscreen"] {
            display: none;
        }
        /* Adjust sidebar width */
        [data-testid=stSidebar] {
            width: 17%;
        }
    </style>
    """, unsafe_allow_html=True
)

# Load the logo
st.sidebar.image("img/pangaea-logo.png", use_column_width=False)

# Main title
st.markdown("# PANGAEA GPT")
SYSTEM_ICON = "img/11111111.png"
USER_ICON = "img/2222222.png"

# 2. Initial Setup and Configuration

# Initialize session state for messages and datasets
session_state_defaults = {
    "messages_search": [],
    "messages_data_agent": [],
    "datasets_cache": {},
    "datasets_info": None,
    "active_dataset": None,
    "show_dataset": True,
    "current_page": "search",
    "dataset_csv": None,
    "dataset_df": None,
    "dataset_name": None,
    "saved_plot_path": None,
    "visualization_agent_used": False,

}

for key, value in session_state_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory(session_id="search-agent-session")
if "datasets_info" not in st.session_state:
    st.session_state.datasets_info = None

# Sidebar inputs for model selection and search method
with st.sidebar:
    st.title("Configuration")
    # Input fields for API keys
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key",
        key="openai_api_key"
    )

    langchain_api_key = st.text_input(
        "LangSmith API Key (optional)",
        type="password",
        help="Enter your LangSmith API key",
        key="langchain_api_key"
    )

    langchain_project_name = st.text_input(
        "LangSmith Project Name (optional)",
        help="Enter your LangSmith project name",
        key="langchain_project_name"
    )

    # Require OpenAI API key before proceeding
    if not openai_api_key:
        st.warning("Please enter your OpenAI API key to proceed.")
        st.stop()

    model_name = st.selectbox(
        "Select Model",
        ["gpt-3.5-turbo", "gpt-4o", 'gpt-4o-mini', 'o1-mini'],
        index=0,
        key="model_name"
    )
    search_method = st.selectbox(
        "Select Search Method",
        ["PANGAEA Search (default)", "Elastic Search"],
        index=0,
        key="search_method"
    )

# Use the API keys from the sidebar inputs
os.environ["OPENAI_API_KEY"] = openai_api_key
api_key = openai_api_key
selected_model_name = st.session_state.get("model_name", "gpt-4o-mini")
#st.session_state["search_method"] = search_method

# LangSmith API Key for monitoring
if langchain_api_key and langchain_project_name:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = langchain_project_name
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key

# 3. Initialize the Search Agent
search_agent = create_search_agent(
    api_key=api_key,
    datasets_info=st.session_state.datasets_info
)

# 9. Streamlit UI
# 9.1 Search Page
if st.session_state.current_page == "search":
    st.markdown("## Dataset Explorer")

    # Create placeholders for dynamic content
    chat_placeholder = st.container()
    message_placeholder = st.empty()

    user_input = st.chat_input("Enter your query:")

    if user_input:
        st.session_state.messages_search.append({"role": "user", "content": user_input})
        logging.debug("User input: %s", user_input)

        try:
            st.session_state.chat_history.add_user_message(user_input)

            search_agent_with_memory = RunnableWithMessageHistory(
                search_agent,
                lambda session_id: st.session_state.chat_history,
                input_messages_key="input",
                history_messages_key="chat_history",
            )

            with message_placeholder:
                st.info("Work in progress...")

            response = search_agent_with_memory.invoke(
                {"input": user_input, "chat_history": st.session_state.chat_history.messages},
                {"configurable": {"session_id": "search-agent-session"}}
            )
            logging.debug("Response from search agent: %s", response)

            ai_message = response["output"]
            st.session_state.chat_history.add_ai_message(ai_message)
            st.session_state.messages_search.append({"role": "assistant", "content": ai_message})
            logging.debug("Assistant message: %s", ai_message)

            message_placeholder.empty()

        except Exception as e:
            logging.error(f"Error invoking search agent: {e}")
            with message_placeholder:
                st.error(f"An error occurred: {e}")

        st.session_state.show_dataset = False

    with chat_placeholder:
        for i, message in enumerate(st.session_state.messages_search):
            if message["role"] == "system":
                with st.chat_message(message["role"], avatar=SYSTEM_ICON):
                    st.markdown(message["content"])
            elif message["role"] == "user":
                with st.chat_message(message["role"], avatar=USER_ICON):
                    st.markdown(message["content"])
            else:  # assistant
                with st.chat_message(message["role"], avatar=SYSTEM_ICON):
                    st.markdown(message["content"])
            if "table" in message:
                df = pd.read_json(StringIO(message["table"]))
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
                    cols[5].write(f"**Score:** {row['Score']}")
                    if cols[5].button("Fetch", key=f"{i}-{index}"):
                        logging.debug("Fetch button clicked for %s", row['DOI'])
                        dataset, name = fetch_dataset(row['DOI'])
                        if dataset is not None:
                            logging.debug("Dataset %s fetched successfully.", row['DOI'])
                            st.session_state.datasets_cache[row['DOI']] = (dataset, name)
                            st.session_state.active_dataset = row['DOI']
                            st.session_state.dataset_name = name
                            st.session_state.show_dataset = True
                            st.rerun()

    if st.session_state.active_dataset:
        doi = st.session_state.active_dataset
        dataset, name = st.session_state.datasets_cache[doi]
        csv_data = convert_df_to_csv(dataset)

        with st.expander(f"Current Dataset: {doi}", expanded=st.session_state.show_dataset):
            st.dataframe(dataset)

            col1, col2, col3 = st.columns([1, 3, 1])
            with col1:
                st.download_button(
                    label="Download data as CSV",
                    data=csv_data,
                    file_name=f"dataset.csv",
                    mime='text/csv',
                    key=f"download-{doi.split('/')[-1]}",
                    use_container_width=True
                )
            with col3:
                if st.button("Send to Data Agent", use_container_width=True):
                    st.session_state.dataset_csv = csv_data
                    st.session_state.dataset_df = dataset
                    st.session_state.dataset_name = st.session_state.datasets_info.loc[
                        st.session_state.datasets_info['DOI'] == doi, 'Name'].values[0]
                    st.session_state.current_page = "data_agent"
                    st.rerun()

        message_placeholder = st.empty()

# 9.2 Data Agent Page
if st.session_state.current_page == "data_agent":
    st.markdown("## Data Agent")

    # Create a placeholder for the 'Work in progress...' message
    message_placeholder = st.empty()

    # Create columns for layout, with a button to go back to the search page
    col1, col2 = st.columns([8, 1])
    with col2:
        if st.button("Back to Search", use_container_width=True):
            st.session_state.current_page = "search"
            st.rerun()

    # Function to check if a new plot has been generated
    def has_new_plot():
        if 'new_plot_generated' not in st.session_state:
            st.session_state.new_plot_generated = False
        return st.session_state.new_plot_generated

    # Check if the dataset is loaded
    if st.session_state.dataset_df is not None:
        # Extract necessary information from the session state
        user_query = st.session_state.messages_data_agent[-1]["content"] if st.session_state.messages_data_agent else ""
        dataset_name = st.session_state.dataset_name
        dataset_description = st.session_state.datasets_info.loc[
            st.session_state.datasets_info['DOI'] == st.session_state.active_dataset, 'Description'
        ].values[0]
        df_head = st.session_state.dataset_df.head().to_string()

        # Create the supervisor agent with the extracted information
        graph = create_supervisor_agent(api_key, user_query, dataset_name, dataset_description, df_head)

        # Get user input for the Data Agent
        user_input = st.chat_input("Enter your query:")

        if user_input:
            # Append the user input to the session state messages
            st.session_state.messages_data_agent.append({"role": "user", "content": f"{user_input}"})
            logging.debug("User input: %s", user_input)

            if graph:
                # Show 'Work in progress...' message
                with message_placeholder:
                    st.info("Work in progress...")

                # Initialize the state for the graph invocation
                initial_state = {
                    "messages": [HumanMessage(content=user_input)],
                    "next": "supervisor",
                    "agent_scratchpad": [],
                    "input": user_input,
                    "plot_images": [],
                    "last_agent_message": ""
                }
                logging.debug(f"Initial state before invoking graph: {initial_state}")
                try:
                    # Invoke the graph and get the response
                    response = graph.invoke(initial_state)
                    logging.info(f"Agent response: {response}")

                    # Create a new message with the response content and any generated plots
                    new_message = {
                        "role": "assistant",
                        "content": response['messages'][-1].content,
                        "plot_images": response.get("plot_images", []),
                        "visualization_agent_used": response.get("visualization_agent_used", False)
                    }
                    logging.info(f"New message created: {new_message}")
                    st.session_state.messages_data_agent.append(new_message)
                    logging.info("Added new message to messages_data_agent")

                except Exception as e:
                    logging.error(f"Error invoking graph: {e}")

                # Clear the 'Work in progress...' message
                message_placeholder.empty()

        for message in st.session_state.messages_data_agent:
            logging.info(f"Displaying message: {message['role']}")
            if message["role"] == "system":
                with st.chat_message(message["role"], avatar=SYSTEM_ICON):
                    st.markdown(message["content"])
            elif message["role"] == "user":
                with st.chat_message(message["role"], avatar=USER_ICON):
                    st.markdown(message["content"])
            else:  # assistant
                with st.chat_message(message["role"], avatar=SYSTEM_ICON):
                    st.markdown(message["content"])
                    logging.info(f"Checking for plot_images in message: {message.get('plot_images', [])}")
                    # Example in the Data Agent Page
                    if "plot_images" in message:
                        for plot_info in message["plot_images"]:
                            if isinstance(plot_info, tuple):
                                plot_path, code_path = plot_info
                                if os.path.exists(plot_path):
                                    col1, col2, col3 = st.columns([1, 2, 1])
                                    with col2:
                                        st.image(plot_path, caption='Generated Plot', use_column_width=True)
                                    if os.path.exists(code_path):
                                        with open(code_path, 'r') as f:
                                            code = f.read()
                                        st.code(code, language='python')
                            else:
                                # Handle the case where plot_info is just a path (for backwards compatibility)
                                if os.path.exists(plot_info):
                                    col1, col2, col3 = st.columns([1, 2, 1])
                                    with col2:
                                        st.image(plot_info, caption='Generated Plot', use_column_width=True)

            # Reset the flag after displaying the message
            st.session_state.visualization_agent_used = False

        # Display dataset info
        with st.expander("Dataset Info", expanded=False):
            st.write(dataset_name)
            st.dataframe(st.session_state.dataset_df)

        # Check for new plot and update the UI if necessary
        if has_new_plot():
            st.session_state.new_plot_generated = False  # Reset the flag
            st.rerun()
    else:
        # Warning if no dataset is loaded
        st.warning("No dataset loaded. Please load a dataset first.")

# Add this at the end of the script
logging.debug("Script execution completed")
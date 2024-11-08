# app.py
# Part 1: Imports and Initial Setup
import streamlit as st 
import pandas as pd 
import uuid
import logging
from langchain_core.messages import HumanMessage, AIMessage 
from io import StringIO
from src.search.dataset_utils import fetch_dataset, convert_df_to_csv
import os
from langchain_community.chat_message_histories import ChatMessageHistory 
from langgraph.checkpoint.memory import MemorySaver 
from langchain_core.runnables.history import RunnableWithMessageHistory 
from src.config import API_KEY, LANGCHAIN_API_KEY, LANGCHAIN_PROJECT_NAME
from src.memory import CustomMemorySaver
from src.agents import create_supervisor_agent, create_search_agent
from src.ui.styles import CUSTOM_UI, SYSTEM_ICON, USER_ICON


# Streamlit page config
st.set_page_config(
    page_title="PANGAEA GPT",
    page_icon="img/pangaea-logo.png",
    layout="wide"
)

# Apply custom UI
st.markdown(CUSTOM_UI, unsafe_allow_html=True)


# 2. Initial Setup and Configuration

# Initialize session state for messages and datasets
# 2. Initial Setup and Configuration

# Initialize session state with default values, including 'datasets_info'
session_state_defaults = {
    "messages_search": [],
    "messages_data_agent": [],
    "datasets_cache": {},
    "datasets_info": None,  # Ensure 'datasets_info' is included
    "active_datasets": [],
    "selected_datasets": set(),
    "show_dataset": True,
    "current_page": "search",
    "dataset_dfs": {},
    "dataset_names": {},
    "saved_plot_paths": {},
    "memory": MemorySaver(),
    "visualization_agent_used": False,
    "chat_history": ChatMessageHistory(session_id="search-agent-session"),
}

# Update session state with defaults if keys are missing
for key, value in session_state_defaults.items():
    st.session_state.setdefault(key, value)

# Load secrets from the `secrets.toml` file
openai_api_key = API_KEY
langchain_api_key = LANGCHAIN_API_KEY
langchain_project_name = LANGCHAIN_PROJECT_NAME


# Sidebar inputs for model selection and search method
with st.sidebar:
    # Update CSS to be more restrictive and remove all extra spacing
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
            max-width: 60px !important;  /* Even smaller width control */
        }

        /* Remove any potential container padding */
        .element-container {
            padding: 0 !important;
            margin: 0 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Adjust column ratios for tighter control
    col1, col2, col3 = st.columns([2, 3, 2])
    with col2:
        st.image("img/pangaea-logo.png", width=100)  # Further reduced width from 80 to 60

    st.title("Configuration")
    model_name = st.selectbox("Select Model", ["gpt-3.5-turbo", "gpt-4o", 'gpt-4o-mini', 'o1-mini'], index=0,
                              key="model_name")
    # Remove search method selection
    st.session_state["search_method"] = "PANGAEA Search (default)"  # Set default search method

# Use the loaded secrets in your application
#api_key = openai_api_key
selected_model_name = st.session_state.get("model_name", "gpt-4o-mini")

# LangSmith API Key for monitoring
if langchain_api_key and langchain_project_name:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = langchain_project_name
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key

# 3. Initialize the Search Agent
from src.agents import create_search_agent



def get_truncated_chat_history(session_id):
    truncated_messages = st.session_state.chat_history.messages[-10:]
    truncated_history = ChatMessageHistory(session_id=session_id)
    # Add truncated messages to the new chat history
    for message in truncated_messages:
        if isinstance(message, HumanMessage):
            truncated_history.add_user_message(message.content)
        elif isinstance(message, AIMessage):
            truncated_history.add_ai_message(message.content)
        else:
            truncated_history.add_message(message)
    return truncated_history

search_agent = create_search_agent(datasets_info=st.session_state.datasets_info)

# 9.1 Search Page
if st.session_state.current_page == "search":
    st.markdown("## Dataset Explorer")

    # Create placeholders for dynamic content
    chat_placeholder = st.container()
    message_placeholder = st.empty()

    # Create a container for the fixed chat input at the bottom
    chat_input_container = st.container()

    # Add selectbox for predefined queries
    predefined_queries = [
        "Search for data on gelatinous zooplankton in the Fram Strait?",
        "Continuous records of the atmospheric greenhouse gases",
        "Find datasets about renewable energy sources.",
    ]
    selected_query = st.selectbox(
        "Select an example or write down your query:",
        [""] + predefined_queries,
        index=0,
        key="selected_query",
    )

    # If a predefined query is selected, set it as the default text in the input box
    if selected_query != "":
        st.session_state.selected_text = selected_query
    else:
        st.session_state.selected_text = st.session_state.get("selected_text", "")

    # Display chat messages and dataset selection interface
    with chat_placeholder:
        # Display existing messages
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

                    # Add checkbox for selection
                    with cols[5]:
                        checkbox_key = f"select-{i}-{index}"
                        if checkbox_key not in st.session_state:
                            st.session_state[checkbox_key] = False
                        selected = st.checkbox("Select", key=checkbox_key)
                        if selected:
                            st.session_state.selected_datasets.add(row['DOI'])
                        else:
                            st.session_state.selected_datasets.discard(row['DOI'])

    # Create a placeholder for the fixed-position button
    button_placeholder = st.empty()

    # Only display the button when datasets are selected
    if len(st.session_state.selected_datasets) > 0:
        with button_placeholder:
            if st.button('Send Selected Datasets to Data Agent', key='send_datasets_button'):
                for doi in st.session_state.selected_datasets:
                    if doi not in st.session_state.datasets_cache:
                        dataset, name = fetch_dataset(doi)
                        if dataset is not None:
                            st.session_state.datasets_cache[doi] = (dataset, name)
                            st.session_state.dataset_dfs[doi] = dataset
                            st.session_state.dataset_names[doi] = name
                st.session_state.active_datasets = list(st.session_state.selected_datasets)
                st.session_state.current_page = "data_agent"
                st.rerun()

        # Updated button styling with your color scheme
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

    # Chat input at the bottom
    with chat_input_container:
        # Add some spacing before the chat input
        st.markdown("<br>" * 2, unsafe_allow_html=True)

        # Style just the button's colors and hover state with inverted color scheme
        st.markdown("""
            <style>
            .stFormSubmitButton > button {
                background-color: rgba(67, 163, 151, 0.6) !important;  /* Primary teal/green color */
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


            # Reconstruct the chat history from messages_search
            st.session_state.chat_history = ChatMessageHistory(session_id="search-agent-session")
            for message in st.session_state.messages_search:
                if message["role"] == "user":
                    st.session_state.chat_history.add_user_message(message["content"])
                elif message["role"] == "assistant":
                    st.session_state.chat_history.add_ai_message(message["content"])

            try:
                search_agent_with_memory = RunnableWithMessageHistory(
                    search_agent,
                    get_truncated_chat_history,
                    input_messages_key="input",
                    history_messages_key="chat_history",
                )

                with message_placeholder:
                    st.info("Work in progress...")

                response = search_agent_with_memory.invoke(
                    {"input": user_input},
                    {"configurable": {"session_id": "search-agent-session"}},
                )
                logging.debug("Response from search agent: %s", response)

                ai_message = response["output"]

                st.session_state.messages_search.append({"role": "assistant", "content": ai_message})
                logging.debug("Assistant message: %s", ai_message)

                message_placeholder.empty()

            except Exception as e:
                logging.error(f"Error invoking search agent: {e}")
                with message_placeholder:
                    st.error(f"An error occurred: {e}")

            st.session_state.show_dataset = False
            st.rerun()

    # Display active datasets if any
    if st.session_state.active_datasets:
        for doi in st.session_state.active_datasets:
            dataset, name = st.session_state.datasets_cache.get(doi, (None, None))
            if dataset is not None:
                csv_data = convert_df_to_csv(dataset)

                with st.expander(f"Current Dataset: {doi}", expanded=st.session_state.show_dataset):
                    st.dataframe(dataset)

                    col1, col2, col3 = st.columns([1, 3, 1])
                    with col1:
                        st.download_button(
                            label="Download data as CSV",
                            data=csv_data,
                            file_name=f"dataset_{doi.split('/')[-1]}.csv",
                            mime='text/csv',
                            key=f"download-{doi.split('/')[-1]}",
                            use_container_width=True
                        )
                    with col3:
                        if st.button(f"Send {doi} to Data Agent", use_container_width=True):
                            st.session_state.dataset_csv = csv_data
                            st.session_state.dataset_df = dataset
                            st.session_state.dataset_name = name
                            st.session_state.current_page = "data_agent"
                            st.rerun()

    message_placeholder = st.empty()

# 9.2 Data Agent Page
if st.session_state.current_page == "data_agent":
    st.markdown("## Data Agent")

    # Create a placeholder for the 'Work in progress...' message
    message_placeholder = st.empty()

    # Create columns for layout, with a button to go back to the search page

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)  # Top margin
    col1, col2, col3 = st.columns([1, 4, 1])
    with col3:
        if st.button("Back to Search", use_container_width=True, type="secondary"):
            st.session_state.current_page = "search"
            st.rerun()
    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)  # Bottom margin

    # Function to check if a new plot has been generated
    def has_new_plot():
        if 'new_plot_generated' not in st.session_state:
            st.session_state.new_plot_generated = False
        return st.session_state.new_plot_generated

    if st.session_state.active_datasets:
        # Extract datasets_info from active_datasets
        datasets_info = []
        for doi in st.session_state.active_datasets:
            dataset, name = st.session_state.datasets_cache.get(doi, (None, None))
            if dataset is not None:
                # Use the correct column name for the description
                description = st.session_state.datasets_info.loc[
                    st.session_state.datasets_info['DOI'] == doi, 'Short Description'
                ].values[0]
                df_head = dataset.head().to_string()
                datasets_info.append({
                    'doi': doi,
                    'name': name,
                    'description': description,
                    'df_head': df_head,
                    'dataset': dataset
                })

        # Ensure that the 'memory' exists in session_state
        if 'memory' not in st.session_state:
            st.session_state['memory'] = CustomMemorySaver()

        # Ensure consistent 'thread_id' across invocations
        if 'thread_id' not in st.session_state:
            st.session_state['thread_id'] = str(uuid.uuid4())

        # Get user input for the Data Agent
        user_input = st.chat_input("Enter your query:")

        if user_input:
            # Append the user input to the session state messages
            st.session_state.messages_data_agent.append({"role": "user", "content": f"{user_input}"})
            logging.debug("User input: %s", user_input)

            # Now update the user_query with the latest input
            user_query = user_input  # Or retrieve from messages_data_agent[-1]["content"]

            # Create the supervisor agent with the updated user_query
            graph = create_supervisor_agent(user_query, datasets_info, st.session_state['memory'])

            if graph:
                # Show 'Work in progress...' message
                with message_placeholder:
                    st.info("Work in progress...")

                # Use the full conversation history
                messages = []
                for message in st.session_state.messages_data_agent:
                    if message["role"] == "user":
                        messages.append(HumanMessage(content=message["content"], name="User"))
                    elif message["role"] == "assistant":
                        messages.append(AIMessage(content=message["content"], name="Assistant"))
                    else:
                        messages.append(AIMessage(content=message["content"], name=message["role"]))

                # Initialize the state for the graph invocation
                limited_messages = messages[-7:]  # Limit to last 5 messages

                initial_state = {
                    "messages": limited_messages,
                    "next": "supervisor",
                    "agent_scratchpad": [],
                    "input": user_input,
                    "plot_images": [],
                    "last_agent_message": ""
                }

                # Use the consistent thread_id from session_state
                config = {"configurable": {"thread_id": st.session_state['thread_id'], "recursion_limit": 5}}

                logging.debug(f"Initial state before invoking graph: {initial_state}")
                try:
                    # Invoke the graph and get the response
                    response = graph.invoke(initial_state, config=config)
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
                    st.error(f"An error occurred: {e}")

                # Clear the 'Work in progress...' message
                message_placeholder.empty()

            # Display the chat messages including the latest response
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
                                    # Handle the case where plot_info is just a path
                                    if os.path.exists(plot_info):
                                        col1, col2, col3 = st.columns([1, 2, 1])
                                        with col2:
                                            st.image(plot_info, caption='Generated Plot', use_column_width=True)

                # Reset the flag after displaying the message
                st.session_state.visualization_agent_used = False

            # Display datasets info
            for info in datasets_info:
                with st.expander(f"Dataset: {info['doi']}", expanded=False):
                    st.write(f"**Name:** {info['name']}")
                    st.dataframe(info['dataset'])

            # Check for new plot and update the UI if necessary
            if has_new_plot():
                st.session_state.new_plot_generated = False  # Reset the flag
                st.rerun()
        else:
            # If no user input, display previous messages
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
                                    # Handle the case where plot_info is just a path
                                    if os.path.exists(plot_info):
                                        col1, col2, col3 = st.columns([1, 2, 1])
                                        with col2:
                                            st.image(plot_info, caption='Generated Plot', use_column_width=True)

                # Reset the flag after displaying the message
                st.session_state.visualization_agent_used = False

            # Display datasets info
            for info in datasets_info:
                with st.expander(f"Dataset: {info['doi']}", expanded=False):
                    st.write(f"**Name:** {info['name']}")
                    st.dataframe(info['dataset'])

            # Check for new plot and update the UI if necessary
            if has_new_plot():
                st.session_state.new_plot_generated = False  # Reset the flag
                st.rerun()
    else:
        # Warning if no dataset is loaded
        st.warning("No datasets loaded. Please load a dataset first.")
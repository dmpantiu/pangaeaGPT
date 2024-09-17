# 1. Imports
import streamlit as st
import pandas as pd
import warnings
import re
import os
import logging
from datetime import datetime
import operator
import shutil
import uuid
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain_experimental.tools.python.tool import PythonREPLTool
from io import StringIO
import matplotlib.pyplot as plt
from typing import Any
import functools
from langgraph.graph import END
from typing import List, Annotated, Sequence, TypedDict
from plotting_tools.hard_agent import plot_sampling_stations, plot_master_track_map
from plotting_tools.oceanographer_tools import plot_ctd_profiles, plot_ts_diagram
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from search.search_pg_es import search_pg_es
from search.search_pg_default import pg_search_default
from search.dataset_utils import fetch_dataset, convert_df_to_csv
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser

#comment out later
#Graph visualization imports
from langgraph.graph import StateGraph
#/comment out later


#from langchain.memory import ChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

#QA tool imports
from search.publication_qa_tool import answer_publication_questions, PublicationQAArgs


#1. Streamlit app setup
# Streamlit app setup
st.set_page_config(
    page_title="PANGAEA GPT",
    page_icon="img/pangaea-logo.png",
    layout="wide"
)

# Main title with custom font, adjusted sidebar image CSS, and hidden fullscreen button
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
            width: auto;  /* Changed to auto */
            max-width: 99px;  /* Reduced maximum width */
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


# Set up logging
log_filename = f'solving_multiple_images_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(filename=log_filename, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


# 2. Initial Setup and Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore", category=FutureWarning, message="Downcasting behavior in `replace` is deprecated")
logging.getLogger('matplotlib').setLevel(logging.WARNING)


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
}

if "visualization_agent_used" not in st.session_state:
    st.session_state.visualization_agent_used = False

for key, value in session_state_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory(session_id="search-agent-session")
if "datasets_info" not in st.session_state:
    st.session_state.datasets_info = None



# Load secrets from the `secrets.toml` file
openai_api_key = st.secrets["general"]["openai_api_key"]
langchain_api_key = st.secrets["general"].get("langchain_api_key", None)
langchain_project_name = st.secrets["general"].get("langchain_project_name", None)

# Sidebar inputs for model selection and search method
with st.sidebar:
    st.title("Configuration")
    model_name = st.selectbox("Select Model", ["gpt-3.5-turbo", "gpt-4o"], key="model_name")
    search_method = st.selectbox("Select Search Method", ["PANGAEA Search (default)", "Elastic Search"], index=0, key="search_method")

# Use the loaded secrets in your application
api_key = openai_api_key
selected_model_name = st.session_state.get("model_name", "gpt-3.5-turbo")

# Log the selected model to verify it's being set correctly
logging.debug(f"Selected model name: {selected_model_name}")

# LangSmith API Key for monitoring
if langchain_api_key and langchain_project_name:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = langchain_project_name
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
    #os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
    #os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# 3. Utility Functions
#Generate unique image path
def generate_unique_image_path():
    figs_dir = os.path.join('plotting_tools', 'temp_files', 'figs')
    os.makedirs(figs_dir, exist_ok=True)
    unique_path = os.path.join(figs_dir, f'fig_{uuid.uuid4()}.png')
    logging.debug(f"Generated unique image path: {unique_path}")
    return unique_path

# Function to sanitize input
def sanitize_input(query: str) -> str:
    return query.strip()

# Define the function to extract the last Python REPL command
def get_last_python_repl_command():
    file_path = os.path.join('current_data', 'intermediate_steps_visualization_agent.txt')
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            python_repl_commands = []
            for line in lines:
                if "tool='Python_REPL'" in line:
                    python_repl_commands.append(line)
                elif python_repl_commands:
                    python_repl_commands[-1] += line

            if python_repl_commands:
                last_command = python_repl_commands[-1]
                code_match = re.search(r"tool_input=\"(.+?)\" log=", last_command, re.DOTALL)
                if code_match:
                    command = code_match.group(1).replace('\\n', '\n').replace('\\"', '"')
                    logging.debug(f"Extracted last Python REPL command: {command}")
                    return command
    except FileNotFoundError:
        logging.warning("No intermediate steps file found.")
    return None

#Visualization of nodes in the graph
#def save_graph_visualization(graph, output_folder, output_filename="graph.png"):
#    # Ensure the output directory exists
#    os.makedirs(output_folder, exist_ok=True)
#
#    png_output_path = os.path.join(output_folder, output_filename)
#    img_data = graph.get_graph().draw_mermaid_png(
#        curve_style=CurveStyle.LINEAR,
#        node_colors=NodeColors(start="#ffdfba", end="#baffc9", other="#fad7de"),
#        wrap_label_n_words=9,
#        output_file_path=None,
#        draw_method=MermaidDrawMethod.API,
#        background_color="white",
#        padding=10,
#    )
#    with open(png_output_path, 'wb') as f:
#        f.write(img_data)
#    logging.info(f"Graph visualization saved at: {png_output_path}")

# Update the pythonREPL tool to save the plot
class CustomPythonREPLTool(PythonREPLTool):
    def _run(self, query: str, **kwargs) -> Any:
        if self.sanitize_input:
            query = sanitize_input(query)

        dataset_csv_path = os.path.join('current_data', 'dataset.csv')

        local_context = {"st": st, "plt": plt, "pd": pd}
        exec(query, {"df": pd.read_csv(dataset_csv_path)}, local_context)

        plot_generated = False
        plot_dir = os.path.join('plotting_tools', 'temp_files', 'figs')
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        if plt.get_fignums():
            plot_path = generate_unique_image_path()
            plt.savefig(plot_path)

            # Save the code alongside the plot
            code_path = plot_path.replace('.png', '_code.txt')
            with open(code_path, 'w') as f:
                f.write(query)

            st.session_state.saved_plot_path = (plot_path, code_path)
            st.session_state.plot_image = plot_path
            st.session_state.new_plot_path = (plot_path, code_path)
            plot_generated = True

        if plot_generated:
            status_message = f"Plot generated = True. Saved at: {plot_path}"
            logging.info(status_message)
            st.session_state.plot_generated_status = status_message

        return f"Execution completed. Plot saved at: {plot_path if plot_generated else 'No plot generated'}"

# 4. Prompt Generation Functions
def generate_system_prompt_search(user_query, datasets_info):
    datasets_description = "\n"
    for i, row in datasets_info.iterrows():
        datasets_description += f"Dataset {i + 1}:\nName: {row['Name']}\nDescription: {row['Short Description']}\nParameters: {row['Parameters']}\n\n"

    prompt = (
        f"The user has provided the following query: {user_query}\n"
        f"Here are some datasets returned from the search:\n"
        f"{datasets_description}"
        "Please identify the top two datasets that best match the user's query and explain why they are the most relevant. If none are found, also report it.\n"
        "Please respond as a polite PangaeaGPT agent and keep in mind that you are responding to a user. "
        "The response should be at the level of ingenuity of a Nobel Prize laureate.\n"
        "Use the following schema in your response:\n"
        "{dataset name}\n"
        "{reason why relevant}\n"
        "{propose some short analysis and further questions to answer}"
    )
    return prompt

def generate_pandas_agent_system_prompt(user_query, dataset_name, dataset_description, df_head):
    prompt = (
        f"The user has provided the following query for the dataset: {user_query}\n"
        f"Dataset: {dataset_name}\n"
        f"Description: {dataset_description}\n"
        f"The head of the dataframe is (use it only as an example):\n"
        f"{df_head}\n"
        "The dataset 'df' is already loaded and available in your environment. Use this 'df' directly for analysis.\n"
        "Please help the user answer the question about the dataset using the entire DataFrame (not just the head). "
        "Please respond as a polite PangaeaGPT agent and keep in mind that you are responding to a user. "
        "The response should be at the level of ingenuity of a Nobel Prize laureate.\n"
        "Use the following schema in your response:\n"
        "Analysis: ...\n"
        "Further questions: ...\n"
    )
    return prompt

def generate_visualization_agent_system_prompt(user_query, dataset_name, dataset_description, df_head):
    prompt = (
        f"You are an agent designed to write and execute python code to answer questions.\n"
        f"The dataset name is: {dataset_name}\n"
        f"The dataset description is: {dataset_description}\n"
        f"The head of the dataframe is (use it only as an example):\n"
        f"{df_head}\n"
        "The dataset 'df' is already loaded and available in your environment. Use this 'df' directly for generating plots. Do not regenerate it from any headers.\n"
        "!IMPORTANT! PLEASE PAY CLOSE ATTENTION ON THE NAMES OF THE COLUMNS IN THE PROVIDED DATASET AND TAKE ONLY EXISTING COLUMNS.\n"
        "Generate a plot for the following user query: \"{user_query}\" using the provided DataFrame.\n"
        "The plot should be displayed inline and resized to be visually appealing.\n"
        "!IMPORTANT! Only plot something that could be done with the dataset. Do not plot random data. If not possible, return a simple message.\n"
        "{user_query} <- наиболее похожий плот был сгене\n"
        "Always start plot with schema:\n"
        "# Load libraries\n"
        "# Define plot\n"
        "# Make plot\n"
        "# Show plot\n"
    )
    return prompt

def generate_system_prompt_hard_coded_visualization():
    prompt = (
        "You are a hard-coded visualization agent. Your job is to plot sampling stations or the master track map on a map using the provided dataset.\n"
        "If the user request is related to a master track or sampling map, perform the plot accordingly. Use the expedition name (it should be short like PS126, PS121, etc.) as the main title.\n"
        "You must also determine the correct column names for each of the tool cases, for example latitude and longitude, might be named differently in the dataset (e.g., 'Lat', 'Lon').\n"
        "If you generate a meaningful plot, respond with 'The plot has been successfully generated.'. Do not loop again.\n"
        f"Dataset: {dataset_name}\n"
        f"Description: {dataset_description}\n"
        f"The head of the dataframe is (select appropriate attributes based on this):\n"
        f"{df_head}\n"
        "Respond with: 'This is a response from the plot sampling stations tool. Plot was successfully created.'\n"
    )
    return prompt

def generate_system_prompt_oceanographer():
    prompt = (
        "You are the oceanographer agent. Your job is to plot CTD data and TS diagrams using the provided dataset.\n"
        "Use the correct column names for pressure, temperature, and salinity to generate meaningful plots.\n"
        "Respond with: 'This is a response from the CTD plot tool. Plot was successfully created.' or 'This is a response from the TS plot tool. Plot was successfully created.'\n"
        "If you generate a meaningful plot, respond with 'FINISH'. Do not loop again.\n"
        f"Dataset: {dataset_name}\n"
        f"Description: {dataset_description}\n"
        f"The head of the dataframe is (select appropriate attributes based on this):\n"
        f"{df_head}\n"
    )
    return prompt


# 5. Search agent and tools
def search_pg_datasets_tool(query):
    global prompt_search
    if st.session_state.search_method == "Elastic Search":
        datasets_info = search_pg_es(query)
    else:
        datasets_info = pg_search_default(query)

    logging.debug("Datasets info: %s", datasets_info)

    if not datasets_info.empty:
        st.session_state.datasets_info = datasets_info
        st.session_state.messages_search.append({"role": "assistant", "content": f"**Search query:** {query}"})
        st.session_state.messages_search.append(
            {"role": "assistant", "content": "**Datasets Information:**", "table": datasets_info.to_json()}
        )

        datasets_description = "\n"
        for i, row in datasets_info.iterrows():
            datasets_description += f"Dataset {i + 1}:\nName: {row['Name']}\nDescription: {row['Short Description']}\nParameters: {row['Parameters']}\n\n"

        prompt_search = generate_system_prompt_search(query, datasets_info)
        #st.session_state.messages_search.append({"role": "assistant", "content": prompt})

    return datasets_info, prompt_search


memory = ChatMessageHistory(session_id="search-agent-session")


def create_search_agent(datasets_info=None):
    llm = ChatOpenAI(api_key=api_key, model_name=model_name)

    # Generate dataset description string
    datasets_description = ""
    if datasets_info is not None:
        for i, row in datasets_info.iterrows():
            datasets_description += f"Dataset {i + 1}:\nName: {row['Name']}\nDOI: {row['DOI']}\nDescription: {row['Short Description']}\nParameters: {row['Parameters']}\n\n"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             f"You are a powerful assistant primarily designed to search and retrieve datasets from PANGAEA. Your main goal is to help users find relevant datasets using the search_pg_datasets tool. When a user asks about datasets, always use this tool first to provide the most up-to-date and accurate information.\n\n"
             f"Here are some datasets returned from the search:\n{datasets_description}"
             "In addition to dataset searches, you have a secondary capability to answer questions about publications related to specific datasets (or in other words what was published based on this dataset). If a user explicitly asks about publications or research findings based on a particular dataset, you can use the answer_publication_questions tool. For example, you can handle queries like 'What was published based on this dataset?' or 'What were the main conclusions from the research using this dataset?'\n\n"
             "Remember:\n"
             "1. Prioritize dataset searches using the search_pg_datasets tool.\n"
             "2. Only use the answer_publication_questions tool when the user specifically asks about publications or research findings related to a dataset they've already identified. Please make sure that you correctly pass the doi to the tool. It should be doi retrieved after the search (user will point out which dataset it interested in). DO NOT GENERATE DOI ON THIS STEP OUT OF YOUR MIND! JUST TAKE WHAT WAS GIVEN WITH SYSTEM PROMPT.\n"
             "3. If needed, ask the user to clarify which dataset they're referring to before using the publication tool.\n\n"
             "Strive to provide accurate, helpful, and concise responses to user queries."
             ),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="chat_history"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    search_tool = StructuredTool.from_function(
        func=search_pg_datasets_tool,
        name="search_pg_datasets",
        description="List datasets from PANGAEA based on a query"
    )

    publication_qa_tool = StructuredTool.from_function(
        func=answer_publication_questions,
        name="answer_publication_questions",
        description="A tool to answer questions about articles published from this dataset. This will be a journal article for which you should provide the tool with an already structured question about what the user wants. The input should be the DOI of the dataset (e.g. 'https://doi.org/10.1594/PANGAEA.xxxxxx') and the question. The question should be reworded to specifically send it to RAG. E.g. the hypothetical user's question 'Are there any related articles to the first dataset? If so what these articles are about?' will be re-worded for this tool as 'What is this article is about?'",
        args_schema=PublicationQAArgs
    )

    tools = [search_tool, publication_qa_tool]

    llm_with_tools = llm.bind_tools(tools)

    agent = (
            {
                "input": lambda x: x["input"],
                "chat_history": lambda x: x.get("chat_history", []),
                "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
            }
            | prompt
            | llm_with_tools
            | OpenAIToolsAgentOutputParser()
    )

    return AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5)



search_agent = create_search_agent(datasets_info=st.session_state.datasets_info)


# 6. Visualization and Oceanography Tools
#Create tools for map visualization and oceanography
class PlotSamplingStationsArgs(BaseModel):
    main_title: str = Field(description="The main title for the plot")
    lat_col: str = Field(description="Name of the latitude column")
    lon_col: str = Field(description="Name of the longitude column")

class PlotMasterTrackMapArgs(BaseModel):
    main_title: str = Field(description="The main title for the plot")
    lat_col: str = Field(description="Name of the latitude column")
    lon_col: str = Field(description="Name of the longitude column")
    date_col: str = Field(description="Name of the date/time column")

class CTDPlotToolArgs(BaseModel):
    main_title: str = Field(description="The main title for the plot")
    pressure_col: str = Field(description="Name of the pressure column")
    temperature_col: str = Field(description="Name of the temperature column")
    salinity_col: str = Field(description="Name of the salinity column")

class TSPlotToolArgs(BaseModel):
    main_title: str = Field(description="The main title for the plot")
    temperature_col: str = Field(description="Name of the temperature column")
    salinity_col: str = Field(description="Name of the salinity column")



def plot_sampling_stations_tool(main_title, lat_col, lon_col, *args, **kwargs):
    plot_sampling_stations(main_title=main_title, lat_col=lat_col, lon_col=lon_col)
    return {"result": "Plot generated successfully."}

def plot_master_track_map_tool(main_title, lat_col, lon_col, date_col, *args, **kwargs):
    plot_master_track_map(main_title=main_title, lat_col=lat_col, lon_col=lon_col, date_col=date_col)
    return {"result": "Plot generated successfully."}

def plot_ctd_data_tool(main_title, pressure_col, temperature_col, salinity_col, *args, **kwargs):
    return plot_ctd_profiles(main_title = main_title, pressure_col=pressure_col, temperature_col=temperature_col, salinity_col=salinity_col)

def plot_ts_diagram_tool(main_title, temperature_col, salinity_col, *args, **kwargs):
    return plot_ts_diagram(main_title=main_title, temperature_col=temperature_col, salinity_col=salinity_col)


visualization_functions: list[BaseTool] = [
    StructuredTool.from_function(
        func=plot_sampling_stations_tool,
        name="plot_sampling_stations_tool",
        description="Plot sampling stations on a map using the dataset.",
        args_schema=PlotSamplingStationsArgs
    ),
    StructuredTool.from_function(
        func=plot_master_track_map_tool,
        name="plot_master_track_map_tool",
        description="Plot the master track map using the dataset.",
        args_schema=PlotMasterTrackMapArgs
    )
]

oceanography_functions: list[BaseTool] = [
    StructuredTool.from_function(
        func=plot_ctd_data_tool,
        name="plot_ctd_data_tool",
        description="Plot CTD data using the provided columns.",
        args_schema=CTDPlotToolArgs
    ),
    StructuredTool.from_function(
        func=plot_ts_diagram_tool,
        name="plot_ts_diagram_tool",
        description="Plot TS diagram using the provided columns.",
        args_schema=TSPlotToolArgs
    )
]

# 7. Agents Creation
def create_pandas_agent(user_query, dataset_name, dataset_description, df_head):
    prompt = generate_pandas_agent_system_prompt(user_query, dataset_name, dataset_description, df_head)
    llm = ChatOpenAI(api_key=api_key, model_name=model_name)
    agent_pandas = create_pandas_dataframe_agent(
        llm=llm,
        df=st.session_state.dataset_df,
        agent_type="tool-calling",
        prefix="",
        suffix=prompt,
        include_df_in_prompt=False,
        verbose=True,
        allow_dangerous_code=True
    )
    return agent_pandas

def create_visualization_agent(user_query, dataset_name, dataset_description, df_head):
    prompt = generate_visualization_agent_system_prompt(user_query, dataset_name, dataset_description, df_head)
    llm = ChatOpenAI(api_key=api_key, model_name=model_name)
    repl_tool = CustomPythonREPLTool()
    agent_visualization = create_openai_tools_agent(
        llm,
        tools=[repl_tool],
        prompt=ChatPromptTemplate.from_messages(
            [
                ("system", prompt),
                MessagesPlaceholder(variable_name="messages"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ]
        ).partial(
            user_query=user_query,
            dataset_name=dataset_name,
            dataset_description=dataset_description,
            head_of_dataframe=df_head
        )
    )
    return AgentExecutor(agent=agent_visualization, tools=[repl_tool], verbose=True, handle_parsing_errors=True, return_intermediate_steps=True)

def create_hard_coded_visualization_agent():
    llm = ChatOpenAI(api_key=api_key, model_name=model_name)
    system_prompt = generate_system_prompt_hard_coded_visualization()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ]
    )
    agent = create_openai_tools_agent(llm, visualization_functions, prompt)
    return AgentExecutor(agent=agent, tools=visualization_functions)

# Create Oceanographer Agent
def create_oceanographer_agent():
    llm = ChatOpenAI(api_key=api_key, model_name=model_name)
    system_prompt = generate_system_prompt_oceanographer()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ]
    )
    agent = create_openai_tools_agent(llm, oceanography_functions, prompt)
    return AgentExecutor(agent=agent, tools=oceanography_functions)


#tool = Tool(
#    name="search_pg_datasets",
#    func=search_pg_datasets_tool,
#    description="List datasets from PANGAEA based on a query"
#)

#functions = [convert_to_openai_function(tool)]

#client = openai.OpenAI(api_key=api_key)

#search_model = ChatOpenAI(api_key=api_key, model_name=selected_model_name)
#model_with_tools = search_model.bind_tools([tool])


# 8. Agent initialization and execution + Supervisor Agent
def initialize_agents(user_query):
    # Check if the dataset is loaded
    if st.session_state.dataset_df is not None:
        # Extract necessary information from the session state
        df_head = st.session_state.dataset_df.head().to_string()
        dataset_name = st.session_state.dataset_name
        dataset_description = st.session_state.datasets_info.loc[
            st.session_state.datasets_info['DOI'] == st.session_state.active_dataset, 'Description'
        ].values[0]

        # Create the visualization agent
        visualization_agent = create_visualization_agent(
            user_query=user_query,
            dataset_name=dataset_name,
            dataset_description=dataset_description,
            df_head=df_head
        )

        # Create the pandas dataframe agent
        dataframe_agent = create_pandas_agent(
            user_query=user_query,
            dataset_name=dataset_name,
            dataset_description=dataset_description,
            df_head=df_head
        )

        # Create the hard-coded visualization agent
        hard_coded_visualization_agent = create_hard_coded_visualization_agent()

        # Create the oceanographer agent
        oceanographer_agent = create_oceanographer_agent()

        # Return all four agents
        return visualization_agent, dataframe_agent, hard_coded_visualization_agent, oceanographer_agent
    else:
        # Warning if no dataset is loaded
        st.warning("No dataset loaded. Please load a dataset first.")
        # Ensure the function returns four values even if None
        return None, None, None, None


def agent_node(state, agent, name):
    logging.debug(f"Entering agent_node for {name}")
    # Clear the intermediate steps file before each iteration
    intermediate_steps_file_path = os.path.join('current_data', 'intermediate_steps_visualization_agent.txt')
    with open(intermediate_steps_file_path, 'w') as f:
        f.write("")

    # Ensure 'agent_scratchpad' exists in the state and is a list
    if 'agent_scratchpad' not in state or not isinstance(state['agent_scratchpad'], list):
        state['agent_scratchpad'] = []

    # Ensure 'input' exists in the state
    if 'input' not in state:
        logging.debug(f"Adding missing 'input' key to state for agent {name}.")
        state['input'] = state.get('messages', [])[0].content

    # Set the user query in the state
    state['user_query'] = state.get('input', '')

    # Ensure 'plot_images' exists in the state and is a list
    if 'plot_images' not in state or not isinstance(state['plot_images'], list):
        state['plot_images'] = []

    # Invoke the agent with the current state
    result = agent.invoke(state)
    last_message_content = result["output"]
    intermediate_steps = result.get("intermediate_steps", [])

    # Save intermediate steps to a .txt file
    with open(intermediate_steps_file_path, 'a') as f:
        for step in intermediate_steps:
            f.write(f"Action: {step[0]}\nObservation: {step[1]}\n")

    if name == "VisualizationAgent":
        new_plot_path = st.session_state.get("new_plot_path")
        if new_plot_path and isinstance(new_plot_path, tuple):
            plot_path, code_path = new_plot_path
            if os.path.exists(plot_path) and os.path.exists(code_path):
                state["plot_images"].append(new_plot_path)
                st.session_state.new_plot_path = None

    # Check if a new plot was generated
    new_plot_path = st.session_state.get("new_plot_path")
    logging.info(f"New plot path from session state: {new_plot_path}")

    if new_plot_path:
        logging.info(f"Checking if file exists: {new_plot_path}")
        if os.path.exists(new_plot_path):
            logging.info(f"File exists: {new_plot_path}")
            state["plot_images"].append(new_plot_path)
            logging.info(f"Added new plot to state: {new_plot_path}")
            st.session_state.new_plot_path = None  # Reset the new plot path
            logging.info("Reset new_plot_path in session state")
        else:
            logging.warning(f"File does not exist: {new_plot_path}")

    if name == "VisualizationAgent":
        state["visualization_agent_used"] = True

    # Return the updated state
    logging.info(f"Returning state for {name}: {state}")
    return {
        "messages": [HumanMessage(content=last_message_content, name=name)],
        "next": name,
        "agent_scratchpad": state['agent_scratchpad'],
        "last_agent_message": last_message_content,  # Add this line
        "plot_images": state['plot_images']
    }


def create_supervisor_agent(user_query, dataset_name, dataset_description, df_head):
    # Define the members (agents) involved in the workflow
    members = ["VisualizationAgent", "DataFrameAgent", "HardCodedVisualizationAgent", "OceanographerAgent"]

    # Create the system prompt for the supervisor agent
    system_prompt_supervisor = (
        f"You are a supervisor tasked with managing a conversation between the following workers: {members}. "
        f"Given the following user request: '{user_query}', determine and instruct the next worker to act. "
        f"Each worker will perform a task and respond with their results and status. "
        f"If the request involves plotting a master track or sampling map, directly assign the task to the HardCodedVisualizationAgent. "
        f"For CTD data plots, assign the task to the OceanographerAgent. The other requests should be handled by the VisualizationAgent. Extremely important to assign the correct task to the correct agent and use HardCodedVisualizationAgent and OceanographerAgent only for the described cases."
        f"If a meaningful plot is generated, end the process by returning FINISH to avoid unnecessary loops.\n"
        f"The dataset name is: {dataset_name}\n"
        f"The dataset description is: {dataset_description}\n"
        f"The head of the dataframe is (use it only as an example):\n{df_head}\n"
        f"The dataset 'df' is already loaded and available in your environment. Use this 'df' directly for analysis."
    )

    # Define the options for the next task assignment
    options = ["FINISH"] + members

    # Define the function for routing the next task
    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [
                        {"enum": options},
                    ],
                }
            },
            "required": ["next"],
        },
    }

    # Create the prompt template for the supervisor
    prompt_supervisor = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt_supervisor),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
            ("system",
             f"Given the conversation above, who should act next? Or should we FINISH? Select one of: {options}. If the task is related to plots and you receive 'result: Plot generated successfully.', choose 'FINISH'."
             f"If the task is just about the dataset, do not call any plotting agents. Simply answer to the user's question then."
             f"The last agent message was: {{last_agent_message}}")
        ]
    ).partial(options=str(options), members=", ".join(members))

    # Initialize the supervisor LLM
    llm_supervisor = ChatOpenAI(api_key=api_key, model_name=model_name)

    # Create the supervisor chain
    supervisor_chain = (
            {
                "messages": lambda x: x["messages"],
                "agent_scratchpad": lambda x: x["agent_scratchpad"],
                "last_agent_message": lambda x: x.get("last_agent_message", ""),  # Add this line
            }
            | prompt_supervisor
            | llm_supervisor.bind_functions(functions=[function_def], function_call="route")
            | JsonOutputFunctionsParser()
    )

    # Define the AgentState type
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        next: str
        agent_scratchpad: Annotated[Sequence[BaseMessage], operator.add]
        user_query: str
        last_agent_message: str  # Add this line if it's not already there
        plot_images: List[str]

    # Create the workflow graph
    workflow = StateGraph(AgentState)
    visualization_agent, dataframe_agent, hard_coded_visualization_agent, oceanographer_agent = initialize_agents(
        user_query)

    # Add agents to the workflow if they are successfully initialized
    if visualization_agent and dataframe_agent and hard_coded_visualization_agent and oceanographer_agent:
        workflow.add_node("VisualizationAgent",
                          functools.partial(agent_node, agent=visualization_agent, name="VisualizationAgent"))
        workflow.add_node("DataFrameAgent", functools.partial(agent_node, agent=dataframe_agent, name="DataFrameAgent"))
        workflow.add_node("HardCodedVisualizationAgent",
                          functools.partial(agent_node, agent=hard_coded_visualization_agent,
                                            name="HardCodedVisualizationAgent"))
        workflow.add_node("OceanographerAgent",
                          functools.partial(agent_node, agent=oceanographer_agent, name="OceanographerAgent"))
        workflow.add_node("supervisor", supervisor_chain)

        # Connect agents to the supervisor
        for member in members:
            workflow.add_edge(member, "supervisor")

        # Define the conditional map for routing
        conditional_map = {k: k for k in members}
        conditional_map["FINISH"] = END
        workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
        workflow.set_entry_point("supervisor")

        # Compile the workflow into a graph
        graph = workflow.compile()

        # Define the output folder for saving the graph visualization
        output_folder = os.path.join(os.getcwd(), 'plotting_tools', 'temp_files')

        # Save the graph visualization
        #save_graph_visualization(graph, output_folder)

        return graph
    else:
        return None


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

        dataset_folder = os.path.join(os.getcwd(), 'current_data')
        dataset_csv_path = os.path.join(dataset_folder, 'dataset.csv')

        if os.path.exists(dataset_folder):
            shutil.rmtree(dataset_folder)
        os.makedirs(dataset_folder)

        with open(dataset_csv_path, "wb") as f:
            f.write(csv_data)
        st.session_state.dataset_csv_path = dataset_csv_path

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
        graph = create_supervisor_agent(user_query, dataset_name, dataset_description, df_head)

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
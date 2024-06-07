# Imports and initial setup
import streamlit as st
import requests
import pandas as pd
import warnings
import re
import os
import logging
import openai
import json
import operator
import shutil
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools import Tool
from langchain_core.messages import HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain_experimental.tools.python.tool import PythonREPLTool
from io import StringIO
import pangaeapy.pandataset as pdataset
import matplotlib.pyplot as plt
from typing import Optional, Any
import functools
from langgraph.graph import StateGraph, END
from typing import Annotated, Sequence, TypedDict
from plotting_tools.hard_agent import plot_sampling_stations, plot_master_track_map
from langchain_core.tools import BaseTool

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress specific FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning, message="Downcasting behavior in `replace` is deprecated")
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Streamlit app setup
st.set_page_config(layout="wide")

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

for key, value in session_state_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Load secrets from the `secrets.toml` file
openai_api_key = st.secrets["general"]["openai_api_key"]
langchain_api_key = st.secrets["general"].get("langchain_api_key", None)
langchain_project_name = st.secrets["general"].get("langchain_project_name", None)


# Sidebar inputs for model selection
with st.sidebar:
    st.title("Configuration")
    model_name = st.selectbox("Select Model", ["gpt-3.5-turbo", "gpt-4o"], key="model_name")

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


# Define necessary functions
def sanitize_input(query: str) -> str:
    return query.strip()

def pg_search_es(query=None, size=5, from_idx=0, source=None, df=None, analyzer=None,
                 default_operator=None, explain=False, sort=None, track_scores=True,
                 timeout=None, terminate_after=None, search_type=None,
                 analyze_wildcard=False, version=False, **kwargs):
    url = "https://ws.pangaea.de/es/pangaea/panmd/_search"
    params = {
        'q': query,
        'size': str(size),
        'from': str(from_idx),
        '_source': ','.join(source) if source else None,
        'df': df,
        'analyzer': analyzer,
        'default_operator': default_operator,
        'explain': explain,
        'sort': sort,
        'track_scores': track_scores,
        'timeout': timeout,
        'terminate_after': terminate_after,
        'search_type': search_type,
        'analyze_wildcard': analyze_wildcard,
        'version': version
    }

    params = {k: v for k, v in params.items() if v is not None}

    logging.debug("Sending request to PANGAEA with parameters: %s", params)
    response = requests.get(url, params=params, **kwargs)
    response.raise_for_status()
    results = response.json()
    logging.debug("Received response from PANGAEA")

    hits = results['hits']['hits']
    df = pd.DataFrame(hits)
    df.attrs['total'] = results['hits']['total']
    df.attrs['max_score'] = results['hits']['max_score']

    return df

def list_datasets(query):
    results = pg_search_es(query=query, size=5)
    message_content = f"Search phrase: {query} - Total Hits: {results.attrs['total']}, Max Score: {results.attrs['max_score']}"
    st.session_state.messages_search.append({"role": "assistant", "content": message_content})
    logging.debug("Search results: %s", message_content)

    datasets_info = []
    search_results_dir = os.path.join('current_data', 'search_results')
    os.makedirs(search_results_dir, exist_ok=True)  # Ensure the search_results directory exists

    for index, row in results.iterrows():
        xml_content = row['_source'].get('xml', '')
        score = row['_score']

        xml_file_path = os.path.join(search_results_dir, f"dataset_{index + 1}.xml")
        with open(xml_file_path, 'w') as file:
            file.write(xml_content)

        doi_match = re.search(r'<md:URI>(https://doi.org/\S+)</md:URI>', xml_content)
        name_match = re.search(r'<md:title>([^<]+)</md:title>', xml_content)
        description_match = re.search(r'<md:abstract>([^<]+)</md:abstract>', xml_content)
        parameters_match = re.findall(r'<md:matrixColumn.*?source="data".*?<md:name>([^<]+)</md:name>', xml_content, re.DOTALL)

        if doi_match and name_match:
            doi = doi_match.group(1).strip(')')
            doi_number = doi.split('/')[-1]
            name = name_match.group(1)
            description = description_match.group(1) if description_match else "No description available"
            short_description = " ".join(description.split()[:100]) + "..." if len(description.split()) > 100 else description
            parameters = ", ".join(parameters_match[:10]) + ("..." if len(parameters_match) > 10 else "")
            datasets_info.append(
                {'Number': index + 1, 'Name': name, 'DOI': doi, 'DOI Number': doi_number, 'Description': description,
                 'Short Description': short_description, 'Score': score, 'Parameters': parameters})

    df_datasets = pd.DataFrame(datasets_info)
    return df_datasets


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
        "Always start plot with schema:\n"
        "# Load libraries\n"
        "# Define plot\n"
        "# Make plot\n"
        "# Show plot\n"
    )
    return prompt

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

def convert_df_to_csv(df):
    logging.debug("Converting DataFrame to CSV")
    return df.to_csv().encode('utf-8')

def list_datasets_tool(query):
    datasets_info = list_datasets(query["__arg1"])
    logging.debug("Datasets info: %s", datasets_info)

    if not datasets_info.empty:
        st.session_state.datasets_info = datasets_info
        st.session_state.messages_search.append(
            {"role": "assistant", "content": "Datasets Information:", "table": datasets_info.to_json()})

        system_prompt = generate_system_prompt_search(query["__arg1"], datasets_info)
        logging.debug("Generated system prompt: %s", system_prompt)

        response = model_with_tools.invoke([HumanMessage(content=system_prompt)])
        logging.debug("Response from model: %s", response)

        top_datasets_message = response.content
        logging.debug("Top datasets message: %s", top_datasets_message)

        st.session_state.messages_search.append({"role": "assistant", "content": top_datasets_message})
    return datasets_info

def plot_sampling_stations_tool(main_title, *args, **kwargs):
    plot_sampling_stations(main_title=main_title)
    return {"result": "Plot generated successfully."}

def plot_master_track_map_tool(main_title, *args, **kwargs):
    plot_master_track_map(main_title=main_title)
    return {"result": "Plot generated successfully."}

visualization_functions: list[BaseTool] = [
    Tool(
        name="plot_sampling_stations_tool",
        func=plot_sampling_stations_tool,
        description="Plot sampling stations on a map using the dataset."
    ),
    Tool(
        name="plot_master_track_map_tool",
        func=plot_master_track_map_tool,
        description="Plot the master track map using the dataset."
    )
]

def generate_system_prompt_hard_coded_visualization():
    prompt = (
        "You are a hard-coded visualization agent. Your job is to plot sampling stations or the master track map on a map using the provided dataset.\n"
        "If the user request is related to a master track or sampling map, perform the plot accordingly. Use the expedition name (it should be short like PS126, PS121, etc.) as the main title.\n"
        "If you generate a meaningful plot, respond with 'FINISH'. Do not loop again.\n"
        f"Dataset: {dataset_name}\n"
        f"Description: {dataset_description}\n"
        "Respond with: 'This is a response from the plot sampling stations tool.'\n"
    )
    return prompt

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

tool = Tool(
    name="list_datasets",
    func=list_datasets_tool,
    description="List datasets from PANGAEA based on a query"
)

functions = [convert_to_openai_function(tool)]

client = openai.OpenAI(api_key=api_key)

search_model = ChatOpenAI(api_key=api_key, model_name=selected_model_name)
model_with_tools = search_model.bind_tools([tool])

def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

def agent_node(state, agent, name):
    if 'agent_scratchpad' not in state or not isinstance(state['agent_scratchpad'], list):
        state['agent_scratchpad'] = []
    if 'input' not in state:
        logging.debug(f"Adding missing 'input' key to state for agent {name}.")
        state['input'] = state.get('messages', [])[0].content

    state['user_query'] = state.get('input', '')

    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)], "agent_scratchpad": state['agent_scratchpad']}


class CustomPythonREPLTool(PythonREPLTool):
    def _run(self, query: str, **kwargs) -> Any:
        if self.sanitize_input:
            query = sanitize_input(query)

        dataset_csv_path = os.path.join('current_data', 'dataset.csv')

        local_context = {"st": st, "plt": plt, "pd": pd}
        exec(query, {"df": pd.read_csv(dataset_csv_path)}, local_context)

        plot_generated = False
        plot_dir = os.path.join('plotting_tools', 'temp_files')
        plot_path = os.path.join(plot_dir, 'plot.png')
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        if plt.get_fignums():
            plt.savefig(plot_path)
            st.session_state.saved_plot_path = plot_path
            plot_generated = True

        if plot_generated:
            status_message = "Plot generated = True"
            logging.info(status_message)
            st.session_state.plot_generated_status = status_message

        return "Execution completed"


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
    return AgentExecutor(agent=agent_visualization, tools=[repl_tool], verbose=True, handle_parsing_errors=True)

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
        verbose=True
    )
    return agent_pandas

def initialize_agents(user_query):
    if st.session_state.dataset_df is not None:
        df_head = st.session_state.dataset_df.head().to_string()
        dataset_name = st.session_state.dataset_name
        dataset_description = st.session_state.datasets_info.loc[
            st.session_state.datasets_info['DOI'] == st.session_state.active_dataset, 'Description'
        ].values[0]

        visualization_agent = create_visualization_agent(
            user_query=user_query,
            dataset_name=dataset_name,
            dataset_description=dataset_description,
            df_head=df_head
        )
        dataframe_agent = create_pandas_agent(
            user_query=user_query,
            dataset_name=dataset_name,
            dataset_description=dataset_description,
            df_head=df_head
        )
        hard_coded_visualization_agent = create_hard_coded_visualization_agent()

        return visualization_agent, dataframe_agent, hard_coded_visualization_agent
    else:
        st.warning("No dataset loaded. Please load a dataset first.")
        return None, None, None

def create_supervisor_agent(user_query, dataset_name, dataset_description, df_head):
    members = ["VisualizationAgent", "DataFrameAgent", "HardCodedVisualizationAgent"]
    system_prompt_supervisor = (
        f"You are a supervisor tasked with managing a conversation between the following workers: {members}. "
        f"Given the following user request: '{user_query}', determine and instruct the next worker to act. "
        f"Each worker will perform a task and respond with their results and status. "
        f"If the request involves plotting a master track or sampling map, directly assign the task to the HardCodedVisualizationAgent in other plotting cases call VisualizationAgent, it can produce plots with non hardcoded code. "
        f"If a meaningful plot is generated, end the process by returning FINISH to avoid unnecessary loops.\n"
        f"The dataset name is: {dataset_name}\n"
        f"The dataset description is: {dataset_description}\n"
        f"The head of the dataframe is (use it only as an example):\n{df_head}\n"
        f"The dataset 'df' is already loaded and available in your environment. Use this 'df' directly for analysis."
    )
    options = ["FINISH"] + members
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
    prompt_supervisor = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt_supervisor),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
            ("system", "Given the conversation above, who should act next? Or should we FINISH? Select one of: {options}")
        ]
    ).partial(options=str(options), members=", ".join(members))

    llm_supervisor = ChatOpenAI(api_key=api_key, model_name=model_name)
    supervisor_chain = (
        prompt_supervisor
        | llm_supervisor.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
    )

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        next: str
        agent_scratchpad: Annotated[Sequence[BaseMessage], operator.add]
        user_query: str

    workflow = StateGraph(AgentState)
    visualization_agent, dataframe_agent, hard_coded_visualization_agent = initialize_agents(user_query)
    if visualization_agent and dataframe_agent and hard_coded_visualization_agent:
        workflow.add_node("VisualizationAgent", functools.partial(agent_node, agent=visualization_agent, name="VisualizationAgent"))
        workflow.add_node("DataFrameAgent", functools.partial(agent_node, agent=dataframe_agent, name="DataFrameAgent"))
        workflow.add_node("HardCodedVisualizationAgent", functools.partial(agent_node, agent=hard_coded_visualization_agent, name="HardCodedVisualizationAgent"))
        workflow.add_node("supervisor", supervisor_chain)

        for member in members:
            workflow.add_edge(member, "supervisor")

        conditional_map = {k: k for k in members}
        conditional_map["FINISH"] = END
        workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
        workflow.set_entry_point("supervisor")

        graph = workflow.compile()
        return graph
    else:
        return None

# Process user input
if st.session_state.current_page == "search":
    st.title("PANGAEA Dataset Explorer")
    user_input = st.chat_input("Enter your query:")

    if user_input:
        st.session_state.messages_search.append({"role": "user", "content": f"User input: {user_input}"})
        logging.debug("User input: %s", user_input)
        message = model_with_tools.invoke([HumanMessage(content=user_input)])
        logging.debug("Message from model: %s", message)

        tool_call = message.additional_kwargs.get("tool_calls", [])
        if tool_call:
            for call in tool_call:
                tool_response = call['function']['arguments']
                st.session_state.messages_search.append({"role": "assistant", "content": json.dumps(tool_response)})
                logging.debug("Tool response: %s", tool_response)

                if isinstance(tool_response, str):
                    tool_response = json.loads(tool_response)

                if isinstance(tool_response, dict) and "__arg1" in tool_response:
                    datasets_info = list_datasets_tool({"__arg1": tool_response["__arg1"]})
                    if datasets_info.empty:
                        st.session_state.messages_search.append({"role": "assistant", "content": "No datasets found."})
                        logging.debug("No datasets found.")
                else:
                    st.session_state.messages_search.append({"role": "assistant", "content": "Invalid tool response."})
                    logging.debug("Invalid tool response: %s", tool_response)
        else:
            st.session_state.messages_search.append({"role": "assistant", "content": message.content})
            logging.debug("Assistant message: %s", message.content)

        st.session_state.show_dataset = False
        st.rerun()

    for i, message in enumerate(st.session_state.messages_search):
        with st.chat_message(message["role"]):
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
                    cols[4].write(f"**Parameters:** {row['Parameters']}")
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

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="Download data as CSV",
                    data=csv_data,
                    file_name=f"dataset.csv",
                    mime='text/csv',
                    key=f"download-{doi.split('/')[-1]}"
                )
            with col2:
                if st.button("Send to Data Agent"):
                    st.session_state.dataset_csv = csv_data
                    st.session_state.dataset_df = dataset
                    st.session_state.dataset_name = st.session_state.datasets_info.loc[
                        st.session_state.datasets_info['DOI'] == doi, 'Name'].values[0]
                    st.session_state.current_page = "data_agent"
                    st.rerun()

if st.session_state.current_page == "data_agent":
    st.title("Pangaea Data Agent")

    col1, col2 = st.columns([9, 1])
    with col2:
        if st.button("Back to Search"):
            st.session_state.current_page = "search"
            st.rerun()

    if st.session_state.dataset_df is not None:
        user_query = st.session_state.messages_data_agent[-1]["content"] if st.session_state.messages_data_agent else ""
        dataset_name = st.session_state.dataset_name
        dataset_description = st.session_state.datasets_info.loc[
            st.session_state.datasets_info['DOI'] == st.session_state.active_dataset, 'Description'
        ].values[0]
        df_head = st.session_state.dataset_df.head().to_string()

        graph = create_supervisor_agent(user_query, dataset_name, dataset_description, df_head)

        user_input = st.chat_input("Enter your query:")

        if user_input:
            st.session_state.messages_data_agent.append({"role": "user", "content": f"User input: {user_input}"})
            logging.debug("User input: %s", user_input)

            if graph:
                initial_state = {
                    "messages": [HumanMessage(content=user_input)],
                    "next": "supervisor",
                    "agent_scratchpad": [],
                    "input": user_input
                }
                logging.debug(f"Initial state before invoking graph: {initial_state}")
                try:
                    response = graph.invoke(initial_state)
                    logging.debug("Agent response: %s", response)
                    st.session_state.messages_data_agent.append(
                        {"role": "assistant", "content": response['messages'][-1].content})
                except Exception as e:
                    logging.error(f"Error invoking graph: {e}")

            st.rerun()

        for message in st.session_state.messages_data_agent:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Force check if plot is generated and show it
        plot_path = os.path.join('plotting_tools', 'temp_files', 'plot.png')
        if os.path.exists(plot_path):
            st.session_state.saved_plot_path = plot_path

        #if st.session_state.saved_plot_path and os.path.exists(st.session_state.saved_plot_path):
        #    with st.expander("Generated Plot", expanded=False):
        #        st.image(st.session_state.saved_plot_path, caption='Generated Plot', use_column_width=True)

        if "plot_generated_status" in st.session_state:
            st.markdown(f"**{st.session_state.plot_generated_status}**")

        col1, col2 = st.columns([5, 5])
        with col1:
            with st.expander("Dataset Info", expanded=False):
                st.write(dataset_name)
                st.dataframe(st.session_state.dataset_df)
        with col2:
            with st.expander("Current Plot", expanded=False):
                if st.session_state.saved_plot_path and os.path.exists(st.session_state.saved_plot_path):
                    st.image(st.session_state.saved_plot_path, caption='Current Plot', use_column_width=True)

    else:
        st.warning("No dataset loaded. Please load a dataset first.")



# src/agents.py
import base64
import os
import logging
import functools
import subprocess
import sys
from io import StringIO
from typing import List, Annotated, Sequence, TypedDict
import operator
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import ChatOpenAI, OpenAI
from langchain_core.messages import AIMessage
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from pydantic import BaseModel, Field, PrivateAttr
from langchain_core.tools import StructuredTool
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langgraph.graph import StateGraph, END
from langchain.agents.agent_types import AgentType


from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from typing import Any
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI


#sys.path
# Get the absolute path of the current file (agents.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

# Add the parent directory to sys.path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


# Import custom modules
from .search.search_pg_default import pg_search_default
from .search.publication_qa_tool import answer_publication_questions, PublicationQAArgs
from .plotting_tools.hard_agent import plot_master_track_map
from .plotting_tools.oceanographer_tools import plot_ts_diagram
from .prompts import Prompts
from .utils import generate_unique_image_path
from .config import API_KEY


# 1. Search Agent and Tools
class CustomPythonREPLTool(PythonREPLTool):
    _datasets: dict = PrivateAttr()

    def __init__(self, datasets, **kwargs):
        """
        Custom Python REPL tool that injects dataset variables and logs plot generation.
        :param datasets: Dictionary { "dataset_1": <DataFrame>, "dataset_2": <DataFrame>, ... }
        """
        super().__init__(**kwargs)
        self._datasets = datasets

    def _run(self, query: str, **kwargs) -> Any:
        """
        Execute the user-provided Python code in a local context containing:
          - st (Streamlit)
          - plt (Matplotlib Pyplot)
          - pd (Pandas)
          - All loaded dataset variables (self._datasets)
          - A dynamically generated plot_path

        If a figure is saved to plot_path, a "plot_generated" event will be logged in session_state["execution_history"].
        """
        import streamlit as st
        import matplotlib.pyplot as plt
        import pandas as pd
        import logging
        from io import StringIO
        from src.utils import log_history_event, generate_unique_image_path

        # Prepare local context with necessary packages
        local_context = {"st": st, "plt": plt, "pd": pd}

        # Inject the user’s datasets under the specified variable names (e.g. dataset_1, dataset_2, etc.)
        local_context.update(self._datasets)

        # Generate a unique file path for the plot (plot_path)
        plot_path = generate_unique_image_path()
        local_context['plot_path'] = plot_path

        # Redirect stdout so we can capture any output from exec(...)
        old_stdout = sys.stdout
        redirected_output = StringIO()
        sys.stdout = redirected_output

        try:
            # Execute user code
            exec(query, local_context)
            output = redirected_output.getvalue()

        except ModuleNotFoundError as e:
            missing_module = e.name
            logging.warning(f"Module '{missing_module}' not found during code execution.")
            return {
                "error": "ModuleNotFoundError",
                "missing_module": missing_module,
                "message": f"The Python module '{missing_module}' is not installed."
            }
        except Exception as e:
            logging.error(f"Error during code execution: {e}")
            return {
                "error": "ExecutionError",
                "message": str(e)
            }
        finally:
            # Restore stdout
            sys.stdout = old_stdout

        # Check if a plot was actually saved to plot_path
        plot_generated = False
        if os.path.exists(plot_path):
            st.session_state.saved_plot_path = plot_path
            st.session_state.plot_image = plot_path
            st.session_state.new_plot_path = plot_path
            plot_generated = True

        if plot_generated:
            status_message = f"Plot generated = True. Saved at: {plot_path}"
            logging.info(status_message)
            st.session_state.plot_generated_status = status_message
            
            from src.utils import log_history_event
            log_history_event(
                st.session_state,
                "plot_generated",
                {
                    "plot_path": plot_path.replace("sandbox:", ""),  # Remove sandbox prefix
                    "agent": "VisualizationAgent",
                    "description": "Python_REPL Generated Plot",
                    "content": query  # Store the actual code used
                }
            )

        return {
            "result": f"Execution completed. Plot saved at: {plot_path if plot_generated else 'No plot generated'}",
            "output": output,
            "plot_images": [plot_path] if plot_generated else []
        }



def search_pg_datasets_tool(query):
    global prompt_search

    datasets_info = pg_search_default(query)

    logging.debug("Datasets info: %s", datasets_info)

    if not datasets_info.empty:
        st.session_state.datasets_info = datasets_info
        st.session_state.messages_search.append({
            "role": "assistant", 
            "content": f"**Search query:** {query}"
        })
        # Pass the table as JSON (you can use orient="split" or the default, as long as your UI can parse it)
        st.session_state.messages_search.append({
            "role": "assistant", 
            "content": "**Datasets Information:**", 
            "table": datasets_info.to_json(orient="split")
        })

        # Optionally, build a detailed description string for the prompt:
        datasets_description = ""
        for i, row in datasets_info.iterrows():
            datasets_description += (
                f"Dataset {i + 1}:\n"
                f"Name: {row['Name']}\n"
                f"Description: {row['Short Description']}\n"
                f"Parameters: {row['Parameters']}\n\n"
            )

        prompt_search = (
            f"The user has provided the following query: {query}\n"
            f"Available datasets:\n{datasets_description}\n"
            "Please identify the top two datasets that best match the user's query and explain why they are the most relevant. "
            "Do not suggest datasets without values in the Parameters field, because they cannot be directly downloaded.\n"
            "Respond using the following schema:\n"
            "{dataset name}\n{reason why relevant}\n{propose some short analysis and further questions to answer}"
        )

    return datasets_info, prompt_search


def create_search_agent(datasets_info=None):
    model_name = st.session_state.get("model_name", "gpt-3.5-turbo")
    if model_name == "o3-mini":
        llm = ChatOpenAI(api_key=API_KEY, model_name=model_name)
    else:
        llm = ChatOpenAI(api_key=API_KEY, model_name=model_name)

    # Generate dataset description string
    datasets_description = ""
    if datasets_info is not None:
        for i, row in datasets_info.iterrows():
            datasets_description += f"Dataset {i + 1}:\nName: {row['Name']}\nDOI: {row['DOI']}\nDescription: {row['Short Description']}\nParameters: {row['Parameters']}\n\n"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             f"You are a powerful assistant primarily designed to search and retrieve datasets from PANGAEA. Your main goal is to help users find relevant datasets using the search_pg_datasets tool. When a user asks about datasets, always use this tool first to provide the most up-to-date and accurate information.\n\n"
             #f"Here are some datasets returned from the search:\n{datasets_description}"
             "In addition to dataset searches, you have a secondary capability to answer questions about publications related to specific datasets (or in other words what was published based on this dataset). If a user explicitly asks about publications or research findings based on a particular dataset, you can use the answer_publication_questions tool. For example, you can handle queries like 'What was published based on this dataset?' or 'What were the main conclusions from the research using this dataset?'\n\n"
             "Remember:\n"
             "1. Prioritize dataset searches using the search_pg_datasets tool. Make sure that the query you pass to the tool is rephrased so that elastic search gives the best match. Also try not to include words like 'search' and etc. in the search query.\n"
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

# 2. Visualization and Oceanography Tools

class PlotMasterTrackMapArgs(BaseModel):
    dataset_var: str = Field(description="The variable name of the dataset to use (e.g., 'dataset_1', 'dataset_2').")
    main_title: str = Field(description="The main title for the plot.")
    lat_col: str = Field(description="Name of the latitude column.")
    lon_col: str = Field(description="Name of the longitude column.")
    date_col: str = Field(description="Name of the date/time column.")

class TSPlotToolArgs(BaseModel):
    dataset_var: str = Field(description="The variable name of the dataset to use (e.g., 'dataset_1', 'dataset_2').")
    main_title: str = Field(description="The main title for the plot.")
    temperature_col: str = Field(description="Name of the temperature column.")
    salinity_col: str = Field(description="Name of the salinity column.")

# 3. Agent Creation Functions
def create_pandas_agent(user_query, datasets_info):
    if st.session_state.model_name == "o3-mini":
        llm = ChatOpenAI(api_key=API_KEY, model_name=st.session_state.model_name)
    else:
        llm = ChatOpenAI(api_key=API_KEY, model_name=st.session_state.model_name)

    # Assign unique variable names to each dataframe and collect dataframes
    dataset_variables = []
    dataframes = []
    datasets_text = ""  # Initialize datasets_text
    for i, info in enumerate(datasets_info, 1):  # Start enumeration at 1
        var_name = f"df{i}"  # Consistently name as df1, df2, etc.
        dataframes.append(info['dataset'])  # Collect dataframes into a list
        dataset_variables.append(var_name)
        # Build datasets_text
        datasets_text += (
            f"Dataset {i}:\n"  # Adjust index to match variable naming
            f"Variable Name: {var_name}\n"
            f"Name: {info['name']}\n"
            f"Description: {info['description']}\n"
            f"Head of DataFrame (use it only as an example):\n"
            f"{info['df_head']}\n\n"
        )


    # Create a custom system prompt that includes information about each dataframe
    system_prompt = Prompts.generate_pandas_agent_system_prompt(user_query, datasets_text, dataset_variables)

    # Create a ChatPromptTemplate with the system prompt
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="chat_history"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # Create the pandas dataframe agent with the list of dataframes and the chat prompt
    agent_pandas = create_pandas_dataframe_agent(
        llm=llm,
        df=dataframes,  # Pass the list of dataframes
        agent_type=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        return_intermediate_steps=True,
        max_iterations=5,
        early_stopping_method="generate",
        handle_parsing_errors=True,
        #prefix=system_prompt,
        suffix=system_prompt,
        allow_dangerous_code=True,
        chat_prompt=chat_prompt
    )

    return agent_pandas


# Define the function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def reflect_on_image(image_path: str) -> str:
    if not os.path.exists(image_path):
        return f"Error: The file {image_path} does not exist."

    base64_image = encode_image(image_path)

    prompt = """You are a professional reviewer of scientific images. Your task is to provide review and pass it back to the visual creator agent, so that it could improve it. At each step provide idea for improvements (only if necessary). Be sure to be critical and provide a source for improvement. Conduct a quality check of the provided image using the following criteria:

1. Axis and Font Quality: Evaluate the visibility of axes and appropriateness of font size and style. Are the axes clearly visible and labeled? Is the font legible and suitable for the image size?
2. Label Clarity: Assess if labels are well-positioned and not overlapping. Are all labels clearly readable and properly placed?
3. Color Scheme: Analyze the color choices. Is the color scheme appropriate for the data presented? Are the colors distinguishable and not causing visual confusion?
4. Data Representation: Evaluate how well the data is represented. Are data points clearly visible? Is the chosen chart or graph type appropriate for the data?
5. Legend and Scale: Check the presence and clarity of legends and scales. Are they present when necessary and easy to understand?
6. Overall Layout: Assess the overall layout and use of space. Is the image well-balanced and visually appealing?
7. Technical Issues: Identify any technical problems such as pixelation, blurriness, or artifacts that might affect the image quality.
8. Scientific Accuracy: To the best of your ability, comment on whether the image appears scientifically accurate and free from obvious errors or misrepresentations.
9. Check that the figure make sense from an observing human's point of view, for example, if the figure have a variable ‘Depth of water or smth’ it should be on the Y-AXIS and go from surface to depth, so minimum at the top, max depth in the bottom. If there are remarks about these things, severely underestimate the final mark for the figure and force agent to redo the graph, with precise instructions. SUPER IMPORTANT -> IF DEPTH OF WATER OR ANY VERTICAL DIMENSIONS ARE PRESENT, AND THEY ARE ON THE HORIZONTAL X-AXIS, AND NOT ON Y-AXIS, RETURN FIGURE BACK WITH SCORE 1/10, PUNISH SEVERELY FOR THIS! <- SUPER IMPORTANT

Please provide a structured review addressing each of these points. Conclude with an overall assessment of the image quality, highlighting any significant issues or exemplary aspects. Finally, give the image a score out of 10, where 10 is perfect quality and 1 is unusable.
"""
    openai_client = OpenAI(api_key=API_KEY)
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=1000
    )

    return response.choices[0].message.content


# Define the args schema for reflect_on_image
class ReflectOnImageArgs(BaseModel):
    image_path: str = Field(description="The path to the image to reflect on.")

# Define the reflect_on_image tool
reflect_tool = StructuredTool.from_function(
    func=reflect_on_image,
    name="reflect_on_image",
    description="A tool to reflect on an image and provide feedback for improvements.",
    args_schema=ReflectOnImageArgs
)

#Planning tool

class PlanningToolArgs(BaseModel):
    goal: str = Field(
        description="A short statement of the user's main objective or question to create a plan for."
    )
    constraints: List[str] = Field(
        default_factory=list,
        description="Any constraints or conditions to be respected in the plan (e.g., time or resource constraints)."
    )
    user_query: str = Field(
        default="",
        description="The original user query or question that triggered the plan request."
    )
    datasets_summary: str = Field(
        default="",
        description="A concise summary of the current datasets or project context that the plan should consider."
    )


def planning_tool(
    goal: str,
    constraints: List[str],
    user_query: str,
    datasets_summary: str
) -> dict:
    """
    A planning function that uses a ChatCompletion to create a step-by-step plan,
    referencing the user query, constraints, and dataset info for context.
    Returns a dict with at least "messages" so it updates the state in langgraph.
    """

    from langchain_openai import ChatOpenAI
    from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

    # Create a system prompt that instructs the LLM how to create the plan:
    system_prompt = (
        "You are an advanced 'PlanningTool' that must generate a step-by-step plan. "
        "Consider the user’s ultimate goal, the constraints, the original query, and the dataset context. "
        "Respond with a thorough but concise plan that can be used by the system to coordinate tasks."
    )

    # We'll build a user message that includes all relevant info:
    # (goal, constraints, user_query, and the dataset summary).
    user_message = (
        f"Goal: {goal}\n\n"
        f"Constraints: {constraints}\n\n"
        f"User Query: {user_query}\n\n"
        f"Dataset Info:\n{datasets_summary}\n\n"
        "Please produce a plan with carefully enumerated steps."
    )

    # Create an LLM instance
    model_name = st.session_state.get("model_name", "gpt-3.5-turbo")
    if model_name == "o3-mini":
        llm = ChatOpenAI(api_key=API_KEY, model_name=model_name)
    else:
        llm = ChatOpenAI(api_key=API_KEY, model_name=model_name)


    # Construct messages for the chat
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ]

    # Call the LLM
    response = llm(messages)

    # The text of the plan is in response.content
    final_plan_text = response.content

    # Return a dictionary that merges into state["messages"] 
    # (this is how the graph update won't fail with InvalidUpdateError)
    return {
        "messages": [
            AIMessage(content=final_plan_text, name="Planner")
        ]
    }




def install_package(package_name: str, pip_options: str = ""):
    #ALLOWED_PACKAGES = {"matplotlib", "seaborn", "plotly", "pandas", "numpy", "gsw", "scipy"}
    #if package_name not in ALLOWED_PACKAGES:
    #    return f"Installation of package '{package_name}' is not allowed."
    try:
        command = [sys.executable, '-m', 'pip', 'install'] + pip_options.split() + [package_name]
        subprocess.check_call(command)
        return f"Package '{package_name}' installed successfully."
    except Exception as e:
        return f"Failed to install package '{package_name}': {e}"


# Define the args schema for install_package
class InstallPackageArgs(BaseModel):
    package_name: str = Field(description="The name of the package to install.")
    pip_options: str = Field(default="", description="Additional pip options (e.g., '--force-reinstall').")

# Create the install_package_tool
install_package_tool = StructuredTool.from_function(
    func=install_package,
    name="install_package",
    description="Installs a Python package using pip. Use this tool if you encounter a ModuleNotFoundError or need a package that's not installed.",
    args_schema=InstallPackageArgs
)

def get_example_of_visualizations(query: str) -> str:
    """
    Retrieves example visualizations related to the query.

    Parameters:
    - query (str): The user's query about plotting.

    Returns:
    - str: The content of the most relevant example file.
    """
    # Initialize embeddings
    #api_key = st.secrets["general"]["openai_api_key"]
    embeddings = OpenAIEmbeddings(api_key=API_KEY)

    # Load the existing vector store
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory=os.path.join('data', 'examples_database', 'chroma_langchain_notebooks')
    )

    # Perform a similarity search
    results = vector_store.similarity_search_with_score(query, k=1)

    # Extract the most relevant document
    doc, score = results[0]

    # Construct the full path to the txt file
    file_name = doc.metadata['source'].lstrip('./')
    full_path = os.path.join('data', 'examples_database', file_name)

    # Read and return the content of the txt file
    try:
        with open(full_path, 'r', encoding='utf-8') as file:
            content = file.read()
            return content
    except Exception as e:
        logging.error(f"An error occurred while reading the file: {str(e)}")
        return ""  # Return empty string if error occurs


class ExampleVisualizationArgs(BaseModel):
    query: str = Field(description="The user's query about plotting.")

example_visualization_tool = StructuredTool.from_function(
    func=get_example_of_visualizations,
    name="get_example_of_visualizations",
    description="Retrieves example visualization code related to the user's query.",
    args_schema=ExampleVisualizationArgs
)

########################################
# 1) DEFINE THE TOOL FOR LISTING FILES #
########################################

class ListPlottingDataFilesArgs(BaseModel):
    # No arguments needed here if it just lists everything
    dummy: str = Field(default="", description="(No arguments needed)")  

def list_plotting_data_files(dummy: str = "") -> str:
    """
    Lists all files and subdirectories under data/plotting_data.
    Returns a single string containing each path on a new line.
    """
    base_dir = os.path.join("data", "plotting_data")
    all_paths = []

    for root, dirs, files in os.walk(base_dir):
        # Optionally skip hidden dirs/files, etc.
        for filename in files:
            rel_path = os.path.relpath(os.path.join(root, filename), start=base_dir)
            all_paths.append(rel_path)

    if not all_paths:
        return "No files found in data/plotting_data."

    return "Files under data/plotting_data:\n" + "\n".join(all_paths)

list_plotting_data_files_tool = StructuredTool.from_function(
    func=list_plotting_data_files,
    name="list_plotting_data_files",
    description="Lists all files under data/plotting_data directory (including subfolders).",
    args_schema=ListPlottingDataFilesArgs
)



def create_visualization_agent(user_query, datasets_info):
    datasets_text = ""  # Initialize datasets_text
    dataset_variables = []
    datasets = {}
    for i, info in enumerate(datasets_info):
        var_name = f"dataset_{i + 1}"
        datasets[var_name] = info['dataset']
        dataset_variables.append(var_name)
        # Build datasets_text
        datasets_text += (
            f"Dataset {i + 1}:\n"
            f"Variable Name: {var_name}\n"
            f"Name: {info['name']}\n"
            f"Description: {info['description']}\n"
            f"Head of DataFrame (use it only as an example):\n"
            f"{info['df_head']}\n\n"
        )

    # Generate the system prompt using datasets_text
    prompt = Prompts.generate_visualization_agent_system_prompt(user_query, datasets_text, dataset_variables)

    llm = ChatOpenAI(api_key=API_KEY, model_name=st.session_state.model_name)
    repl_tool = CustomPythonREPLTool(datasets=datasets)
    tools_vis = [
        repl_tool,
        reflect_tool,
        install_package_tool,
        example_visualization_tool,
        list_plotting_data_files_tool
    ]
    agent_visualization = create_openai_tools_agent(
        llm,
        tools=tools_vis,
        prompt=ChatPromptTemplate.from_messages(
            [
                ("system", prompt),
                MessagesPlaceholder(variable_name="messages"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ]
        )
    )
    return AgentExecutor(
        agent=agent_visualization,
        tools=tools_vis,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )





def create_hard_coded_visualization_agent(user_query, datasets_info):
    import streamlit as st
    model_name = st.session_state.get("model_name", "gpt-3.5-turbo")
    if model_name == "o3-mini":
        llm = ChatOpenAI(api_key=API_KEY, model_name=model_name)
    else:
        llm = ChatOpenAI(api_key=API_KEY, model_name=model_name)

    # Prepare datasets
    datasets = {}
    datasets_text = ""
    dataset_variables = []
    for i, info in enumerate(datasets_info):
        var_name = f"dataset_{i + 1}"
        datasets[var_name] = info['dataset']
        dataset_variables.append(var_name)
        datasets_text += (
            f"Dataset {i + 1}:\n"
            f"Variable Name: {var_name}\n"
            f"Name: {info['name']}\n"
            f"Description: {info['description']}\n"
            f"Head of DataFrame (select appropriate attributes based on this):\n"
            f"{info['df_head']}\n\n"
        )

    # Generate the system prompt
    system_prompt = Prompts.generate_system_prompt_hard_coded_visualization(user_query, datasets_text, dataset_variables)

    def plot_master_track_map_tool(dataset_var, main_title, lat_col, lon_col, date_col):
        dataset_df = datasets.get(dataset_var)
        if dataset_df is None:
            return {"result": f"Dataset '{dataset_var}' not found."}
        return plot_master_track_map(main_title=main_title, lat_col=lat_col, lon_col=lon_col, date_col=date_col, dataset_df=dataset_df)

    # Define visualization tools
    visualization_functions = [
        StructuredTool.from_function(
            func=plot_master_track_map_tool,
            name="plot_master_track_map_tool",
            description="Plot the master track map using the specified dataset.",
            args_schema=PlotMasterTrackMapArgs
        )
    ]

    # Create the agent with tools and prompt
    agent = create_openai_tools_agent(
        llm,
        tools=visualization_functions,
        prompt=ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="messages"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ]
        )
    )

    return AgentExecutor(agent=agent, tools=visualization_functions)


# Create Oceanographer Agent
def create_oceanographer_agent(user_query, datasets_info):
    import streamlit as st
    model_name = st.session_state.get("model_name", "gpt-3.5-turbo")
    if model_name == "o3-mini":
        llm = ChatOpenAI(api_key=API_KEY, model_name=model_name)
    else:
        llm = ChatOpenAI(api_key=API_KEY, model_name=model_name)

    # Prepare datasets
    datasets = {}
    datasets_text = ""
    dataset_variables = []
    for i, info in enumerate(datasets_info):
        var_name = f"dataset_{i + 1}"
        datasets[var_name] = info['dataset']
        dataset_variables.append(var_name)
        datasets_text += (
            f"Dataset {i + 1}:\n"
            f"Variable Name: {var_name}\n"
            f"Name: {info['name']}\n"
            f"Description: {info['description']}\n"
            f"Head of DataFrame (select appropriate attributes based on this):\n"
            f"{info['df_head']}\n\n"
        )

    # Generate the system prompt
    system_prompt = Prompts.generate_system_prompt_oceanographer(user_query, datasets_text, dataset_variables)

    def plot_ts_diagram_tool(dataset_var, main_title, temperature_col, salinity_col):
        dataset_df = datasets.get(dataset_var)
        if dataset_df is None:
            return {"result": f"Dataset '{dataset_var}' not found."}
        return plot_ts_diagram(main_title=main_title, temperature_col=temperature_col, salinity_col=salinity_col, dataset_df=dataset_df)

    # Define oceanography tools
    oceanography_functions = [
        StructuredTool.from_function(
            func=plot_ts_diagram_tool,
            name="plot_ts_diagram_tool",
            description="Plot TS diagram using the specified dataset.",
            args_schema=TSPlotToolArgs
        )
    ]

    # Create the agent with tools and prompt
    agent = create_openai_tools_agent(
        llm,
        tools=oceanography_functions,
        prompt=ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="messages"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ]
        )
    )

    return AgentExecutor(agent=agent, tools=oceanography_functions)


def initialize_agents(user_query, datasets_info):
    if datasets_info:
        # Create agents
        visualization_agent = create_visualization_agent(
            user_query=user_query,
            datasets_info=datasets_info
        )

        dataframe_agent = create_pandas_agent(
            user_query=user_query,
            datasets_info=datasets_info
        )

        hard_coded_visualization_agent = create_hard_coded_visualization_agent(
            user_query=user_query,
            datasets_info=datasets_info
        )

        oceanographer_agent = create_oceanographer_agent(
            user_query=user_query,
            datasets_info=datasets_info
        )

        return visualization_agent, dataframe_agent, hard_coded_visualization_agent, oceanographer_agent
    else:
        st.warning("No datasets loaded. Please load datasets first.")
        return None, None, None, None


def agent_node(state, agent, name):
    import streamlit as st  # Ensure Streamlit is imported
    logging.debug(f"Entering agent_node for {name}")

    if 'agent_scratchpad' not in state or not isinstance(state['agent_scratchpad'], list):
        state['agent_scratchpad'] = []

    user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    if user_messages:
        last_user_message = user_messages[-1].content
        state['input'] = last_user_message
    else:
        state['input'] = state.get('input', '')

    if 'plot_images' not in state or not isinstance(state['plot_images'], list):
        state['plot_images'] = []

    # Invoke the agent
    result = agent.invoke(state)
    last_message_content = result.get("output", "")
    intermediate_steps = result.get("intermediate_steps", [])
    returned_plot_images = result.get("plot_images", [])  # Gather newly returned images

    # Store intermediate steps
    if 'intermediate_steps' not in st.session_state:
        st.session_state['intermediate_steps'] = []
    st.session_state['intermediate_steps'].extend(intermediate_steps)

    from src.utils import log_history_event
    for step in intermediate_steps:
        action = step[0]
        observation = step[1]
        tool_name = action.tool
        tool_input = action.tool_input
        log_history_event(
            st.session_state,
            "tool_usage",
            {
                "agent_name": name,
                "tool_name": tool_name,
                "tool_input": tool_input,
                "observation": observation
            }
        )

    # If a ModuleNotFoundError was returned
    if name == "VisualizationAgent":
        if isinstance(last_message_content, dict):
            if last_message_content.get("error") == "ModuleNotFoundError":
                missing_module = last_message_content.get("missing_module")
                logging.info(f"Detected missing module: {missing_module}")
                install_result = install_package_tool.run({"package_name": missing_module})
                logging.info(f"Install package result: {install_result}")
                if "successfully" in install_result:
                    retry_result = agent.invoke(state)
                    last_message_content = retry_result.get("output", "")
                else:
                    last_message_content = f"Failed to install the missing package '{missing_module}'. Please install it manually."

    # Check if a new plot path was set in session_state
    new_plot_path = st.session_state.get("new_plot_path")
    logging.info(f"New plot path from session state: {new_plot_path}")
    if new_plot_path:
        if os.path.exists(new_plot_path):
            state["plot_images"].append(new_plot_path)
            st.session_state.new_plot_path = None
            log_history_event(
                st.session_state,
                "plot_generated",  # Use consistent event type
                {
                    "plot_path": new_plot_path,
                    "agent_name": name,
                    "description": f"Plot generated by {name}"
                }
            )
    if new_plot_path:
        log_history_event(
            st.session_state,
            "plot_generated_final",
            {"plot_path": new_plot_path}
        )

    # Combine the newly returned images with state images
    all_plot_images = list(returned_plot_images) + state["plot_images"]

    # Create a new AIMessage with additional info.
    # Note: We add a "plot" field so that it appears in the final JSON.
    ai_message = AIMessage(
        content=last_message_content,
        name=name,
        additional_kwargs={
            "plot_images": all_plot_images,
            "plot": all_plot_images[0] if all_plot_images else None
        }
    )
    state["messages"].append(ai_message)

    # Trim messages if needed
    state["messages"] = state["messages"][-7:]

    if name == "VisualizationAgent":
        state["visualization_agent_used"] = True

    state["last_agent_message"] = last_message_content
    return state



def supervisor_response(state):
    import streamlit as st
    from main import get_datasets_info_for_active_datasets  # Adjust import as needed

    model_name = st.session_state.get("model_name", "gpt-3.5-turbo")
    if model_name == "o3-mini":
        llm = ChatOpenAI(api_key=API_KEY, model_name=model_name)
    else:
        llm = ChatOpenAI(api_key=API_KEY, model_name=model_name)

    # Build dataset context from the active (selected) datasets only.
    active_datasets_info = get_datasets_info_for_active_datasets(st.session_state)
    datasets_text = ""
    if active_datasets_info:
        for i, info in enumerate(active_datasets_info, 1):
            datasets_text += (
                f"Dataset {i}:\n"
                f"Name: {info['name']}\n"
                f"DOI: {info['doi']}\n"
                f"Description: {info['description']}\n"
                f"Parameters: {info.get('parameters', '')}\n\n"
            )
    else:
        datasets_text = "No active dataset selected."

    # Build the system prompt using the active dataset context.
    system_message = (
        "You are a supervisor capable of answering simple questions directly. "
        "If the user's query is basic (e.g., about available analysis), "
        "answer using the selected dataset context below:\n\n"
        f"{datasets_text}\n\n"
        "For complex queries, follow these agent guidelines:\n"
        "- Use VisualizationAgent for general plotting\n"
        "- Use HardCodedVisualizationAgent ONLY for track maps\n"
        "- Use OceanographerAgent ONLY for TS diagrams\n"
        "Format any code in markdown and keep responses concise."
    )

    # Build the complete conversation history.
    # Here we include both human and assistant messages with labels.
    full_history = "\n".join([
        f"{msg.name}: {msg.content}" for msg in state["messages"] if hasattr(msg, "content") and hasattr(msg, "name")
    ])

    prompt = f"{system_message}\n\nConversation history:\n{full_history}"

    # Invoke the LLM with the full conversation context.
    response = llm.invoke([HumanMessage(content=prompt)])

    # Append the supervisor's answer to the state and mark the conversation as finished.
    state["messages"].append(AIMessage(content=response.content, name="Supervisor"))
    state["next"] = "FINISH"
    return state





def create_supervisor_agent(user_query, datasets_info, memory):
    members = ["VisualizationAgent", "DataFrameAgent", "HardCodedVisualizationAgent", "OceanographerAgent"]

    # Prepare datasets_text and dataset_variables
    datasets_text = ""
    dataset_variables = []
    datasets = {}
    for i, info in enumerate(datasets_info):
        var_name = f"df{i}" if i > 0 else "df"
        datasets_text += (
            f"Dataset {i + 1}:\n"
            f"Variable Name: {var_name}\n"
            f"Name: {info['name']}\n"
            f"Description: {info['description']}\n"
            f"Head of DataFrame (use it only as an example):\n"
            f"{info['df_head']}\n\n"
        )
        dataset_variables.append(var_name)
        datasets[var_name] = info['dataset']

    system_prompt_supervisor = (
        f"You are a supervisor tasked with managing a conversation between the following workers: {members}. "
        f"Given the following user request: '{user_query}', determine and instruct the next worker to act. "
        f"Each worker will perform a task and respond with their results and status. "
        f"If the request involves plotting a master track, directly assign the task to the HardCodedVisualizationAgent. "
        f"For TS diagram, assign the task to the OceanographerAgent. The other requests should be handled by the VisualizationAgent. It is extremely important to assign the correct task to the correct agent and use HardCodedVisualizationAgent and OceanographerAgent only for the described cases.\n"
        f"If a meaningful response from the agent has been provided, end the process by returning 'FINISH' and not 'RESPOND' to avoid unnecessary loops.\n"
        f"The dataset info is:\n{datasets_text}\n"
        f"### Agents and Their Capabilities:\n"
        "- **VisualizationAgent:** A major visualization tool to be called. Generates various plots using the dataset with tools like Python_REPL, reflect_on_image, install_package, and get_example_of_visualizations.\n"
        "- **DataFrameAgent:** Performs data analysis and manipulation on the dataset using pandas.\n"
        "- **HardCodedVisualizationAgent:** Only can plot master track map using predefined functions (call only if you are 100% sure that you need a master track map from an expedition; otherwise, call VisualizationAgent).\n"
        "- **OceanographerAgent:** Only can plot TS diagrams (call only if you are 100% sure that you need to create a TS diagram; otherwise, call VisualizationAgent).\n\n"
        f"### Available Tools:\n"
        f"- **Python_REPL:** Executes Python code for data analysis and visualization.\n"
        f"- **reflect_on_image:** Provides feedback on generated images to improve their quality.\n"
        f"- **install_package:** Installs necessary Python packages when encountering missing modules.\n"
        f"- **get_example_of_visualizations:** Retrieves example visualization code related to user queries.\n"
        f"\n"
        f"The datasets are accessible via variables: {', '.join(dataset_variables)}.\n"
    )

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
                        {"enum": ["FINISH", "RESPOND"] + members},
                    ],
                }
            },
            "required": ["next"],
        },
    }

    # Create the supervisor chain
    prompt_supervisor = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt_supervisor),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
            ("system",
             f"Given the conversation above, decide who should act next. Options are: ['FINISH', 'RESPOND'] + {members}.\n"
             "Select 'FINISH' if the last agent has provided a meaningful and complete response to the user's query.\n"
             "Select 'RESPOND' if you need to provide additional information or clarification to the user.\n"
             "Otherwise, select the next agent to act.\n"
             f"The last agent message was: {{last_agent_message}}")
        ]
    ).partial(options=str(["FINISH", "RESPOND"] + members), members=", ".join(members))

    llm_supervisor = ChatOpenAI(api_key=API_KEY, model_name=st.session_state.model_name)

    supervisor_chain = (
        {
            "messages": lambda x: x["messages"],
            "agent_scratchpad": lambda x: x["agent_scratchpad"],
            "last_agent_message": lambda x: x.get("last_agent_message", ""),
        }
        | prompt_supervisor
        | llm_supervisor.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
    )

    # Define the AgentState type
    class AgentState(TypedDict):
        messages: Sequence[BaseMessage]
        next: str
        agent_scratchpad: Sequence[BaseMessage]
        user_query: str
        last_agent_message: str
        plot_images: List[str]
        model_name: str

    # Create the workflow graph
    workflow = StateGraph(AgentState)
    visualization_agent, dataframe_agent, hard_coded_visualization_agent, oceanographer_agent = initialize_agents(
        user_query, datasets_info
    )

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
        workflow.add_node("supervisor_response", supervisor_response)

        # Connect agents to the supervisor
        for member in members:
            workflow.add_edge(member, "supervisor")

        # Define the conditional map for routing
        conditional_map = {k: k for k in members}
        conditional_map["FINISH"] = END
        conditional_map["RESPOND"] = "supervisor_response"
        workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
        workflow.set_entry_point("supervisor")

        #memory = MemorySaver()
        # Compile the workflow into a graph
        graph = workflow.compile(checkpointer=memory)

        return graph
    else:
        return None
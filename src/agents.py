# src/agents.py
import base64
import os
import logging
import functools
import subprocess
import sys
import uuid
import json
import re
from io import StringIO
from typing import List, Annotated, Sequence, TypedDict, Dict
import operator
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
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
from .prompts import Prompts
from .utils import generate_unique_image_path, escape_curly_braces
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
        - Path variables for each dataset (dataset_1_path, dataset_2_path, etc.)
        - A dynamically generated plot_path
        
        If a figure is saved to plot_path, a "plot_generated" event will be logged.
        
        Returns:
            dict: Results including output and paths to any generated plots
        """
        import streamlit as st
        import matplotlib.pyplot as plt
        import pandas as pd
        import xarray as xr
        import os
        import logging
        from io import StringIO
        from src.utils import log_history_event, generate_unique_image_path

        # Prepare local context with necessary packages
        local_context = {
            "st": st, 
            "plt": plt, 
            "pd": pd,
            "xr": xr,
            "os": os
        }

        # Inject the user's datasets
        local_context.update(self._datasets)
        
        # Add dataset path variables for any string paths (sandbox directories)
        for key, value in self._datasets.items():
            if isinstance(value, str) and os.path.isdir(value):
                # Create a path variable like dataset_1_path for dataset_1
                path_var_name = f"{key}_path"
                # Use the absolute path with consistent slash direction
                abs_path = os.path.abspath(value).replace('\\', '/')
                local_context[path_var_name] = abs_path
                
                # Also log which path variables are available
                logging.info(f"Added path variable {path_var_name} = {abs_path}")

        # Generate a unique file path for the plot
        plot_path = generate_unique_image_path()
        local_context['plot_path'] = plot_path
        
        # Log the code being executed for debugging
        logging.info(f"Executing code with available variables: {list(local_context.keys())}")
        logging.info(f"Code to execute:\n{query}")

        # Redirect stdout to capture output
        old_stdout = sys.stdout
        redirected_output = StringIO()
        sys.stdout = redirected_output

        try:
            # Execute user code
            exec(query, local_context)
            output = redirected_output.getvalue()
            
            # Check if plt.savefig was used or figure was created
            plot_generated = False
            if os.path.exists(plot_path):
                plot_generated = True
                
                # Store the path in session state for UI to display
                st.session_state.saved_plot_path = plot_path
                st.session_state.plot_image = plot_path
                st.session_state.new_plot_path = plot_path
                st.session_state.new_plot_generated = True
                
                # Log the plot generation
                logging.info(f"Plot generated successfully and saved to: {plot_path}")
                
                log_history_event(
                    st.session_state,
                    "plot_generated",
                    {
                        "plot_path": plot_path,
                        "agent": "VisualizationAgent",
                        "description": "Python_REPL Generated Plot",
                        "content": query  # Store the actual code used
                    }
                )
            else:
                # If no plot was generated, check if we need to give feedback about path issues
                if '/mnt/data' in query:
                    error_msg = "ERROR: Detected invalid path '/mnt/data'. You must use the provided dataset path variables."
                    logging.warning(error_msg)
                    output += f"\n\n{error_msg}"
                
                if "plot_path" in query and "plt.savefig" not in query:
                    warning_msg = "WARNING: You defined plot_path but didn't use plt.savefig(plot_path). Plots won't be displayed."
                    logging.warning(warning_msg)
                    output += f"\n\n{warning_msg}"

            return {
                "result": f"Execution completed. Plot saved at: {plot_path if plot_generated else 'No plot generated'}",
                "output": output,
                "plot_images": [plot_path] if plot_generated else []
            }
        except Exception as e:
            logging.error(f"Error during code execution: {e}")
            error_output = f"ERROR: {str(e)}\n\n"
            
            # Add helpful error diagnostics
            if "FileNotFoundError" in str(e):
                error_output += "This looks like a path problem. Please check:\n"
                error_output += "1. You're using the exact path variables (dataset_1_path, etc.)\n"
                error_output += "2. You're using os.path.join() to combine paths\n"
                error_output += "3. The file you're trying to access actually exists\n\n"
                error_output += "Available path variables:\n"
                for key in local_context:
                    if key.endswith('_path'):
                        error_output += f"- {key}: {local_context[key]}\n"
            
            elif "ModuleNotFoundError" in str(e):
                module_name = str(e).split("No module named ")[-1].strip("'")
                error_output += f"Missing module: {module_name}\\n"
                error_output += "You can install it using the install_package tool."
            
            elif "name 'data' is not defined" in str(e) or "not defined" in str(e):
                var_name = str(e).split("'")[1] if "'" in str(e) else "unknown"
                error_output += f"Variable '{var_name}' is not defined. Available variables:\n"
                error_output += f"- Dataset variables: {[k for k in self._datasets.keys()]}\n"
                error_output += f"- Path variables: {[k for k in local_context.keys() if k.endswith('_path')]}\n"
                error_output += "Make sure you're using the correct variable names."
            
            return {
                "error": "ExecutionError",
                "message": error_output,
                "output": redirected_output.getvalue()
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

    prompt = """You are a professional reviewer of scientific images. Your task is to provide feedback to the visual creator agent so they can improve their visualization. Provide constructive criticism and specific improvement suggestions where necessary. Evaluate the provided image using the following criteria:

1. Axis and Font Quality: Evaluate the visibility of axes and appropriateness of font size and style. Are the axes clearly visible and labeled? Is the font legible and suitable for the image size?
2. Label Clarity: Assess if labels are well-positioned and not overlapping. Are all labels clearly readable and properly placed?
3. Color Scheme: Analyze the color choices. Is the color scheme appropriate for the data presented? Are the colors distinguishable and not causing visual confusion?
4. Data Representation: Evaluate how well the data is represented. Are data points clearly visible? Is the chosen chart or graph type appropriate for the data?
5. Legend and Scale: Check the presence and clarity of legends and scales. Are they present when necessary and easy to understand?
6. Overall Layout: Assess the overall layout and use of space. Is the image well-balanced and visually appealing?
7. Technical Issues: Identify any technical problems such as pixelation, blurriness, or artifacts that might affect the image quality.
8. Scientific Accuracy: To the best of your ability, comment on whether the image appears scientifically accurate and free from obvious errors or misrepresentations.
9. **Convention Adherence**: Verify that the figure follows scientific conventions. For example, when depicting variables like 'Depth of water' or other vertical dimensions, these should appear on the Y-axis with minimum values at the top and maximum depth at the bottom. This is a critically important scientific convention - if depth/vertical dimensions are incorrectly presented on the horizontal X-axis, assign a significantly lower score (1/10) and provide clear instructions for correction.

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

# New Planning Tool
class PlanningToolArgs(BaseModel):
    user_query: str = Field(description="The user's original query")
    conversation_history: str = Field(description="The conversation history so far")
    available_agents: List[str] = Field(description="List of available agent types")
    current_plan: str = Field(description="The current plan, if any exists")
    datasets_info: str = Field(description="Information about available datasets")

def planning_tool(user_query: str, conversation_history: str, available_agents: List[str], 
                  current_plan: str, datasets_info: str) -> str:
    """
    A tool for creating or updating a plan based on user query and conversation.
    Returns a JSON string containing task steps with assigned agents and status.
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage

    model_name = st.session_state.get("model_name", "gpt-4o")
    llm = ChatOpenAI(api_key=API_KEY, model_name=model_name)
    
    # Optimized system message to create simpler, more efficient plans
    system_message = """
    You are a Planning Tool that creates MINIMAL, EFFICIENT task plans for data analysis workflows.
    Based on the user query, conversation history, and available agents, create a plan with these components:
    1. A list of tasks needed to address the user query
    2. Assignment of each task to the appropriate agent type
    3. Status tracking (pending, in_progress, completed, failed)
    
    CRITICAL PLANNING GUIDELINES:
    - SIMPLICITY IS KEY: Create the MINIMUM number of tasks needed (1-2 tasks if possible)
    - AVOID TASK SPLITTING: For simple queries that can be solved in one step, create JUST ONE task
    - DIRECT IMPLEMENTATION: For basic operations like counting, finding maximums, or calculating statistics, use a SINGLE task
    
    Examples of queries that should be ONE TASK:
    - "What is the most common species?" → ONE task for DataFrameAgent
    - "Calculate the average temperature" → ONE task for DataFrameAgent
    - "Count how many records are in the dataset" → ONE task for DataFrameAgent
    - "Show the distribution of species" → ONE task for VisualizationAgent
    
    FORMAT YOUR RESPONSE AS A VALID JSON ARRAY where each item has:
    - "task": task description (be specific and include the complete action needed)
    - "agent": agent name (must be one from the available_agents list)
    - "status": "pending" (for new tasks), "in_progress", "completed", or "failed"
    
    AGENT SELECTION GUIDELINES:
    - DataFrameAgent: Use for data analysis, filtering, counting, statistics, and identifying patterns
    - VisualizationAgent: Use ONLY when the user explicitly requests a plot or when visualization would significantly enhance understanding
    """
    
    # Set current_plan to empty array if it's not provided
    if not current_plan or current_plan.strip() == "":
        current_plan = "[]"
    
    human_message = f"""
    USER QUERY: {user_query}
    
    AVAILABLE AGENTS: {available_agents}
    
    DATASETS INFO: {datasets_info}
    
    CURRENT PLAN (if any): {current_plan}
    
    CONVERSATION HISTORY: {conversation_history}
    
    Create a MINIMAL EFFICIENT plan that addresses the user's query using the available agents.
    For simple analysis like finding frequency, counting occurrences, or basic statistics, use JUST ONE TASK with the specific action.
    Return only the JSON array with no additional text.
    """
    
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=human_message)
    ]
    
    response = llm.invoke(messages)
    return response.content

planning_tool = StructuredTool.from_function(
    func=planning_tool,
    name="create_or_update_plan",
    description="Creates or updates a plan for addressing the user's query with a sequence of tasks assigned to specific agents",
    args_schema=PlanningToolArgs
)

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
    import os
    import pandas as pd
    import xarray as xr

    # Initialize variables
    datasets_text = ""
    dataset_variables = []
    datasets = {}
    
    # Create a section with crystal clear path instructions
    uuid_paths = "### ⚠️ CRITICAL: EXACT DATASET PATHS - MUST USE THESE EXACTLY AS SHOWN ⚠️\n"
    uuid_paths += "The following paths contain unique IDs that MUST be used with os.path.join():\n\n"
    
    # First list all datasets with their paths
    for i, info in enumerate(datasets_info):
        var_name = f"dataset_{i + 1}"
        datasets[var_name] = info['dataset']  # Sandbox path or dataset object
        dataset_variables.append(var_name)
        
        # Add the FULL UUID PATH for each dataset
        if isinstance(info['dataset'], str) and os.path.isdir(info['dataset']):
            # Get the absolute path and ensure it has forward slashes for consistency
            full_uuid_path = os.path.abspath(info['dataset']).replace('\\', '/')
            
            # Create the path variable assignment with proper escaping
            uuid_paths += f"# Dataset {i+1}: {info['name']}\n"
            uuid_paths += f"{var_name}_path = r'{full_uuid_path}'\n\n"
            
            # Check if the directory exists and list its contents
            if os.path.exists(full_uuid_path):
                try:
                    files = os.listdir(full_uuid_path)
                    uuid_paths += f"# Files available in {var_name}_path:\n"
                    uuid_paths += f"# {', '.join(files)}\n\n"
                    
                    # Now add dataset-specific examples based on actual files
                    uuid_paths += f"# DATASET-SPECIFIC WORKING CODE EXAMPLES FOR {var_name}:\n"
                    
                    # Example for listing files
                    uuid_paths += f"# Example: List all files in {var_name}_path\n"
                    uuid_paths += f"import os\n"
                    uuid_paths += f"files = os.listdir({var_name}_path)\n"
                    uuid_paths += f"print(f\"Files in {var_name}:\")\n"
                    uuid_paths += f"for file in files:\n"
                    uuid_paths += f"    print(f\"  - {{file}}\")\n\n"
                    
                    # Add file-specific examples
                    if any(f.endswith('.csv') for f in files):
                        csv_file = next(f for f in files if f.endswith('.csv'))
                        uuid_paths += f"# Example: Load CSV file '{csv_file}' from {var_name}_path\n"
                        uuid_paths += f"import pandas as pd\n"
                        uuid_paths += f"csv_path = os.path.join({var_name}_path, '{csv_file}')\n"
                        uuid_paths += f"df = pd.read_csv(csv_path)\n"
                        uuid_paths += f"print(df.head())\n\n"
                    
                    if any(f.endswith(('.nc', '.cdf', '.netcdf')) for f in files):
                        nc_file = next(f for f in files if f.endswith(('.nc', '.cdf', '.netcdf')))
                        uuid_paths += f"# Example: Load netCDF file '{nc_file}' from {var_name}_path\n"
                        uuid_paths += f"import xarray as xr\n"
                        uuid_paths += f"nc_path = os.path.join({var_name}_path, '{nc_file}')\n"
                        uuid_paths += f"ds = xr.open_dataset(nc_path)\n"
                        uuid_paths += f"print(ds)\n\n"
                    
                    # Plotting example specific to this dataset
                    if any(f.endswith('.csv') for f in files):
                        csv_file = next(f for f in files if f.endswith('.csv'))
                        uuid_paths += f"# Example: Create plot using data from {var_name}_path\n"
                        uuid_paths += f"import pandas as pd\n"
                        uuid_paths += f"import matplotlib.pyplot as plt\n"
                        uuid_paths += f"csv_path = os.path.join({var_name}_path, '{csv_file}')\n"
                        uuid_paths += f"df = pd.read_csv(csv_path)\n"
                        uuid_paths += f"plt.figure(figsize=(10, 6))\n"
                        uuid_paths += f"# Replace 'column_name' with an actual column from your data\n"
                        uuid_paths += f"plt.plot(df.index, df.iloc[:, 0])\n"
                        uuid_paths += f"plt.title('Data from {info['name']}')\n"
                        uuid_paths += f"plt.savefig(plot_path)  # ALWAYS use this plot_path variable\n\n"
                    
                except Exception as e:
                    uuid_paths += f"# Error listing files: {str(e)}\n\n"
            
            uuid_paths += f"# ⚠️ WARNING: ALWAYS USE THE EXACT PATH WITH os.path.join({var_name}_path, 'filename')! ⚠️\n\n"
        else:
            # For non-directory datasets (like pandas DataFrames)
            uuid_paths += f"# Dataset {i+1}: {info['name']} (in-memory dataset, not a directory)\n"
            uuid_paths += f"# Access this dataset directly with the variable name '{var_name}'\n\n"
    
    # Global warning about path handling
    uuid_paths += "# ⚠️ CRITICAL WARNINGS ⚠️\n"
    uuid_paths += "# 1. NEVER use '/mnt/data/...' or similar paths - they DO NOT EXIST and WILL CAUSE ERRORS\n"
    uuid_paths += "# 2. ALWAYS use the exact dataset_X_path variables shown above\n"
    uuid_paths += "# 3. ALWAYS check which files exist before trying to read them\n\n"
    
    # Continue with standard dataset info after the path examples
    for i, info in enumerate(datasets_info):
        var_name = f"dataset_{i + 1}"
        datasets_text += (
            f"Dataset {i + 1}:\n"
            f"Variable Name: {var_name}\n"
            f"Name: {info['name']}\n"
            f"Description: {info['description']}\n"
            f"Type: {info['data_type']}\n"
            f"Sample Data: {info['df_head']}\n\n"
        )
    
    # Put the UUID paths section at the beginning of the datasets_text
    datasets_text = uuid_paths + datasets_text
    
    # Generate the prompt with the modified datasets_text
    from src.prompts import Prompts
    prompt = Prompts.generate_visualization_agent_system_prompt(user_query, datasets_text, dataset_variables)
  
    # Initialize the LLM
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(api_key=API_KEY, model_name=st.session_state.model_name)

    # Create the CustomPythonREPLTool with sandbox paths
    from src.agents import CustomPythonREPLTool, reflect_tool, install_package_tool
    from src.agents import example_visualization_tool, list_plotting_data_files_tool
    repl_tool = CustomPythonREPLTool(datasets=datasets)

    # Define the tools available to the agent
    tools_vis = [
        repl_tool,
        reflect_tool,
        install_package_tool,
        example_visualization_tool,
        list_plotting_data_files_tool
    ]

    # Create the agent with the updated prompt and tools
    from langchain.agents import create_openai_tools_agent
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    agent_visualization = create_openai_tools_agent(
        llm,
        tools=tools_vis,
        prompt=ChatPromptTemplate.from_messages(
            [
                ("system", prompt),
                ("user", "{input}"),  # THIS LINE IS KEY - ensures task info is passed
                MessagesPlaceholder(variable_name="messages"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ]
        )
    )

    # Create the agent executor
    from langchain.agents import AgentExecutor
    return AgentExecutor(
        agent=agent_visualization,
        tools=tools_vis,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )


def initialize_agents(user_query, datasets_info):
    if datasets_info:
        # Create agents
        visualization_agent = create_visualization_agent(
            user_query=user_query,
            datasets_info=datasets_info
        )

        # Only create DataFrameAgent if there are pandas DataFrames
        dataframe_agent = None
        dataframe_datasets = [info for info in datasets_info if isinstance(info['dataset'], pd.DataFrame)]
        if dataframe_datasets:
            dataframe_agent = create_pandas_agent(
                user_query=user_query,
                datasets_info=dataframe_datasets
            )
            logging.info(f"DataFrameAgent initialized with {len(dataframe_datasets)} DataFrames")
        else:
            logging.info("No pandas DataFrames available; skipping DataFrameAgent initialization")

        return visualization_agent, dataframe_agent
    else:
        st.warning("No datasets loaded. Please load datasets first.")
        logging.warning("No datasets provided to initialize_agents")
        return None, None


def agent_node(state, agent, name):
    import time
    import os
    logging.info(f"Entering agent_node for {name}")
    from src.utils import log_history_event, update_thinking_log
    
    # Generate unique IDs for this agent execution
    agent_id = str(uuid.uuid4())
    agent_start_time = time.time()
    
    # Add thinking log entry for agent start
    update_thinking_log(
        st.session_state, 
        f"Starting processing with {name}", 
        agent_name=name,
        step_id=agent_id,
        start_time=agent_start_time
    )
    st.session_state.processing = True
    
    if 'agent_scratchpad' not in state or not isinstance(state['agent_scratchpad'], list):
        state['agent_scratchpad'] = []

    user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    if user_messages:
        last_user_message = user_messages[-1].content
        state['input'] = last_user_message
    else:
        state['input'] = state.get('input', '')
        
    # Find and include the task from the plan for this agent
    current_task = None
    for task in state.get('plan', []):
        if task.get('agent') == name and task.get('status') in ['in_progress', 'pending']:
            current_task = task
            break
            
    if current_task:
        task_description = current_task.get('task', '')
        # Only use the task description without the original user query
        state['input'] = f"TASK: {task_description}"
        logging.info(f"Modified input for {name} to include ONLY task: {task_description}")

    if 'plot_images' not in state or not isinstance(state['plot_images'], list):
        state['plot_images'] = []

    # Log that the agent is thinking
    thinking_id = str(uuid.uuid4())
    update_thinking_log(
        st.session_state, 
        "Thinking...", 
        agent_name=name,
        parent_id=agent_id,
        step_id=thinking_id,
        start_time=time.time()
    )
    
    # Invoke the agent
    # Add a placeholder for 'file' if the agent is VisualizationAgent
    if name == "VisualizationAgent" and 'file' not in state:
        state['file'] = None  # or provide an appropriate default value
    llm_start_time = time.time()
    result = agent.invoke(state)
    llm_end_time = time.time()
    
    # Log thinking completion
    update_thinking_log(
        st.session_state, 
        "Completed thinking", 
        agent_name=name,
        parent_id=agent_id,
        step_id=thinking_id,
        start_time=llm_start_time,
        end_time=llm_end_time
    )
    
    last_message_content = result.get("output", "")
    intermediate_steps = result.get("intermediate_steps", [])
    returned_plot_images = result.get("plot_images", [])

    if 'intermediate_steps' not in st.session_state:
        st.session_state['intermediate_steps'] = []
    st.session_state['intermediate_steps'].extend(intermediate_steps)
    logging.info(f"Stored {len(intermediate_steps)} intermediate steps for {name}")

    # Log each step to the thinking log with hierarchy
    for i, step in enumerate(intermediate_steps):
        action = step[0]
        observation = step[1]
        tool_name = action.tool
        tool_input = action.tool_input
        
        # Create a unique ID for this tool execution
        tool_id = str(uuid.uuid4())
        tool_start_time = time.time() 
        
        # Add to thinking log
        input_summary = str(tool_input)
        if len(input_summary) > 100:
            input_summary = input_summary[:100] + "..."
            
        update_thinking_log(
            st.session_state,
            f"Input: {input_summary}",
            tool_name=tool_name,
            parent_id=agent_id,
            step_id=tool_id,
            start_time=tool_start_time
        )
        
        # Summarize the observation for the thinking log
        obs_summary = str(observation)
        if len(obs_summary) > 150:
            obs_summary = obs_summary[:150] + "..."
        
        # Add small delay to simulate execution time
        tool_end_time = time.time()
        
        update_thinking_log(
            st.session_state,
            f"Result: {obs_summary}",
            tool_name=tool_name,
            parent_id=tool_id,  # Make this a child of the tool input
            step_id=str(uuid.uuid4()),
            start_time=tool_start_time,
            end_time=tool_end_time
        )
        
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
        logging.info(f"Logged tool usage for {name}, tool: {tool_name}")

    # Handle special cases like missing modules
    if name == "VisualizationAgent":
        if isinstance(last_message_content, dict):
            if last_message_content.get("error") == "ModuleNotFoundError":
                missing_module = last_message_content.get("missing_module")
                logging.info(f"Detected missing module: {missing_module}")
                
                install_id = str(uuid.uuid4())
                install_start = time.time()
                
                update_thinking_log(
                    st.session_state, 
                    f"Installing missing module: {missing_module}", 
                    agent_name=name,
                    parent_id=agent_id,
                    step_id=install_id,
                    start_time=install_start
                )
                
                install_result = install_package_tool.run({"package_name": missing_module})
                install_end = time.time()
                
                update_thinking_log(
                    st.session_state, 
                    f"Install result: {install_result}", 
                    agent_name=name,
                    parent_id=install_id,
                    step_id=str(uuid.uuid4()),
                    start_time=install_start,
                    end_time=install_end
                )
                
                logging.info(f"Install package result: {install_result}")
                if "successfully" in install_result:
                    update_thinking_log(
                        st.session_state, 
                        f"Successfully installed {missing_module}, retrying", 
                        agent_name=name,
                        parent_id=agent_id,
                        step_id=str(uuid.uuid4())
                    )
                    retry_result = agent.invoke(state)
                    last_message_content = retry_result.get("output", "")
                else:
                    update_thinking_log(
                        st.session_state, 
                        f"Failed to install {missing_module}", 
                        agent_name=name,
                        parent_id=agent_id,
                        step_id=str(uuid.uuid4())
                    )
                    last_message_content = f"Failed to install the missing package '{missing_module}'. Please install it manually."

    # Handle plot generation - UPDATED SECTION FOR FIXING FIGURE SAVING/DISPLAY
    new_plot_path = st.session_state.get("new_plot_path")
    logging.info(f"New plot path from session state: {new_plot_path}")
    
    # Add validation to ensure the plot path is valid and exists
    if new_plot_path and os.path.exists(new_plot_path):
        plot_id = str(uuid.uuid4())
        plot_time = time.time()
        
        # Create a log for the plot generation
        update_thinking_log(
            st.session_state, 
            f"Generated plot: {os.path.basename(new_plot_path)}", 
            agent_name=name,
            parent_id=agent_id,
            step_id=plot_id,
            start_time=plot_time,
            end_time=time.time()
        )
        
        # Ensure the plot is added to the state's plot_images if not already there
        if new_plot_path not in state["plot_images"]:
            state["plot_images"].append(new_plot_path)
            st.session_state.new_plot_generated = True
        
        # Log the event for history tracking
        log_history_event(
            st.session_state,
            "plot_generated",
            {
                "plot_path": new_plot_path,
                "agent_name": name,
                "description": f"Plot generated by {name}"
            }
        )
        logging.info(f"Logged plot generation for {name}, path: {new_plot_path}")
    
    # Clear the new_plot_path to prevent duplicating the same plot
    st.session_state.new_plot_path = None
    
    # Finalize the agent execution
    # Combine plots from returned_plot_images and state["plot_images"], ensuring no duplicates
    all_plot_images = []
    for img_path in returned_plot_images:
        if os.path.exists(img_path) and img_path not in all_plot_images:
            all_plot_images.append(img_path)
    
    for img_path in state["plot_images"]:
        if os.path.exists(img_path) and img_path not in all_plot_images:
            all_plot_images.append(img_path)
    
    # Create the AI message with plot info
    ai_message = AIMessage(
        content=last_message_content,
        name=name,
        additional_kwargs={
            "plot_images": all_plot_images,
            "plot": all_plot_images[0] if all_plot_images else None
        }
    )
    state["messages"].append(ai_message)
    logging.info(f"Appended AI message for {name} with {len(all_plot_images)} plot images")

    # Trim messages if needed
    state["messages"] = state["messages"][-10:]
    
    # Store visualization state clearly
    if name == "VisualizationAgent":
        state["visualization_agent_used"] = True
        
    # Ensure last agent message is clearly stored
    state["last_agent_message"] = last_message_content
    
    # Add completion note to thinking log
    agent_end_time = time.time()
    update_thinking_log(
        st.session_state, 
        f"Completed processing with {name}", 
        agent_name=name,
        step_id=agent_id,
        start_time=agent_start_time,
        end_time=agent_end_time
    )
    
    logging.info(f"Completed agent_node for {name}")
    return state

def supervisor_response(state):
    """
    Responsible for generating the final response after all agents have contributed.
    Crucially maintains access to the conversation history and previous agent outputs.
    Includes extensive debugging to troubleshoot dataset parameter passing issues.
    """
    import streamlit as st
    import logging
    import traceback
    import json
    from main import get_datasets_info_for_active_datasets
    
    # ===== Begin Extended Debugging =====
    logging.info("\n==================== SUPERVISOR RESPONSE DEBUGGING ====================")
    logging.info(f"Entering supervisor_response with state keys: {list(state.keys())}")
    logging.info(f"Messages count: {len(state.get('messages', []))}")
    logging.info(f"Last agent message: {state.get('last_agent_message', 'None')}")
    logging.info(f"Current plot_images: {state.get('plot_images', [])}")
    
    # Trace stack to find call origin
    logging.info("Call stack trace:")
    stack_trace = traceback.format_stack()
    for line in stack_trace[:-1]:  # Exclude the current frame
        logging.info(line.strip())
    
    # Get model name from session state
    model_name = st.session_state.get("model_name", "gpt-4o")
    logging.info(f"Using model: {model_name}")
    
    # Debug session state keys
    session_state_keys = list(st.session_state.keys())
    logging.info(f"Available session_state keys: {session_state_keys}")
    
    # ===== Extract datasets info with extensive debugging =====
    try:
        # Debug the datasets retrieval function
        logging.info("Attempting to get active datasets info...")
        active_datasets_info = get_datasets_info_for_active_datasets(st.session_state)
        
        # Verify what we received
        logging.info(f"Type of active_datasets_info: {type(active_datasets_info)}")
        logging.info(f"Length of active_datasets_info: {len(active_datasets_info) if isinstance(active_datasets_info, (list, dict)) else 'Not a collection'}")
        
        # Try to sample the first dataset if available
        if isinstance(active_datasets_info, list) and active_datasets_info:
            sample_keys = list(active_datasets_info[0].keys())
            logging.info(f"Sample dataset keys: {sample_keys}")
    except Exception as e:
        logging.error(f"CRITICAL ERROR retrieving datasets: {str(e)}")
        logging.error(traceback.format_exc())
        active_datasets_info = []  # Fallback to empty list
    
    # ===== Extract and validate all critical components =====
    
    # 1. Extract visualization agent outputs and plot information
    visualization_used = state.get("visualization_agent_used", False)
    plot_images = state.get("plot_images", [])
    logging.info(f"Visualization used: {visualization_used}, Plot images count: {len(plot_images)}")
    
    # 2. Get conversation history with validation
    all_messages = state.get("messages", [])
    logging.info(f"Total message count: {len(all_messages)}")
    
    # Debug message types
    message_types = {}
    for msg in all_messages:
        msg_type = type(msg).__name__
        if msg_type not in message_types:
            message_types[msg_type] = 0
        message_types[msg_type] += 1
    logging.info(f"Message types: {message_types}")
    
    # 3. Extract visualization messages specifically
    visualization_messages = [msg for msg in all_messages 
                             if hasattr(msg, 'name') and msg.name == "VisualizationAgent"]
    logging.info(f"Visualization messages count: {len(visualization_messages)}")
    
    # 4. Extract latest user query with validation
    user_messages = [msg for msg in all_messages if not hasattr(msg, 'name') or not msg.name]
    if user_messages:
        latest_user_query = user_messages[-1].content
        logging.info(f"Latest user query: '{latest_user_query[:50]}...'")
    else:
        latest_user_query = state.get("input", "No query found")
        logging.info(f"No user messages found, using state input: '{latest_user_query[:50]}...'")
    
    # ===== Format datasets information with deep validation =====
    datasets_text = ""
    if active_datasets_info:
        logging.info("Formatting datasets information...")
        for i, info in enumerate(active_datasets_info, 1):
            # Validate each required field exists
            required_fields = ['name', 'description', 'data_type']
            missing_fields = [field for field in required_fields if field not in info]
            
            if missing_fields:
                logging.warning(f"Dataset {i} missing fields: {missing_fields}")
            
            # Get DOI with fallbacks and validation
            doi = info.get('doi', 'Not available')
            
            # Build dataset text with explicit format
            current_dataset = (
                f"Dataset {i}:\n"
                f"Name: {info.get('name', 'Unknown')}\n"
                f"DOI: {doi}\n"
                f"Description: {info.get('description', 'No description available')}\n"
                f"Type: {info.get('data_type', 'Unknown type')}\n"
            )
            datasets_text += current_dataset
            
            # Log what we're adding
            logging.info(f"Added dataset {i}: {info.get('name', 'Unknown')}")
    else:
        datasets_text = "No active dataset selected."
        logging.warning("No active datasets found - using empty placeholder")
    
    # Log the final datasets_text for debugging
    logging.info(f"Final datasets_text length: {len(datasets_text)}")
    logging.info(f"datasets_text preview: {datasets_text[:200]}...")
    
    # ===== Format visualization information =====
    visualization_info = ""
    if visualization_used or visualization_messages:
        visualization_info = "The VisualizationAgent was used in this conversation.\n"
        if visualization_messages:
            recent_vis_content = visualization_messages[-1].content
            visualization_info += f"Most recent visualization analysis: {recent_vis_content[:100]}...\n"
        if plot_images:
            visualization_info += f"Plots were generated and are available at: {', '.join(plot_images)}\n"
    
    logging.info(f"Visualization info length: {len(visualization_info)}")
    
    # ===== Get agent context =====
    last_agent_message = state.get("last_agent_message", "")
    logging.info(f"Last agent message length: {len(last_agent_message)}")
    
    # ===== Construct system message with debugging =====
    try:
        # Create a direct substitution in the system message (not using f-string with placeholders)
        system_message = f"""You are a supervisor capable of answering simple questions directly. If the user's query is basic (e.g., about available analysis), answer using the selected dataset context below:

{datasets_text}

{visualization_info}
Always address the latest user query directly, even if it is similar to previous queries. You can acknowledge repetition (e.g., 'As I mentioned earlier...') and reference previous answers if relevant, but ensure each query receives a clear and complete response. For complex queries, follow these agent guidelines:
- Use VisualizationAgent for general plotting
Format any code in markdown and keep responses concise.

The last agent that processed this request said: {last_agent_message}

Latest user query: {latest_user_query}

Please provide a response to the latest user query, taking into account the conversation history."""
        
        logging.info(f"System message constructed successfully, length: {len(system_message)}")
        
        # Check for template literals that weren't substituted
        if "{datasets_text}" in system_message:
            logging.error("ERROR: '{datasets_text}' template literal found in system message!")
        if "{visualization_info}" in system_message:
            logging.error("ERROR: '{visualization_info}' template literal found in system message!")
        
        # Log the first part of the system message
        logging.info(f"System message preview: {system_message[:300]}...")
        
    except Exception as e:
        logging.error(f"Error building system message: {str(e)}")
        logging.error(traceback.format_exc())
        # Emergency fallback
        system_message = "You are a supervisor assistant. Please respond to the latest query as best you can."
    
    # ===== Build conversation history for context =====
    try:
        full_history = "\n".join([
            f"{msg.name if hasattr(msg, 'name') else 'User'}: {msg.content}" 
            for msg in all_messages 
            if hasattr(msg, "content")
        ])
        
        logging.info(f"Built conversation history, length: {len(full_history)}")
        logging.info(f"History preview: {full_history[:200]}...")
        
    except Exception as e:
        logging.error(f"Error building conversation history: {str(e)}")
        logging.error(traceback.format_exc())
        full_history = "Error retrieving conversation history."
    
    # ===== Prepare final prompt and invoke LLM =====
    prompt = f"{system_message}\n\nConversation history:\n{full_history}"
    logging.info(f"Final prompt length: {len(prompt)}")
    
    # Log key statistics about the prompt to diagnose issues
    prompt_components = {
        "system_message": len(system_message),
        "conversation_history": len(full_history),
        "datasets_text": len(datasets_text),
        "visualization_info": len(visualization_info),
        "last_agent_message": len(last_agent_message),
        "latest_user_query": len(latest_user_query)
    }
    logging.info(f"Prompt component lengths: {json.dumps(prompt_components)}")
    
    # Invoke the LLM with the full context
    try:
        llm = ChatOpenAI(api_key=API_KEY, model_name=model_name)
        response = llm.invoke([HumanMessage(content=prompt)])
        logging.info(f"LLM response received, length: {len(response.content)}")
        logging.info(f"Response preview: {response.content[:200]}...")
    except Exception as e:
        logging.error(f"Error invoking LLM: {str(e)}")
        logging.error(traceback.format_exc())
        response_content = "I encountered an error processing your request. Please try again."
        response = type('obj', (object,), {'content': response_content})
    
    # ===== Update state and return =====
    try:
        # IMPORTANT: Instead of creating a new state, modify the existing one
        # This preserves all the existing information including plot_images
        state["messages"] = state.get("messages", []) + [
            AIMessage(content=response.content, name="Supervisor")
        ]
        state["next"] = "FINISH"
        
        # Final state validation
        logging.info(f"Final state keys: {list(state.keys())}")
        logging.info(f"Final messages count: {len(state['messages'])}")
        logging.info(f"Final plot_images: {state.get('plot_images', [])}")
        logging.info("==================== END SUPERVISOR RESPONSE DEBUGGING ====================\n")
        
    except Exception as e:
        logging.error(f"Error updating state: {str(e)}")
        logging.error(traceback.format_exc())
        # Emergency recovery - create minimal valid state
        state["messages"] = state.get("messages", []) + [
            AIMessage(content="Error processing your request. Technical details have been logged.", name="Supervisor")
        ]
        state["next"] = "FINISH"
    
    return state


def create_supervisor_agent(user_query, datasets_info, memory):
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
    import functools
    import json

    # Initialize agents
    visualization_agent, dataframe_agent = initialize_agents(user_query, datasets_info)
    logging.info("Initialized agents for workflow")

    # Dynamically build the members list
    members = []
    if visualization_agent:
        members.append("VisualizationAgent")
    if dataframe_agent:
        members.append("DataFrameAgent")
    logging.info(f"Supervisor managing members: {members}")

    # Format dataset information for both planning and direct responses
    datasets_text = ""
    if datasets_info:
        for i, info in enumerate(datasets_info, 1):
            datasets_text += (
                f"Dataset {i}:\n"
                f"Name: {info['name']}\n"
                f"DOI: {info.get('doi', 'Not available')}\n"
                f"Description: {info['description']}\n"
                f"Type: {info['data_type']}\n\n"
            )
    else:
        datasets_text = "No datasets available."

    # Define the updated supervisor system prompt
    system_prompt_supervisor = f"""
You are a supervisor managing conversations and tasks. Your primary role is to efficiently handle user queries by either responding directly or delegating tasks to specialized agents.

### CRITICAL ROUTING INSTRUCTIONS
Your ONLY options for the "next" field are:
- "RESPOND" - Use this when you want to answer directly without delegating
- "FINISH" - Use this when all tasks are complete
- {', '.join(members)} - These are the agent names you can route to

You must NEVER set "next" to "create_or_update_plan" or any other tool name!

### Direct Response Instructions
For the following types of queries, respond directly using the 'RESPOND' option:
- **Questions about conversation history**, such as 'what we discussed before,' 'summarize our conversation,' or 'remind me of previous topics.' Use the provided conversation history to answer accurately.
- **Simple questions about datasets**, such as 'What's the description of the first dataset?', 'How many datasets are loaded?', or 'What are the parameters of dataset 2?' Use the dataset information below to answer directly:
  {datasets_text}

When responding directly, provide a concise and accurate answer using the conversation history and dataset information, without involving other agents. Skip the planning phase to save time and improve efficiency.

### Complex Task Instructions
For visualization and data analysis requests (like plotting, data manipulation, etc.):
1. Review the current plan in the "plan" field
2. Choose the appropriate agent from: {', '.join(members)}
3. Set "next" ONLY to one of the agent names or "FINISH" or "RESPOND"

### Available Agents and Their Capabilities
{', '.join(members)}
- VisualizationAgent: Use for all plotting and visualization tasks. Always call it when request is related to the visualization.
- DataFrameAgent: Use for simple data analysis, filtering, and manipulation

### Examples of Correct Routing
- User asks: "Summarize our conversation" → Set "next" to "RESPOND"
- User asks: "Plot the data" → Set "next" to "VisualizationAgent"
- User asks: "Analyze trends" → Set "next" to "DataFrameAgent"
- All tasks completed → Set "next" to "FINISH"

### REMEMBER
- Tools like 'create_or_update_plan' are NOT valid options for the "next" field!
- The planning process happens automatically - your job is only to decide which agent should handle the task next.
- Only use agent names that are actually available: {', '.join(members)}
"""

    # Define the routing function schema
    function_def = {
        "name": "route",
        "description": "Select the next step in the workflow.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "type": "string",
                    "enum": ["FINISH", "RESPOND"] + members,
                    "description": "MUST be one of: 'RESPOND', 'FINISH', or an agent name. NEVER use tool names here."
                },
                "plan": {
                    "title": "Plan",
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "task": {"type": "string"},
                            "agent": {"type": "string", "enum": members},
                            "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "failed"]}
                        },
                        "required": ["task", "agent", "status"]
                    }
                }
            },
            "required": ["next", "plan"],
        },
    }


    # Create the supervisor prompt template
    prompt_supervisor = ChatPromptTemplate.from_messages([
        ("system", system_prompt_supervisor),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Current plan: {plan}"),  # Add this line
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        ("system", "Based on the user query, conversation, and current plan, decide the next step.")
    ])

    # Initialize the LLM
    llm_supervisor = ChatOpenAI(api_key=API_KEY, model_name=st.session_state.get("model_name", "gpt-4o"))

    # Bind the planning tool to the LLM
    tools = [planning_tool]
# Create the supervisor chain with forced function calling
    llm_with_tools = llm_supervisor.bind_functions(
        functions=[function_def], 
        function_call={"name": "route"}  # Force it to use the route function
    )

    # FIX 5: Update chain definition to handle plan correctly
    supervisor_chain = (
        {
            "messages": lambda x: x["messages"],
            "agent_scratchpad": lambda x: x["agent_scratchpad"],
            "plan": lambda x: json.dumps(x["plan"] if x.get("plan") else [])  # Better handling
        }
        | prompt_supervisor
        | llm_with_tools
        | JsonOutputFunctionsParser()
    )

    # Define the supervisor node function
    def supervisor_with_planning(state):
        # Safety net for infinite loops
        if "iteration_count" not in state:
            state["iteration_count"] = 0
        state["iteration_count"] += 1
        if state["iteration_count"] > 10:
            state["next"] = "FINISH"
            state["messages"].append(AIMessage(content="Max iterations reached. Stopping.", name="Supervisor"))
            return state

        # First create/update the plan
        if not state.get("plan") or len(state.get("plan", [])) == 0:
            # Create a plan first before making routing decisions
            conversation_history = "\n".join([f"{msg.name if hasattr(msg, 'name') else 'User'}: {msg.content}" 
                                            for msg in state["messages"][-5:] if hasattr(msg, "content")])
            try:
                # Get user query from the state
                user_query = state.get("input", "")
                plan_result = planning_tool.invoke({
                    "user_query": user_query,
                    "conversation_history": conversation_history,
                    "available_agents": members,
                    "current_plan": json.dumps([]),
                    "datasets_info": datasets_text
                })
                
                # Improved plan extraction and validation
                if isinstance(plan_result, str):
                    try:
                        # Try direct JSON parsing
                        plan = json.loads(plan_result)
                    except json.JSONDecodeError:
                        # Try extracting JSON from structured response
                        import re
                        json_match = re.search(r'(\[.*\])', plan_result, re.DOTALL)
                        if json_match:
                            extracted_json = json_match.group(1)
                            logging.info(f"Extracted JSON from string: {extracted_json}")
                            plan = json.loads(extracted_json)
                        else:
                            plan = []
                else:
                    # If already a Python object, use directly
                    plan = plan_result
                    
                # Ensure plan is stored properly
                state["plan"] = plan
                logging.info(f"Successfully created initial plan: {json.dumps(plan)}")
            except Exception as e:
                logging.error(f"Error creating plan: {str(e)}")
                state["plan"] = []
        
        # Explicitly copy plan into a new variable for the chain
        plan_for_chain = state.get("plan", [])
        
        # Explicitly convert plan to string when logging
        logging.info(f"Plan before supervisor chain: {json.dumps(plan_for_chain)}")
        
        # Modify how the chain is invoked to ensure plan is passed correctly
        result = supervisor_chain.invoke({
            "messages": state["messages"],
            "agent_scratchpad": state.get("agent_scratchpad", []),
            "plan": plan_for_chain  # Pass explicitly rather than relying on state
        })
        
        logging.info(f"Supervisor chain output: {result}")
        
        # Extract next step from the result
        next_step = result.get("next")
        
        # Add validation for the returned plan
        updated_plan = result.get("plan", state.get("plan", []))
        
        # Save the updated plan back with validation
        if isinstance(updated_plan, list):
            state["plan"] = updated_plan
        elif isinstance(updated_plan, str):
            try:
                state["plan"] = json.loads(updated_plan)
            except Exception as e:
                # Keep existing plan if parsing fails
                logging.error(f"Failed to parse updated plan: {updated_plan}, error: {str(e)}")
        
        # Critical validation step
        valid_next_steps = ["RESPOND", "FINISH"] + members
        logging.info(f"next_step type: {type(next_step)}, value: {repr(next_step)}, valid_steps: {valid_next_steps}")
        
        if next_step not in valid_next_steps:
            # Handle the case where LLM incorrectly outputs a tool name instead of a routing option
            if next_step == "create_or_update_plan":
                # If LLM tries to route to the planning tool, find the first pending task in the plan
                # and route to that agent instead
                if state["plan"]:
                    for task in state["plan"]:
                        if task.get("status") == "pending":
                            next_step = task.get("agent")
                            logging.info(f"Corrected routing from 'create_or_update_plan' to '{next_step}'")
                            break
                    else:
                        # If no pending tasks, finish
                        next_step = "FINISH"
                        logging.info("No pending tasks in plan, routing to FINISH")
                else:
                    # No plan, route to first available agent
                    next_step = members[0] if members else "FINISH"
                    logging.info(f"No plan available, routing to {next_step}")
            else:
                # For any other invalid routing, default to FINISH
                original_next_step = next_step
                next_step = "FINISH"
                logging.error(f"Invalid next_step '{original_next_step}' received. Defaulting to 'FINISH'.")
                state["messages"].append(AIMessage(content=f"Error: Invalid step '{original_next_step}'. Finishing.", name="Supervisor"))

        # Handle special case for RESPOND option
        if next_step == "RESPOND":
            state["next"] = "RESPOND"
            return state

        # Update task statuses based on last agent message
        last_agent_message = next((msg for msg in reversed(state["messages"]) 
                                if hasattr(msg, "name") and msg.name in members), None)
        if last_agent_message and state["plan"]:
            agent_name = last_agent_message.name
            for task in state["plan"]:
                if task.get("agent") == agent_name and task.get("status") in ["pending", "in_progress"]:
                    task["status"] = "completed"
                    logging.info(f"Marked task '{task.get('task')}' as completed for {agent_name}")
                    break

        # Check if all tasks are done
        if state["plan"] and all(task.get("status") == "completed" for task in state["plan"]):
            state["next"] = "FINISH"
            logging.info("All tasks completed; setting next to FINISH")
            return state

        # Set final routing decision
        state["next"] = next_step
        
        # If routing to an agent, mark the corresponding task as in_progress
        if next_step in members and state["plan"]:
            for task in state["plan"]:
                if task.get("agent") == next_step and task.get("status") == "pending":
                    task["status"] = "in_progress"
                    logging.info(f"Setting task '{task.get('task')}' as in_progress for {next_step}")
                    break

        # Log the final plan and routing decision
        logging.info(f"Final plan: {json.dumps(state.get('plan', []))}")
        logging.info(f"Final routing decision: {next_step}")
        return state

    # Define the agent state
    class AgentState(TypedDict):
        messages: Sequence[BaseMessage]
        next: str
        agent_scratchpad: Sequence[BaseMessage]
        user_query: str
        last_agent_message: str
        plot_images: List[str]
        model_name: str
        plan: List[Dict[str, str]]

    # Initialize the workflow
    workflow = StateGraph(AgentState)

    # Add nodes
    if visualization_agent:
        workflow.add_node("VisualizationAgent", 
                         functools.partial(agent_node, agent=visualization_agent, name="VisualizationAgent"))
    if dataframe_agent:
        workflow.add_node("DataFrameAgent", 
                         functools.partial(agent_node, agent=dataframe_agent, name="DataFrameAgent"))
    workflow.add_node("supervisor", supervisor_with_planning)
    workflow.add_node("supervisor_response", supervisor_response)

    # Configure edges
    for member in members:
        workflow.add_edge(member, "supervisor")
    conditional_map = {k: k for k in members}
    conditional_map["FINISH"] = END
    conditional_map["RESPOND"] = "supervisor_response"
    workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
    workflow.set_entry_point("supervisor")

    # Compile and return the graph
    graph = workflow.compile(checkpointer=memory)
    logging.info("Compiled workflow graph")
    return graph
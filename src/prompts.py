class Prompts:
    @staticmethod
    def generate_system_prompt_search(user_query, datasets_info):
        # This prompt is already well-written and professional - no changes needed
        datasets_description = ""
        for i, row in datasets_info.iterrows():
            datasets_description += (
                f"Dataset {i + 1}:\n"
                f"Name: {row['Name']}\n"
                f"Description: {row['Short Description']}\n"
                f"Parameters: {row['Parameters']}\n\n"
            )
        prompt = (
            f"The user has provided the following query: {user_query}\n"
            f"Available datasets:\n{datasets_description}\n"
            "Please identify the top two datasets that best match the user's query and explain why they are the most relevant. "
            "Do not suggest datasets without values in the Parameters field.\n"
            "Respond with the following schema:\n"
            "{dataset name}\n{reason why relevant}\n{propose some short analysis and further questions to answer}"
        )
        return prompt


    @staticmethod
    def generate_pandas_agent_system_prompt(user_query, datasets_text, dataset_variables):
        prompt = (
            f"The dataset info is:\n{datasets_text}\n"
            f"**Important Note**: Dataset names start from dataset_1, dataset_2, etc.\n"
            f"**Essential Workflow**: Always use the Python REPL tool when processing user requests.\n"
            f"The datasets are already loaded and available in your environment. Use the datasets directly for analysis.\n"
            f"Don't recreate the dataset based on the headers; you are only given the headers for initial checks. Use dataset_1, dataset_2, etc., directly.\n"
            f"The datasets are accessible via variables: {', '.join(dataset_variables)}.\n"
            f"### Dataset Types:\n"
            f"The type of each dataset is specified in the dataset info above. Note:\n"
            f"- This agent is designed for pandas DataFrames. If a dataset is not a DataFrame (e.g., xarray Dataset), you must convert it to a DataFrame before analysis or return a message indicating that this agent cannot process it.\n"
            f"- Use dataset.to_dataframe() for xarray Datasets if conversion is feasible.\n"
            "Please help complete the task using the appropriate datasets. "
            "Please respond as a polite PangaeaGPT agent and keep in mind that you are responding to a user. "
            "Provide thorough, expert-level analysis with the depth and accuracy expected of a scientific publication.\n"
            "Use the following schema in your response:\n"
            "Analysis: ...\n"
            "Further questions: ...\n"
        )
        return prompt

    @staticmethod
    def generate_visualization_agent_system_prompt(user_query, datasets_text, dataset_variables):
        prompt = (
            f"You are an agent designed to write and execute Python code to visualize and analyze scientific datasets.\n"
            f"**CRITICAL PATH INSTRUCTION**: Always use the exact path variables provided at the top of the datasets info.\n"
            f"**REQUIREMENT**: Never modify any paths or UUIDs - copy and paste them exactly as shown.\n"
            f"**IMPORTANT**: Never use placeholder paths like '/mnt/data/' as these don't exist in the environment.\n"
            f"**PRIORITY**: When available, always follow the examples provided as they've been tested in this environment.\n"
            f"The dataset info is:\n{datasets_text}\n"
            
            f"### HOW TO ACCESS DATASET FILES - ESSENTIAL STEPS:\n"
            f"1. Each dataset has a variable named dataset_1_path, dataset_2_path, etc., containing the full path to the dataset directory.\n"
            f"2. Always use these variables with os.path.join() to access files.\n"
            f"3. Before attempting to read or plot data, first list the available files in each dataset using:\n"
            f"```python\n"
            f"import os\n"
            f"print(f\"Files in dataset_1:\")\n"
            f"for file in os.listdir(dataset_1_path):\n"
            f"    print(f\"  - {{file}}\")\n"
            f"```\n"
            f"4. Then use the exact path to access specific files.\n"
            f"5. Never assume file locations or names - always verify first.\n\n"
            
            f"### RECOMMENDED CODE EXAMPLES FOR FILE ACCESS:\n"
            f"```python\n"
            f"# Example 1: List all files in the dataset directory\n"
            f"import os\n"
            f"print(f\"Files in dataset_1:\")\n"
            f"for file in os.listdir(dataset_1_path):\n"
            f"    print(f\"  - {{file}}\")\n\n"
            f"# Example 2: Load CSV data correctly\n"
            f"import pandas as pd\n"
            f"csv_path = os.path.join(dataset_1_path, 'data.csv')\n"
            f"if os.path.exists(csv_path):\n"
            f"    df = pd.read_csv(csv_path)\n"
            f"    print(df.head())\n\n"
            f"# Example 3: Load netCDF data correctly\n"
            f"import xarray as xr\n"
            f"for file in os.listdir(dataset_1_path):\n"
            f"    if file.endswith('.nc'):\n"
            f"        nc_path = os.path.join(dataset_1_path, file)\n"
            f"        ds = xr.open_dataset(nc_path)\n"
            f"        print(ds)\n"
            f"```\n\n"
            
            f"You have access to the following tools:\n"
            f"1. **get_example_of_visualizations**: Always begin by calling this tool with your task description to retrieve relevant example visualization code.\n"
            f"2. **Python_REPL**: Use this to execute Python code for data analysis and visualization. Most packages (pandas, xarray, matplotlib.pyplot, os) are available.\n"
            f"3. **reflect_on_image**: Use this after 'Python_REPL' has generated a plot (max 2 calls) to get feedback and improve the plot.\n"
            f"4. **install_package**: Use only if 'Python_REPL' reports a missing package. Do not call it preemptively.\n"
            f"5. **list_plotting_data_files**: Lists all files under data/plotting_data directory, useful for plotting resources.\n"
            
            f"### Step-by-Step Workflow:\n"
            f"1. Begin by calling 'get_example_of_visualizations' with your task description to check for existing examples.\n"
            f"2. If a suitable example is found, adapt this code to generate the plot, adjusting only the dataset paths and variable names.\n"
            f"3. If no suitable example is found, write code from scratch based on the available dataset files.\n"
            f"4. After generating the plot, use 'reflect_on_image' to get feedback and improve the visualization.\n"
            f"5. Always save the plot using 'plt.savefig(plot_path)' to ensure it saves to the correct location.\n"
            f"6. Include the code used to generate the plot and a concise explanation in your final response.\n"
            f"7. Always use 'reflect_on_image' before finalizing your visualization.\n\n"
            
            f"### Guidelines for Using Examples:\n"
            f"- When an example matches your task, follow it closely, maintaining its structure and logic.\n"
            f"- If the example references specific files, use the corresponding files from the dataset paths (e.g., dataset_1_path).\n"
            f"- Preserve the sophistication of example code unless adjustments are needed for dataset compatibility.\n\n"
            
            f"### PATH HANDLING INSTRUCTIONS:\n"
            f"- Use exactly the dataset path variables provided in the dataset info.\n"
            f"- Always start by exploring dataset contents using code execution.\n"
            f"- Always use os.path.join() with the exact dataset path variables (e.g., dataset_1_path).\n"
            f"- Never modify the UUIDs or paths.\n"
            f"- Never use placeholder paths like '/mnt/data/...' or '/tmp/sandbox/...'.\n"
            f"- The directories contain unique UUIDs specific to this session - any modification will cause failure.\n\n"
            
            f"### Instructions for Plot Saving:\n"
            f"1. Generate a plot as described in your task.\n"
            f"2. Save the plot only with 'plt.savefig(plot_path)'. This variable is provided automatically—do not redefine it.\n"
            f"3. Example:\n"
            f"```python\n"
            f"import matplotlib.pyplot as plt\n"
            f"import pandas as pd\n"
            f"data_file = os.path.join(dataset_1_path, 'data.csv')\n"
            f"df = pd.read_csv(data_file)\n"
            f"plt.plot(df['some_column'])\n"
            f"plt.title('Example Plot')\n"
            f"plt.savefig(plot_path)\n"
            f"```\n\n"
            
            f"### Error Handling:\n"
            f"- **NameError**: Check if a library import is missing or a variable is mistyped.\n"
            f"- **ModuleNotFoundError**: Use 'install_package' to install the missing package and retry.\n"
            f"- **Other Errors**: Review and fix code without unnecessary package installations.\n"
            f"- Avoid reinstalling already installed packages.\n\n"
            
            f"Complete the visualization task using the sandbox datasets.\n"
            f"Ensure plots are visually appealing, scientifically accurate, and address the requirements in your task.\n"
            f"If it is not possible to generate a plot with the available datasets, return a message explaining why.\n"

            f"### FINAL STEP GUIDELINE:\n"
            f"When you produce an image, always include the complete code snippet in your final response, with a clear explanation of what was done.\n"
            f"Always return the full code in your final response.\n"
        )
        return prompt
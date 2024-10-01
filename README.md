# PANGAEA Dataset Explorer

## Overview

This project is a Streamlit-based application for exploring scientific datasets from the PANGAEA database. The app allows users to search for datasets, view their contents, and perform data analysis and visualization using advanced AI models and tools.

## Features

- **Search and List Datasets**: Query the PANGAEA database and list datasets based on user input.
- **Dataset Analysis**: Analyze datasets using AI agents to provide insights and answer user queries.
- **Data Visualization**: Generate visualizations such as sampling stations maps and master track maps.
- **Interactive Interface**: An intuitive interface built with Streamlit for easy user interaction.

## Setup

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository**:

    ```sh
    git clone https://github.com/dmpantiu/pangaeaGPT.git
    cd pangaeaGPT
    ```

2. **Install the required packages**:

    ```sh
    pip install -r requirements.txt
    ```

3. **Download and prepare necessary data**:

    Run the `setup.py` script to download and extract shapefiles and bathymetry data, and to configure API keys.

    ```sh
    python setup.py
    ```

    Follow the prompts to enter your OpenAI API key and optionally your LangChain API key and project name.

4. **Prepare Streamlit secrets**:

    The `setup.py` script will create a `.streamlit/secrets.toml` file with your API keys.

### Running the Application

1. **Start the Streamlit application**:

    ```sh
    streamlit run app.py
    ```

2. **Use the app**:

    Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

    - Use the sidebar to configure settings such as the AI model.
    - Enter your search queries in the input field to search for datasets.
    - Fetch and analyze datasets, and view generated visualizations directly in the app.

## Project Structure

- `main.py`: The main Streamlit application file containing the logic for searching, displaying, and analyzing datasets.
- `setup.py`: A setup script to download required shapefiles and bathymetry data, and to configure API keys.
- `plotting_tools/hard_agent.py`: Contains functions for generating visualizations of sampling stations and master track maps.
- `requirements.txt`: Lists the Python packages required for the project.

## Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug reports, please create an issue or submit a pull request.


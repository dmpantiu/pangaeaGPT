import os
import requests
import zipfile
try:
    from tqdm import tqdm
except ImportError:
    print("Warning: tqdm not found. Using a simple progress indicator.")
    def tqdm(iterable, *args, **kwargs):
        return iterable  # Simple fallback that just returns the iterable

# Get API keys from environment variables or prompt user for input
openai_api_key = os.environ.get('OPENAI_API_KEY') or input("Please enter your OpenAI API key: ")
langchain_api_key = os.environ.get('LANGCHAIN_API_KEY') or input("Please enter your LangChain API key (optional): ")
langchain_project_name = os.environ.get('LANGCHAIN_PROJECT_NAME') or input("Please enter your LangChain project name (optional): ")

# Define directories and URLs for the shapefiles and bathymetry
base_dir = os.path.join(os.getcwd(), 'data', 'plotting_data')
shape_files_dir = os.path.join(base_dir, 'shape_files')
bathymetry_dir = os.path.join(base_dir, 'bathymetry', 'etopo')

# URLs for the shapefiles and bathymetry
ne_10m_coastline_url = 'https://naciscdn.org/naturalearth/10m/physical/ne_10m_coastline.zip'
ne_10m_land_url = 'https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip'
ne_10m_ocean_url = 'https://naciscdn.org/naturalearth/10m/physical/ne_10m_ocean.zip'
etopo_bathymetry_url = 'https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2/ETOPO2v2-2006/ETOPO2v2c/netCDF/ETOPO2v2c_f4_netCDF.zip'

# Function to download and extract files
def download_and_extract(url, extract_to, extract_to_subfolder=True):
    file_name = url.split('/')[-1]
    dir_name = file_name.replace('.zip', '')
    local_zip_path = os.path.join(extract_to, file_name)

    if extract_to_subfolder:
        dir_path = os.path.join(extract_to, dir_name)
        os.makedirs(dir_path, exist_ok=True)
    else:
        dir_path = extract_to

    if not os.path.exists(local_zip_path):
        print(f"Downloading {url}...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte
            t = tqdm(total=total_size, unit='iB', unit_scale=True)
            with open(local_zip_path, 'wb') as f:
                for data in response.iter_content(block_size):
                    t.update(len(data))
                    f.write(data)
            t.close()
            if total_size != 0 and t.n != total_size:
                print("ERROR, something went wrong")
            print(f"Downloaded {local_zip_path}")
        else:
            print(f"Failed to download {url} - Status code: {response.status_code}")
            return

    print(f"Extracting {local_zip_path}...")
    with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
        zip_ref.extractall(dir_path)
    print(f"Extracted {local_zip_path}")

    # Remove the zip file after extraction
    os.remove(local_zip_path)
    print(f"Removed {local_zip_path}")

# Ensure base directories exist
os.makedirs(shape_files_dir, exist_ok=True)
os.makedirs(bathymetry_dir, exist_ok=True)

# Download and extract shapefiles
download_and_extract(ne_10m_coastline_url, shape_files_dir)
download_and_extract(ne_10m_land_url, shape_files_dir)
download_and_extract(ne_10m_ocean_url, shape_files_dir)

# Download and extract bathymetry data (without creating a subfolder)
download_and_extract(etopo_bathymetry_url, bathymetry_dir, extract_to_subfolder=False)

print("All required shapefiles and bathymetry data downloaded, extracted, and zip files removed successfully.")

# Sanitize inputs to remove any invalid characters
def sanitize_input(input_str):
    return ''.join(c for c in str(input_str) if ord(c) < 128) if input_str else ""

openai_api_key = sanitize_input(openai_api_key)
langchain_api_key = sanitize_input(langchain_api_key)
langchain_project_name = sanitize_input(langchain_project_name)

# Create secrets.toml file
secrets_dir = os.path.join(os.getcwd(), '.streamlit')
os.makedirs(secrets_dir, exist_ok=True)
secrets_path = os.path.join(secrets_dir, 'secrets.toml')

with open(secrets_path, 'w') as f:
    f.write("[general]\n")
    if openai_api_key:
        f.write(f"openai_api_key = \"{openai_api_key}\"\n")
    if langchain_api_key:
        f.write(f"langchain_api_key = \"{langchain_api_key}\"\n")
    if langchain_project_name:
        f.write(f"langchain_project_name = \"{langchain_project_name}\"\n")

print("API keys and project name saved to secrets.toml.")

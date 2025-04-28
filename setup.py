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
anthropic_api_key = os.environ.get('ANTHROPIC_API_KEY') or input("Please enter your Anthropic API key: ")

# LangChain API key and project name
langchain_api_key = os.environ.get('LANGCHAIN_API_KEY') or input("Please enter your LangChain API key (optional): ")
langchain_project_name = os.environ.get('LANGCHAIN_PROJECT_NAME') or input("Please enter your LangChain project name (optional): ")

# Ask for region selection if LangChain API key is provided
langchain_region = "us"  # Default to US region
if langchain_api_key:
    while True:
        region_input = os.environ.get('LANGCHAIN_REGION') or input("Please select your LangSmith region (us/eu) [default: us]: ").lower()
        if not region_input:
            region_input = "us"  # Default if nothing entered
        
        if region_input in ["us", "eu"]:
            langchain_region = region_input
            break
        else:
            print("Invalid region. Please enter 'us' or 'eu'.")

# Define directories and URLs for the shapefiles and bathymetry
base_dir = os.path.join(os.getcwd(), 'data', 'plotting_data')
shape_files_dir = os.path.join(base_dir, 'shape_files')
bathymetry_dir = os.path.join(base_dir, 'bathymetry', 'etopo')

# URLs for the shapefiles and bathymetry
ne_10m_coastline_url = 'https://naciscdn.org/naturalearth/10m/physical/ne_10m_coastline.zip'
ne_10m_land_url = 'https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip'
ne_10m_ocean_url = 'https://naciscdn.org/naturalearth/10m/physical/ne_10m_ocean.zip'

# Define both your new URL and the previously working URL as a fallback
nextcloud_url = 'https://nextcloud.awi.de/s/EiWqETB4Ko5qFEM'
fallback_etopo_url = 'https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2/ETOPO2v2-2006/ETOPO2v2c/netCDF/ETOPO2v2c_f4_netCDF.zip'

# Function to download and extract files with improved error handling
def download_and_extract(url, extract_to, extract_to_subfolder=True):
    file_name = url.split('/')[-1]
    dir_name = file_name.replace('.zip', '')
    local_zip_path = os.path.join(extract_to, file_name)

    if extract_to_subfolder:
        dir_path = os.path.join(extract_to, dir_name)
        os.makedirs(dir_path, exist_ok=True)
    else:
        dir_path = extract_to

    success = False
    
    if not os.path.exists(local_zip_path):
        print(f"Downloading {url}...")
        try:
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
                    print("ERROR, something went wrong with download size")
                    success = False
                else:
                    print(f"Downloaded {local_zip_path}")
                    success = True
            else:
                print(f"Failed to download {url} - Status code: {response.status_code}")
                success = False
        except Exception as e:
            print(f"Error during download: {str(e)}")
            success = False
    else:
        print(f"File {local_zip_path} already exists")
        success = True

    if success:
        try:
            print(f"Extracting {local_zip_path}...")
            with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
                zip_ref.extractall(dir_path)
            print(f"Extracted {local_zip_path}")
            
            # Remove the zip file after extraction
            os.remove(local_zip_path)
            print(f"Removed {local_zip_path}")
            return True
        except zipfile.BadZipFile:
            print(f"Error: {local_zip_path} is not a valid zip file")
            if os.path.exists(local_zip_path):
                os.remove(local_zip_path)
            success = False
        except Exception as e:
            print(f"Error during extraction: {str(e)}")
            success = False
    
    return False

# Function to try downloading from Nextcloud with different URL patterns
def try_nextcloud_download(base_url, extract_to, extract_to_subfolder=False):
    print("Attempting to download from Nextcloud...")
    
    # Try different Nextcloud URL variations
    nextcloud_urls = [
        base_url,  # Original URL
        f"{base_url}/download",  # Common Nextcloud direct download URL
        f"{base_url.rstrip('/')}/download"  # Ensure no double slash
    ]
    
    for url in nextcloud_urls:
        print(f"Trying Nextcloud URL: {url}")
        if download_and_extract(url, extract_to, extract_to_subfolder):
            return True
    
    print("All Nextcloud download attempts failed.")
    return False

# Ensure base directories exist
os.makedirs(shape_files_dir, exist_ok=True)
os.makedirs(bathymetry_dir, exist_ok=True)

# Download and extract shapefiles
download_and_extract(ne_10m_coastline_url, shape_files_dir)
download_and_extract(ne_10m_land_url, shape_files_dir)
download_and_extract(ne_10m_ocean_url, shape_files_dir)

# First try the Nextcloud URL, then fall back to the original URL if needed
if not try_nextcloud_download(nextcloud_url, bathymetry_dir, extract_to_subfolder=False):
    print("\nFalling back to the previously working URL...")
    if not download_and_extract(fallback_etopo_url, bathymetry_dir, extract_to_subfolder=False):
        print("\nWarning: Failed to download bathymetry data from both URLs.")
        print("You may need to manually download the ETOPO2 data and place it in the bathymetry/etopo directory.")
    else:
        print("\nSuccessfully downloaded bathymetry data using fallback URL.")
else:
    print("\nSuccessfully downloaded bathymetry data from Nextcloud.")

print("\nAll required shapefiles and any successfully downloaded bathymetry data have been extracted.")

# Sanitize inputs to remove any invalid characters
def sanitize_input(input_str):
    return ''.join(c for c in str(input_str) if ord(c) < 128) if input_str else ""

openai_api_key = sanitize_input(openai_api_key)
anthropic_api_key = sanitize_input(anthropic_api_key)
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
    if anthropic_api_key:
        f.write(f"anthropic_api_key = \"{anthropic_api_key}\"\n")
    if langchain_api_key:
        f.write(f"langchain_api_key = \"{langchain_api_key}\"\n")
    if langchain_project_name:
        f.write(f"langchain_project_name = \"{langchain_project_name}\"\n")
    if langchain_region:
        f.write(f"langchain_region = \"{langchain_region}\"\n")

# Also update the config.yaml file to include the langchain_region setting
config_path = os.path.join(os.getcwd(), 'config.yaml')
try:
    with open(config_path, 'r') as f:
        config_content = f.read()

    # Check if the config already has a langchain_region entry
    if 'langchain_region:' not in config_content:
        with open(config_path, 'a') as f:
            f.write(f"\n# LangChain region - 'us' or 'eu'\nlangchain_region: {langchain_region}  # Set to 'eu' for EU region\n")
    
    print("API keys, project name, and region saved to secrets.toml.")
    print(f"LangChain region set to: {langchain_region}")
except FileNotFoundError:
    print("Warning: config.yaml not found. Creating it...")
    with open(config_path, 'w') as f:
        f.write(f"# LangChain region - 'us' or 'eu'\nlangchain_region: {langchain_region}  # Set to 'eu' for EU region\n")
    print("Created config.yaml with LangChain region setting.")
# search_pg_default.py
import requests
import pandas as pd
import logging
import re
from typing import List, Optional
from bs4 import BeautifulSoup
import json
import os
import pangaeapy.pandataset as pdataset
from .dataset_utils import fetch_dataset_details

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to check if a variable is of a specific type
def check_if(x, cls):
    if x is not None and not isinstance(x, cls):
        raise TypeError(f"{x} must be of class: {', '.join([str(c) for c in cls])}")

# Utility functions equivalent to R functions
def pgc(x):
    return {k: v for k, v in x.items() if v is not None}

def strextract(string, pattern):
    match = re.search(pattern, string)
    return match.group(0) if match else None


# Function to parse the result
def parse_res(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    citation_tag = soup.select_one('div.citation a')
    citation = citation_tag.get_text(strip=True) if citation_tag else None

    supp_tag = soup.select_one('tr:contains("Supplement to:") .content')
    supp = supp_tag.get_text(strip=True) if supp_tag else None

    size_tag = soup.select_one('tr:contains("Size:") .content')
    size = size_tag.get_text(strip=True) if size_tag else None

    size_val = strextract(size, r"[0-9]+") if size else None
    meas = strextract(size, r"[A-Za-z].+") if size else None

    parameters = ', '.join([tag.text for tag in soup.select('tr:contains("Parameter") .content')[:10]]) + "..." if len(
        soup.select('tr:contains("Parameter") .content')) > 10 else ', '.join(
        [tag.text for tag in soup.select('tr:contains("Parameter") .content')])

    # Extract file URLs
    file_tags = soup.select('a[title="Download data"]')
    file_urls = [tag['href'] for tag in file_tags]

    return {
        'size': int(size_val.replace(",", "")) if size_val else None,
        'size_measure': meas,
        'citation': citation,
        'supplement_to': supp,
        'parameters': parameters,
        'file_urls': file_urls if file_urls else None  # Add file URLs to parsed result
    }

# Main search function
def pg_search_default(query: str, count: int = 15, from_idx: int = 0, topic: Optional[str] = None,
                      mindate: Optional[str] = None, maxdate: Optional[str] = None,
                      minlat: Optional[float] = None, maxlat: Optional[float] = None,
                      minlon: Optional[float] = None, maxlon: Optional[float] = None, **kwargs) -> pd.DataFrame:
    # Type checking for all parameters
    check_if(count, (int,))
    check_if(topic, (str,))
    check_if(mindate, (str,))
    check_if(maxdate, (str,))
    check_if(minlat, (float, int))  # Allow integers as well
    check_if(maxlat, (float, int))
    check_if(minlon, (float, int))
    check_if(maxlon, (float, int))

    # Build parameters dictionary, converting floats to strings for API
    params = pgc({
        'q': query,
        'count': count,
        'offset': from_idx,
        'topic': topic,
        'mindate': mindate,
        'maxdate': maxdate,
        'minlat': str(minlat) if minlat is not None else None,
        'maxlat': str(maxlat) if maxlat is not None else None,
        'minlon': str(minlon) if minlon is not None else None,
        'maxlon': str(maxlon) if maxlon is not None else None
    })

    url = "https://www.pangaea.de/advanced/search.php"
    logging.debug("Sending request to PANGAEA with parameters: %s", params)
    response = requests.get(url, params=params, **kwargs)
    response.raise_for_status()
    logging.debug(f"URL: {response.url}")
    logging.debug(f"Response Status Code: {response.status_code}")
    results = response.json()
    logging.debug("Received response from PANGAEA")

    # Save the initial JSON response to transit.json
    transit_json_path = os.path.join(os.getcwd(), 'transit.json')
    with open(transit_json_path, 'w') as f:
        json.dump(results, f, indent=4)
    logging.info(f"Initial JSON response saved to {transit_json_path}")

    parsed = []
    for index, res in enumerate(results.get('results', [])):
        html_content = res.get('html', '')
        res['doi'] = f"https://doi.org/{res['URI'].replace('doi:', '')}"
        parsed_res = parse_res(html_content)
        res.update(parsed_res)

        name = res.get('citation', 'No name available')
        abstract, parameters = fetch_dataset_details(res['doi'])
        short_description = " ".join(abstract.split()[:100]) + "..." if len(abstract.split()) > 100 else abstract

        parsed.append({
            'Number': index + 1,
            'Name': name,
            'DOI': res['doi'],
            'DOI Number': res['doi'].split('/')[-1],
            'Description': abstract,
            'Short Description': short_description,
            'Score': res.get('score', 0),
            'Parameters': parameters,
            'file_urls': parsed_res['file_urls']
        })

    df = pd.DataFrame(parsed)
    total_hits = results.get('totalCount', 0)
    df.attrs['total'] = total_hits
    df.attrs['max_score'] = results.get('maxScore', None)
    return df


def direct_access_doi(doi: str):
    """
    Directly access datasets by DOI without using the search function.
    
    Args:
        doi: One or more DOIs separated by commas
        
    Returns:
        tuple: (datasets_info, prompt_text)
    """
    import pandas as pd
    import logging
    import streamlit as st
    import re
    import pangaeapy.pandataset as pdataset
    
    # Parse input to get list of DOIs
    dois = [d.strip() for d in doi.split(',')]
    logging.info(f"Processing direct DOI access for: {dois}")
    
    # Create a list to store dataset info
    datasets_list = []
    
    for i, curr_doi in enumerate(dois):
        try:
            # Normalize DOI format if needed
            original_doi = curr_doi  # Keep original for reference
            pangaea_id = None
            
            if not curr_doi.startswith(("http://", "https://")):
                # Check if it's a PANGAEA ID number
                if re.match(r"^\d+$", curr_doi):
                    pangaea_id = curr_doi
                    curr_doi = f"https://doi.pangaea.de/10.1594/PANGAEA.{curr_doi}"
                # Check if it's a PANGAEA DOI with PANGAEA prefix
                elif curr_doi.startswith("PANGAEA."):
                    pangaea_id = curr_doi.split("PANGAEA.")[1]
                    curr_doi = f"https://doi.pangaea.de/10.1594/{curr_doi}"
                # Check if it's just the DOI part
                elif curr_doi.startswith("10.1594/"):
                    match = re.search(r"PANGAEA\.(\d+)", curr_doi)
                    pangaea_id = match.group(1) if match else None
                    curr_doi = f"https://doi.org/{curr_doi}"
                else:
                    # Assume it's a PANGAEA ID
                    pangaea_id = curr_doi
                    curr_doi = f"https://doi.pangaea.de/10.1594/PANGAEA.{curr_doi}"
            else:
                # Extract ID from full URL
                match = re.search(r"PANGAEA\.(\d+)", curr_doi)
                pangaea_id = match.group(1) if match else None
            
            logging.info(f"Fetching details for DOI: {curr_doi} (ID: {pangaea_id})")
            
            # Get dataset details directly
            try:
                if pangaea_id and pangaea_id.isdigit():
                    dataset = pdataset.PanDataSet(id=int(pangaea_id))
                    dataset.setMetadata()
                    
                    # Get title
                    dataset_title = getattr(dataset, 'title', None) or f"Dataset from DOI: {curr_doi}"
                    
                    # Get abstract
                    abstract = getattr(dataset, 'abstract', "No description available") or "No description available"
                    
                    # Get parameters
                    param_dict = dataset.getParamDict()
                    short_names = param_dict.get('shortName', [])
                    parameters = ', '.join(short_names) + "..." if len(short_names) > 10 else ', '.join(short_names)
                else:
                    dataset_title = f"Dataset from DOI: {curr_doi}"
                    abstract = "No description available"
                    parameters = "No parameters available"
            except Exception as e:
                logging.error(f"Error getting dataset details for DOI {curr_doi}: {str(e)}")
                dataset_title = f"Dataset from DOI: {curr_doi}"
                abstract = f"Error fetching details: {str(e)}"
                parameters = "No parameters available due to error"
            
            short_description = " ".join(abstract.split()[:100]) + "..." if len(abstract.split()) > 100 else abstract
            
            # Add dataset to list
            datasets_list.append({
                'Number': i + 1,
                'Name': dataset_title,
                'DOI': curr_doi,
                'DOI Number': curr_doi.split('/')[-1] if '/' in curr_doi else curr_doi,
                'Description': abstract,
                'Short Description': short_description,
                'Score': 1.0,  # Default score
                'Parameters': parameters,
                'file_urls': []
            })
            
            logging.info(f"Successfully added DOI to results: {curr_doi} with title: {dataset_title}")
            
        except Exception as e:
            logging.error(f"Error processing DOI {curr_doi}: {str(e)}")
            # Add error dataset to list
            datasets_list.append({
                'Number': i + 1,
                'Name': f"Error: Could not access DOI: {curr_doi}",
                'DOI': curr_doi,
                'DOI Number': curr_doi.split('/')[-1] if '/' in curr_doi else curr_doi,
                'Description': f"Error: {str(e)}",
                'Short Description': f"Error: {str(e)}",
                'Score': 0.0,
                'Parameters': "No parameters available due to error",
                'file_urls': []
            })
    
    # Create DataFrame from list
    datasets_info = pd.DataFrame(datasets_list)
    
    # Store in session state
    st.session_state.datasets_info = datasets_info
    
    # Add message to chat
    st.session_state.messages_search.append({
        "role": "assistant",
        "content": f"**Direct access to DOIs:** {', '.join(dois)}"
    })
    
    if not datasets_list:
        st.session_state.messages_search.append({
            "role": "assistant",
            "content": "No valid datasets found from the provided DOIs."
        })
        return datasets_info, "No valid datasets found."
    
    st.session_state.messages_search.append({
        "role": "assistant",
        "content": "**Datasets Information:**",
        "table": datasets_info.to_json(orient="split")
    })
    
    # Create prompt for the search agent
    datasets_description = ""
    for i, row in datasets_info.iterrows():
        datasets_description += (
            f"Dataset {i + 1}:\n"
            f"Name: {row['Name']}\n"
            f"Description: {row['Short Description']}\n"
            f"Parameters: {row['Parameters']}\n\n"
        )
    
    prompt_search = (
        f"The user has directly accessed the following DOIs: {', '.join(dois)}\n"
        f"Available datasets:\n{datasets_description}\n"
        "These datasets are now available for selection. You can analyze these datasets to help the user understand their content and potential use."
    )
    
    return datasets_info, prompt_search
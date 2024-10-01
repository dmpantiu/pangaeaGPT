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

    return {
        'size': int(size_val.replace(",", "")) if size_val else None,
        'size_measure': meas,
        'citation': citation,
        'supplement_to': supp,
        'parameters': parameters
    }

# Function to fetch dataset details using pangaeapy
def fetch_dataset_details(doi):
    try:
        dataset = pdataset.PanDataSet(id=doi)
        dataset.setMetadata()  # Ensure metadata is fetched

        abstract = getattr(dataset, 'abstract', "No description available") or "No description available"
        param_dict = dataset.getParamDict()
        short_names = param_dict.get('shortName', [])
        parameters = ', '.join(short_names) + "..." if len(short_names) > 10 else ', '.join(short_names)

        return abstract, parameters

    except Exception as e:
        logging.error(f"Error fetching dataset details for DOI {doi}: {e}")
        return "No description available", "No parameters available"


# Main search function
def pg_search_default(query: str, count: int = 5, from_idx: int = 0, topic: Optional[str] = None,
              mindate: Optional[str] = None, maxdate: Optional[str] = None, **kwargs) -> pd.DataFrame:
    check_if(count, (int,))
    check_if(topic, (str,))
    check_if(mindate, (str,))
    check_if(maxdate, (str,))

    params = pgc({
        'q': query,
        'count': count,
        'offset': from_idx,
        'topic': topic,
        'mindate': mindate,
        'maxdate': maxdate
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

        # Fetch detailed metadata using pangaeapy
        abstract, parameters = fetch_dataset_details(res['doi'])
        print(abstract, parameters)
        short_description = " ".join(abstract.split()[:100]) + "..." if len(abstract.split()) > 100 else abstract

        parsed.append({
            'Number': index + 1,
            'Name': name,
            'DOI': res['doi'],
            'DOI Number': res['doi'].split('/')[-1],
            'Description': abstract,
            'Short Description': short_description,
            'Score': res.get('score', 0),
            'Parameters': parameters
        })

    df = pd.DataFrame(parsed)

    # Check if 'results['totalCount']' is an integer or a dictionary
    total_hits = results.get('totalCount', 0)
    df.attrs['total'] = total_hits
    df.attrs['max_score'] = results.get('maxScore', None)

    # Save DataFrame to CSV in the main folder
    #main_folder_path = os.getcwd()
    #csv_path = os.path.join(main_folder_path, 'search_results.csv')
    #df.to_csv(csv_path, index=False)
    #logging.info(f"Search results saved to {csv_path}")

    return df
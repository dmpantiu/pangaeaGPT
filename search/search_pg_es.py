# search_pg_es.py
import requests
import pandas as pd
import os
import re
import logging
import streamlit as st

# Function to perform the PANGAEA search
def pg_search_es(query=None, size=None, from_idx=0, source=None, df=None, analyzer='folding_search',
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

# Function to search datasets based on a query
def search_pg_es(query):
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

    #Comment out later
    # Save DataFrame to CSV in the main folder
    #main_folder_path = os.getcwd()
    #csv_path = os.path.join(main_folder_path, 'search_results.csv')
    #df_datasets.to_csv(csv_path, index=False)
    #logging.info(f"Search results saved to {csv_path}")
    #/Comment out later

    return df_datasets



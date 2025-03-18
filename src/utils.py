# src/utils.py


import os
import uuid
import re
import logging
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import time
import json

# Generate a unique image path for saving plots
def generate_unique_image_path():
    figs_dir = os.path.join('tmp', 'figs')
    os.makedirs(figs_dir, exist_ok=True)
    unique_filename = f'fig_{uuid.uuid4()}.png'
    unique_path = os.path.join(figs_dir, unique_filename)
    logging.debug(f"Generated unique image path: {unique_path}")
    return unique_path


# Function to sanitize input
def sanitize_input(query: str) -> str:
    return query.strip()

# Define the function to extract the last Python REPL command
def get_last_python_repl_command():
    import streamlit as st  # Ensure Streamlit is imported
    if 'intermediate_steps' not in st.session_state:
        logging.warning("No intermediate steps found in session state.")
        return None

    intermediate_steps = st.session_state['intermediate_steps']
    python_repl_commands = []
    for step in intermediate_steps:
        action = step[0]
        observation = step[1]
        if action.get('tool') == 'Python_REPL':
            python_repl_commands.append(action)

    if python_repl_commands:
        last_command_action = python_repl_commands[-1]
        command = last_command_action.get('tool_input', '')
        logging.debug(f"Extracted last Python REPL command: {command}")
        return command
    else:
        logging.warning("No Python_REPL commands found in intermediate steps.")
        return None


def log_history_event(session_data: dict, event_type: str, details: dict):
    if "execution_history" not in session_data:
        session_data["execution_history"] = []  # fallback

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    event = {
        "type": event_type,
        "timestamp": timestamp
    }
    event.update(details)  # merges in content from details

    session_data["execution_history"].append(event)

def update_thinking_log(session_data: dict, message: str, agent_name: str = None, tool_name: str = None, 
                       parent_id: str = None, step_id: str = None, start_time: float = None, end_time: float = None):
    """
    Updates the thinking log with a new message with hierarchical structure.
    
    Args:
        session_data: The session state dictionary
        message: The message to add to the log
        agent_name: The name of the agent (optional)
        tool_name: The name of the tool being used (optional)
        parent_id: ID of the parent step (for hierarchical display)
        step_id: Unique ID for this step
        start_time: Start time of this step (for timing)
        end_time: End time of this step (for timing)
    """
    if "thinking_log" not in session_data:
        session_data["thinking_log"] = []
    
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    
    # Generate step_id if not provided
    if not step_id:
        step_id = str(uuid.uuid4())
    
    # Calculate duration if both start and end times are provided
    duration = None
    if start_time and end_time:
        duration = round(end_time - start_time, 2)
    
    entry = {
        "id": step_id,
        "parent_id": parent_id,
        "timestamp": timestamp,
        "message": message,
        "agent_name": agent_name,
        "tool_name": tool_name,
        "duration": duration,
        "children": []
    }
    
    # If this is a child step, add it to its parent
    if parent_id:
        for log_entry in session_data["thinking_log"]:
            if log_entry["id"] == parent_id:
                log_entry["children"].append(entry)
                break
    else:
        # This is a top-level step
        session_data["thinking_log"].append(entry)
    
    # Update the UI if this is being called during active processing
    if "thinking_status" in session_data and session_data.get("processing", False):
        _render_thinking_log(session_data)

def _render_thinking_log(session_data):
    """Renders the hierarchical thinking log in the Streamlit UI"""
    with session_data["thinking_status"]:
        with st.expander("**🧠 Agent Execution Trace**", expanded=True):
            # First, create a flattened version sorted by timestamp
            flat_log = []
            
            def flatten_log(entries, level=0):
                for entry in entries:
                    flat_log.append((level, entry))
                    if entry.get("children"):
                        flatten_log(entry["children"], level + 1)
            
            # Start with top-level entries
            flatten_log([entry for entry in session_data["thinking_log"] if not entry.get("parent_id")])
            
            # Then render each entry with proper indentation
            for level, entry in flat_log:
                # Create indentation based on level
                indent = "&nbsp;" * (level * 4)
                
                # Format the line with timing if available
                timing_str = f" ({entry['duration']}s)" if entry.get('duration') else ""
                
                # Format based on entry type
                if entry.get("agent_name"):
                    line = f"{indent}**{entry['agent_name']}**{timing_str}: {entry['message']}"
                elif entry.get("tool_name"):
                    line = f"{indent}*Tool: {entry['tool_name']}*{timing_str}: {entry['message']}"
                else:
                    line = f"{indent}{entry['message']}{timing_str}"
                
                st.markdown(line, unsafe_allow_html=True)

def list_directory_contents(path):
    """
    Generate a formatted string listing all files in a directory and its subdirectories.
    """
    result = []
    for root, dirs, files in os.walk(path):
        level = root.replace(path, '').count(os.sep)
        indent = ' ' * 4 * level
        result.append(f"{indent}{os.path.basename(root)}/")
        sub_indent = ' ' * 4 * (level + 1)
        for file in files:
            result.append(f"{sub_indent}{file}")
    return "\n".join(result)


def escape_curly_braces(text):
    if isinstance(text, str):
        return text.replace('{', '{{').replace('}', '}}')
    return str(text)  # Convert non-strings to strings safely
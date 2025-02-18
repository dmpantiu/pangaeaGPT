# main.py
import logging
import uuid
from typing import List

import pandas as pd
from langchain_community.chat_message_histories import ChatMessageHistory
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory

from src.agents import (
    create_supervisor_agent,
    create_search_agent,
    create_visualization_agent,
    create_pandas_agent,
    create_hard_coded_visualization_agent,
    create_oceanographer_agent,
)
from src.search.dataset_utils import fetch_dataset, convert_df_to_csv
from src.memory import CustomMemorySaver


def initialize_session_state(session_state: dict):
    session_state_defaults = {
        "messages_search": [],
        "messages_data_agent": [],
        "datasets_cache": {},
        "datasets_info": None,
        "active_datasets": [],
        "selected_datasets": set(),
        "show_dataset": True,
        "current_page": "search",
        "dataset_dfs": {},
        "dataset_names": {},
        "saved_plot_paths": {},
        "memory": MemorySaver(),
        "visualization_agent_used": False,
        "chat_history": ChatMessageHistory(session_id="search-agent-session"),
        "search_method": "PANGAEA Search (default)",
        "selected_text": "",
        "new_plot_generated": False,
        "execution_history": []
    }

    for key, value in session_state_defaults.items():
        if key not in session_state:
            session_state[key] = value


def get_search_agent(datasets_info, model_name, api_key):
    return create_search_agent(datasets_info=datasets_info)


def process_search_query(user_input: str, search_agent, session_data: dict):
    session_data["chat_history"] = ChatMessageHistory(session_id="search-agent-session")
    for message in session_data["messages_search"]:
        if message["role"] == "user":
            session_data["chat_history"].add_user_message(message["content"])
        elif message["role"] == "assistant":
            session_data["chat_history"].add_ai_message(message["content"])

    def get_truncated_chat_history(session_id):
        truncated_messages = session_data["chat_history"].messages[-10:]
        truncated_history = ChatMessageHistory(session_id=session_id)
        for msg in truncated_messages:
            if isinstance(msg, HumanMessage):
                truncated_history.add_user_message(msg.content)
            elif isinstance(msg, AIMessage):
                truncated_history.add_ai_message(msg.content)
            else:
                truncated_history.add_message(msg)
        return truncated_history

    search_agent_with_memory = RunnableWithMessageHistory(
        search_agent,
        get_truncated_chat_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    response = search_agent_with_memory.invoke(
        {"input": user_input},
        {"configurable": {"session_id": "search-agent-session"}},
    )

    ai_message = response["output"]
    return ai_message


def add_user_message_to_search(user_input: str, session_data: dict):
    session_data["messages_search"].append({"role": "user", "content": user_input})


def add_assistant_message_to_search(content: str, session_data: dict):
    session_data["messages_search"].append({"role": "assistant", "content": content})


def load_selected_datasets_into_cache(selected_datasets, session_data: dict):
    for doi in selected_datasets:
        if doi not in session_data["datasets_cache"]:
            dataset, name = fetch_dataset(doi)
            if dataset is not None:
                session_data["datasets_cache"][doi] = (dataset, name)
                session_data["dataset_dfs"][doi] = dataset
                session_data["dataset_names"][doi] = name


def set_active_datasets_from_selection(session_data: dict):
    session_data["active_datasets"] = list(session_data["selected_datasets"])


def get_datasets_info_for_active_datasets(session_data: dict):
    if session_data["datasets_info"] is None:
        return []

    datasets_info = []
    for doi in session_data["active_datasets"]:
        dataset, name = session_data["datasets_cache"].get(doi, (None, None))
        if dataset is not None:
            description_row = session_data["datasets_info"].loc[
                session_data["datasets_info"]["DOI"] == doi, "Short Description"
            ]
            description = description_row.values[0] if len(description_row) > 0 else "No description"
            df_head = dataset.head().to_string()
            datasets_info.append({
                'doi': doi,
                'name': name,
                'description': description,
                'df_head': df_head,
                'dataset': dataset
            })
    return datasets_info


def create_and_invoke_supervisor_agent(user_query: str, datasets_info: list, memory, session_data: dict):
    graph = create_supervisor_agent(user_query, datasets_info, memory)
    if graph is None:
        return None

    messages = []
    for message in session_data["messages_data_agent"]:
        if message["role"] == "user":
            messages.append(HumanMessage(content=message["content"], name="User"))
        elif message["role"] == "assistant":
            messages.append(AIMessage(content=message["content"], name="Assistant"))
        else:
            messages.append(AIMessage(content=message["content"], name=message["role"]))

    limited_messages = messages[-7:]
    initial_state = {
        "messages": limited_messages,
        "next": "supervisor",
        "agent_scratchpad": [],
        "input": user_query,
        "plot_images": [],
        "last_agent_message": ""
    }

    config = {"configurable": {"thread_id": session_data.get('thread_id', str(uuid.uuid4())), "recursion_limit": 5}}

    response = graph.invoke(initial_state, config=config)
    return response


def add_user_message_to_data_agent(user_input: str, session_data: dict):
    session_data["messages_data_agent"].append({"role": "user", "content": f"{user_input}"})


def add_assistant_message_to_data_agent(content: str, plot_images, visualization_agent_used, session_data: dict):
    new_message = {
        "role": "assistant",
        "content": content,
        "plot_images": plot_images if plot_images else [],
        "visualization_agent_used": visualization_agent_used
    }
    session_data["messages_data_agent"].append(new_message)


def convert_dataset_to_csv(dataset: pd.DataFrame) -> bytes:
    return convert_df_to_csv(dataset)


def has_new_plot(session_data: dict) -> bool:
    return session_data.get("new_plot_generated", False)


def reset_new_plot_flag(session_data: dict):
    session_data["new_plot_generated"] = False


def get_dataset_csv_name(doi: str) -> str:
    return f"dataset_{doi.split('/')[-1]}.csv"


def set_current_page(session_data: dict, page_name: str):
    session_data["current_page"] = page_name


def set_selected_text(session_data: dict, text: str):
    session_data["selected_text"] = text


def set_show_dataset(session_data: dict, show: bool):
    session_data["show_dataset"] = show


def set_dataset_for_data_agent(session_data: dict, doi: str, csv_data: bytes, dataset: pd.DataFrame, name: str):
    session_data["dataset_csv"] = csv_data
    session_data["dataset_df"] = dataset
    session_data["dataset_name"] = name
    session_data["current_page"] = "data_agent"


def ensure_memory(session_data: dict):
    if "memory" not in session_data:
        session_data["memory"] = CustomMemorySaver()


def ensure_thread_id(session_data: dict):
    if "thread_id" not in session_data:
        session_data["thread_id"] = str(uuid.uuid4())

import json
import re
import streamlit as st
import os

from tool_calls import TOOL_CALL_PATTERN


def display_assistant_message(agent_message_list):
    """
    Display assistant message, hiding tool calls in expanders.
    """

    for msg in agent_message_list:
        if msg["type"] == "text":
            st.markdown(msg["content"])
        elif msg["type"] == "tool_call":
            with st.expander(f"Tool Call: {msg['tool_name']}", expanded=False):
                st.code(f"Tool Call: {msg['tool_name']} --- Args: {msg['args']}", language="json")
        


def save_chat_history(chat_id, messages):
    """Save chat history to disk in ChatHistory-compatible format."""
    os.makedirs("/app/chat", exist_ok=True)

    with open(f"/app/chat/{chat_id}.json", "w") as f:
        json.dump(messages, f)

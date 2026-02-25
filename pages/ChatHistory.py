import streamlit as st
from streamlit.components.v1 import html
import os
import time
import uuid
import json
from display_chat import display_assistant_message

# Add a tracking token
html('<script async defer data-website-id="0b8faebf-f345-4b89-a5d2-7cbe1fd7d88d" src="https://analytics.gnps2.org/umami.js"></script>', width=0, height=0)
html('<script async defer data-website-id="74bc9983-13c4-4da0-89ae-b78209c13aaf" src="https://analytics.gnps2.org/umami.js"></script>', width=0, height=0)

# Write the page label
st.set_page_config(
    page_title="Chatbot History",
    page_icon="👋",
)

# Getting a text field
st.write("Welcome to the Chatbot History Viewer!")

query_params = st.query_params

# chat_id = st.text_input("Enter Chat ID", query_params["chat_id"])
chat_id = st.text_input("Enter Chat ID", st.query_params.get("chat_id", ""))
# Draw a vertical line
st.markdown("---")
if not chat_id:
    st.warning("No chat ID provided. Start a chat first, then click Chat History.")
    st.stop()
# Display chat history
chat_history = json.loads(open(f"/app/chat/{chat_id}.json").read())

# Display chat messages from history on app rerun
for message in chat_history:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            for part in message["content"]:
                if part.get("tool_calls", False):
                    display_assistant_message(part["content"], tool_calls=True)
                else:
                    display_assistant_message(part["content"], tool_calls=False)

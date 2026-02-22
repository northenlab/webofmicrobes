import streamlit as st
import os
import time
import uuid
import json
from display_chat import display_assistant_message

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

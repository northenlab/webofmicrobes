import streamlit as st
import os
import time
import uuid
import json

# Write the page label
st.set_page_config(
    page_title="Chatbot History",
    page_icon="👋",
)

# Getting a text field
st.write("Welcome to the Chatbot History Viewer!")

query_params = st.query_params

chat_id = st.text_input("Enter Chat ID", query_params["chat_id"])

# Draw a vertical line
st.markdown("---")

# Display chat history
chat_history = json.loads(open(f"/app/chat/{chat_id}.json").read())

# Display chat messages from history on app rerun
for message in chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

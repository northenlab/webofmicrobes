import streamlit as st
import os
import json

# Write the page label
st.set_page_config(
    page_title="Chatbot History",
    page_icon="👋",
)

st.write("Welcome to the Dev Viewer!")

CHAT_DIR = os.path.join(os.getcwd(), ".", "chat")
DEV_CHAT_DIR = os.path.join(CHAT_DIR, "developer")


def list_json_filenames(directory_path: str) -> list[str]:
    try:
        entries = os.listdir(directory_path)
    except FileNotFoundError:
        return []

    return sorted(
        [name for name in entries if name.endswith(".json")],
        key=str.lower,
    )


dev_chat_files = list_json_filenames(DEV_CHAT_DIR)

table_rows = (
    [{"chat_type": "developer", "filename": name} for name in dev_chat_files]
)

# Go through each chat, parse it and grab the developer key and put it into the table
for row in table_rows:
    with open(os.path.join(DEV_CHAT_DIR, row["filename"]), "r") as file:
        chat_data = file.read()
        try:
            chat_json = json.loads(chat_data)
            row["developer"] = chat_json.get("developer", "N/A")
        except json.JSONDecodeError:
            row["developer"] = "Invalid JSON"

st.subheader("Dev Chat Files")
st.dataframe(table_rows, use_container_width=True, hide_index=True)



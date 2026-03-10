import streamlit as st
from agents.ollama_agent import TestAgent
from pydantic import BaseModel
from pydantic_ai.messages import ModelResponse, ToolCallPart, TextPart
import asyncio
import uuid
import json
import os
from display_chat import display_assistant_message, save_chat_history


DEBUG=True

st.set_page_config(
    page_title="Web of Microbes Agent",
    page_icon="🕸️", # spider web icon
    layout="wide",
)

if not DEBUG:
    # Password gate
    st.write("Please enter the password to access the Web of Microbes Agent:")
    password = st.text_input("Password", type="default")  # we're not going to trigger password saver in the browser
    if password != "webofmicrobes":
        st.error("Incorrect password. Please try again.")
        st.stop()  # Stop execution if the password is incorrect

        


# Initialize the Ollama agent
agent = TestAgent(
    model_name="gpt-oss:120b",
    #model_name="gpt-oss:20b",
    base_url="http://10.10.10.238:11434/v1",
)

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "chat_id" not in st.session_state:
    st.session_state["chat_id"] = str(uuid.uuid4())

# Sidebar
st.sidebar.info(
    """This is an early-stage release of a general purpose microbiome agent for predicting and interpreting exometabolite mediated dynamics in isolates and communities. Currently, the model is limited to 226 isolates across 68 Genera and 119 metabolites so use caution when interpreting and using the results."""
)
if st.sidebar.button("🔄 New Chat", use_container_width=True):
    st.session_state["messages"] = []
    st.session_state.pop("previous_agent_result", None)
    st.session_state["chat_id"] = str(uuid.uuid4())
    st.rerun()
chat_id = st.session_state["chat_id"]
st.sidebar.markdown(
    f'<a href="/ChatHistory?chat_id={chat_id}" target="_self" style="text-decoration:none;">'
    f'<button style="width:100%; padding:0.4rem; border-radius:0.5rem; border:1px solid #ccc; cursor:pointer;">📜 Chat History</button>'
    f'</a>',
    unsafe_allow_html=True
)
with st.sidebar.expander("ℹ️ About & Help"):
    st.markdown("""
**Web of Microbes** predicts which metabolites soil microbes
prefer to consume, based on exometabolomics data from
Lawrence Berkeley National Laboratory.

**Try asking:**
- "What does Pseudomonas like to eat?"
- "Which microbes consume glutamate?"
- "Compare Burkholderia and Variovorax"
- "What is the growth rate of Pseudomonas?"

**Tips:**
- You don't need exact names — fuzzy matching will find close matches
- You can ask follow-up questions in the same chat
- Click **New Chat** to start fresh

**References**
We would like to acknowledge the following manuscripts for tools and data
- Xu, Liang, Emily Zakem, and J. L. Weissman. "Improved maximum growth rate prediction from microbial genomes by integrating phylogenetic information." Nature Communications 16.1 (2025): 4256.
""")

# Streamlit UI
theme = st.get_option("theme.base")
if theme == "dark":
    st.image("images/wom_dark_transparent.png", width=300)
else:
    st.image("images/wom_light_transparent.png", width=300)
st.caption("Explore microbial substrate preferences and growth rates using AI-assisted search. The WoM agent is a collaboration between northenlab.org and cs.ucr.edu/~mingxunw")

# Example queries
EXAMPLE_QUERIES = [
    "What does Pseudomonas like to eat?",
    "Which microbes consume glutamate?",
    "Compare Burkholderia and Variovorax",
    "Tell me about your tools",
]
ADVANCED_QUERIES = [
    {
        "title": "Rank substrates for Pseudomonas",
        "prompt": "For Pseudomonas, rank all 23 of these substrates from highest to lowest preference: phenylalanine, glycine, threonine, isoleucine, leucine, lysine, serine, alanine, proline, arginine, glutamate, adenosine, guanine, hypoxanthine, uridine, 4-guanidinobutanoic acid, glucose, fructose, mannose, arabinose, trehalose, maltose, mannitol",
    },
    {
        "title": "Rank bacteria competitiveness for amino acids",
        "prompt": "Make a table where you rank (highest to lowest) which of these bacteria you predict will be most competitive for a mixture of proteogenic amino acids based on having both a high substrate preferences and high growth rates. Acidovorax, Acinetobacter, Afipia, Arthrobacter, Bacillus, Bradyrhizobium, Brevibacillus, Brevundimonas, Burkholderia, Caulobacter, Chitinophaga, Chryseobacterium, Cupriavidus, Curvibacter, Ensifer, Flavobacterium, Herbaspirillum, Janthinobacterium, Lysinibacillus, Lysobacter, Marmoricola, Massilia, Mesorhizobium, Methylobacterium, Modestobacter, Mucilaginibacter, Niastella, Novosphingobium, Paenibacillus, Pedobacter, Phenylobacterium, Pseudomonas, Rhizobium, Rhodococcus, Sphingomonas, Stenotrophomonas, Streptomyces, Undibacterium, Zoogloea",
    },
    {
        "title": "Rank bacteria competitiveness for aldehydo-D-xylose",
        "prompt": "Make a table where you rank (highest to lowest) of which of these bacteria you predict will be most competitive for aldehydo-D-xylose based on having both a high substrate preferences and high growth rates. Acidovorax, Acinetobacter, Afipia, Arthrobacter, Bacillus, Bradyrhizobium, Brevibacillus, Brevundimonas, Burkholderia, Caulobacter, Chitinophaga, Chryseobacterium, Cupriavidus, Curvibacter, Ensifer, Flavobacterium, Herbaspirillum, Janthinobacterium, Lysinibacillus, Lysobacter, Marmoricola, Massilia, Mesorhizobium, Methylobacterium, Modestobacter, Mucilaginibacter, Niastella, Novosphingobium, Paenibacillus, Pedobacter, Phenylobacterium, Pseudomonas, Rhizobium, Rhodococcus, Sphingomonas, Stenotrophomonas, Streptomyces, Undibacterium, Zoogloea",
    },
]

example_container = st.empty()
example_query = None
if len(st.session_state["messages"]) == 0:
    with example_container.container():
        st.write("**Try an example query:**")
        cols = st.columns(4)
        for col, query in zip(cols, EXAMPLE_QUERIES):
            if col.button(query, use_container_width=True):
                example_query = query
        st.write("**Advanced queries:**")
        adv_cols = st.columns(len(ADVANCED_QUERIES))
        for col, item in zip(adv_cols, ADVANCED_QUERIES):
            if col.button(item["title"], use_container_width=True):
                example_query = item["prompt"]


# Display conversation history
for msg in st.session_state["messages"]:
    if msg["role"] == "assistant":
        with st.chat_message("assistant"):
            display_assistant_message(msg["content"])
    else:
        st.chat_message("user").write(msg["content"])

# Input box for user query at the bottom
chat_input = st.chat_input("Ask me about microbial substrate preferences, growth rates, or microbiome design...")
if user_query := (example_query or chat_input):
    example_container.empty()
    # Add user message to session state
    st.session_state["messages"].append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    # Agent response
    with st.spinner("Agent is thinking..."):
        try:
            previous_agent_result = st.session_state.get("previous_agent_result", None)

            result = agent.run_sync(user_query, previous_result=previous_agent_result)
            st.session_state["previous_agent_result"] = result  # Store the previous result for potential future use

            # Extract the 'content' field from the JSON response
            final_text = result.output  # Assuming result.output is the content field
            text_part = ""
            tool_text = ""

            # We could reformat this to find all the messages
            model_response_list = []
            for msg in result.new_messages():
                if isinstance(msg, ModelResponse):
                    for part in msg.parts:
                        print("Part:", part)  # Debug print to check each part of the message
                        if isinstance(part, ToolCallPart):
                            tool_call_info = {
                                "type": "tool_call",
                                "tool_name": part.tool_name,
                                "args": part.args
                            }
                            model_response_list.append(tool_call_info)
                            tool_text  += f"\n\nTool Call: {part.tool_name}\nArgs: {part.args}\n"
                        elif isinstance(part, TextPart):
                            text_info = {
                                "type": "text",
                                "content": part.content
                            }
                            model_response_list.append(text_info)
                            text_part += f"{part.content}\n"

            # Add agent response to session state
            st.session_state["messages"].append({"role": "assistant", "content": model_response_list})
            save_chat_history(st.session_state["chat_id"], st.session_state["messages"])
            with st.chat_message("assistant"):
                display_assistant_message(model_response_list)

        except Exception as e:
            raise
            error_message = f"An error occurred: {e}"
            st.session_state["messages"].append({"role": "assistant", "content": error_message})
            st.chat_message("assistant").write(error_message)
            
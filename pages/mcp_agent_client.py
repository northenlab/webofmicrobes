import asyncio
import threading
import streamlit as st
from streamlit.components.v1 import html
from agents.ollama_agent import TestAgent
from pydantic_ai.messages import ModelResponse, ToolCallPart, TextPart
import os
import uuid

from display_chat import display_assistant_message, save_chat_history


class AgentRunner:
    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._run_lock = None
        #self._model_name = "gemini-2.5-flash"
        self._model_name = "gemini-3-flash-preview"
        self._provider = "google"
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        future = asyncio.run_coroutine_threadsafe(self._init_loop_state(), self._loop)
        future.result()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _init_loop_state(self):
        self._run_lock = asyncio.Lock()

    async def _run(self, prompt: str, previous_messages=None):
        async with self._run_lock:
            agent_instance = TestAgent(
                model_name=self._model_name,
                provider=self._provider,
            )

            async for _kind, _message in agent_instance.iter_formatted_events(
                prompt,
                previous_result=previous_messages,
            ):
                # Consume stream events to completion; final result is stored on the agent.
                pass
            return agent_instance.latest_response

    def run(self, prompt: str, previous_messages=None):
        future = asyncio.run_coroutine_threadsafe(
            self._run(prompt, previous_messages=previous_messages),
            self._loop,
        )
        return future.result()


@st.cache_resource
def get_runner():
    return AgentRunner()

# Add a tracking token
html('<script async defer data-website-id="0b8faebf-f345-4b89-a5d2-7cbe1fd7d88d" src="https://analytics.gnps2.org/umami.js"></script>', width=0, height=0)
html('<script async defer data-website-id="74bc9983-13c4-4da0-89ae-b78209c13aaf" src="https://analytics.gnps2.org/umami.js"></script>', width=0, height=0)

st.set_page_config(
    page_title="Web of Microbes Agent",
    page_icon="🕸️", # spider web icon
    layout="wide",
)

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "chat_id" not in st.session_state:
    st.session_state["chat_id"] = str(uuid.uuid4())

# Sidebar
st.sidebar.info(
    """This is a general purpose microbiome agent for predicting and interpreting exometabolite mediated dynamics in isolates and communities. Currently, the model is limited to 433 isolates across 117 genera and 311 metabolites, with 80,834 observations. This is an early-stage release so use caution when interpreting and using the results."""
)
if st.sidebar.button("🔄 New Chat", use_container_width=True):
    st.session_state["messages"] = []
    st.session_state.pop("previous_agent_messages", None)
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
- Xu, Liang, Emily Zakem, and J. L. Weissman. "Improved maximum growth rate prediction from microbial genomes by integrating phylogenetic information." *Nature Communications* 16.1 (2025): 4256.
- Han, S., Van Treuren, W., Fischer, C.R. et al. A metabolomics pipeline for the mechanistic interrogation of the gut microbiome. *Nature* 595, 415–420 (2021). [doi:10.1038/s41586-021-03707-9](https://doi.org/10.1038/s41586-021-03707-9)
- Kosina, S.M., Greiner, A.M., Lau, R.K. et al. Web of microbes (WoM): a curated microbial exometabolomics database for linking chemistry and microbes. *BMC Microbiol* 18, 115 (2018). [doi:10.1186/s12866-018-1256-y](https://doi.org/10.1186/s12866-018-1256-y)
- Barnum, T.P., Crits-Christoph, A., Molla, M. et al. Predicting microbial growth conditions from amino acid composition. *bioRxiv* 2024.03.22.586313. [doi:10.1101/2024.03.22.586313](https://doi.org/10.1101/2024.03.22.586313)
- The UniProt Consortium. UniProt: the Universal Protein Knowledgebase in 2025. *Nucleic Acids Research* 53, D609–D617 (2025). [doi:10.1093/nar/gkae1010](https://doi.org/10.1093/nar/gkae1010)
- Malik, A. et al. ChEBI: re-engineered for a sustainable future. *Nucleic Acids Research* 54, D1768–D1778 (2026). [doi:10.1093/nar/gkaf1271](https://doi.org/10.1093/nar/gkaf1271)
- Bansal, P. et al. Rhea, the reaction knowledgebase in 2022. *Nucleic Acids Research* 50, D693–D700 (2022). [doi:10.1093/nar/gkab1016](https://doi.org/10.1093/nar/gkab1016)
""")

# Streamlit UI
theme = st.get_option("theme.base")
if theme == "dark":
    st.image("images/wom_dark_transparent.png", width=300)
else:
    st.image("images/wom_light_transparent.png", width=300)
st.caption("Explore microbial substrate preferences and growth rates using AI-assisted search. The WoM agent is a collaboration between northenlab.org and cs.ucr.edu/~mingxunw")

# Display conversation history
for msg in st.session_state["messages"]:
    if msg["role"] == "assistant":
        with st.chat_message("assistant"):
            display_assistant_message(msg["content"])
    else:
        st.chat_message("user").write(msg["content"])

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
        "prompt": "I am providing a list of bacterial genera. You must predict which will be most competitive for proteogenic amino acids (alanine, arginine, asparagine, aspartate, cysteine, glutamate, glutamine, glycine, histidine, isoleucine, leucine, lysine, methionine, phenylalanine, proline, serine, threonine, tryptophan, tyrosine, valine) by predicting the growth rate and calculating the average substrate preference for these amino acids. Perform exactly three sequential steps. Complete Step 1 fully before beginning Step 2 and Step 2 before Step 3. Step 1: Predict the doubling time for each genus. Remove any genus with a doubling time greater than 5 hours. Do not override this threshold for any reason, even if a filtered-out genus has a high proteogenic amino acids preference. Step 2: For the remaining genera only, use the calculator tool to calculate the average preference for proteogenic amino acids. Step 3: rank them from highest to lowest average substrate preference for proteogenic amino acids. Present your final answer as a numbered list with columns: rank, genus, predicted doubling time (hours), and basis for proteogenic amino acids preference ranking. List of genera: Acidovorax, Acinetobacter, Afipia, Arthrobacter, Bacillus, Bradyrhizobium, Brevibacillus, Brevundimonas, Burkholderia, Caulobacter, Chitinophaga, Chryseobacterium, Cupriavidus, Curvibacter, Ensifer, Flavobacterium, Herbaspirillum, Janthinobacterium, Lysinibacillus, Lysobacter, Marmoricola, Massilia, Mesorhizobium, Methylobacterium, Modestobacter, Mucilaginibacter, Niastella, Novosphingobium, Paenibacillus, Pedobacter, Phenylobacterium, Pseudomonas, Rhizobium, Rhodococcus, Sphingomonas, Stenotrophomonas, Streptomyces, Undibacterium, Zoogloea",
    },
    {
        "title": "Rank bacteria competitiveness for aldehydo-D-xylose",
        "prompt": "I am providing a list of bacterial genera. You must predict which will be most competitive for xylose, defined as the genus with the highest xylose preference among those capable of fast growth (doubling time ≤ 5 hours). Perform exactly two sequential steps. Complete Step 1 fully before beginning Step 2. Step 1: Predict the doubling time for each genus. Remove any genus with a doubling time greater than 5 hours. Do not override this threshold for any reason, even if a filtered-out genus has a high xylose preference. Step 2: For the remaining genera only, rank them from highest to lowest preference for xylose. Present your final answer as a numbered list with columns: rank, genus, predicted doubling time (hours), and basis for xylose preference ranking. List of genera: Acidovorax, Acinetobacter, Afipia, Arthrobacter, Bacillus, Bradyrhizobium, Brevibacillus, Brevundimonas, Burkholderia, Caulobacter, Chitinophaga, Chryseobacterium, Cupriavidus, Curvibacter, Ensifer, Flavobacterium, Herbaspirillum, Janthinobacterium, Lysinibacillus, Lysobacter, Marmoricola, Massilia, Mesorhizobium, Methylobacterium, Modestobacter, Mucilaginibacter, Niastella, Novosphingobium, Paenibacillus, Pedobacter, Phenylobacterium, Pseudomonas, Rhizobium, Rhodococcus, Sphingomonas, Stenotrophomonas, Streptomyces, Undibacterium, Zoogloea.",
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

# Input box for user query at the bottom
user_input = st.chat_input("Ask me about microbial substrate preferences, growth rates, or microbiome design...")
user_query = example_query or user_input
if user_query:
    example_container.empty()
    # Add user message to session state
    st.session_state["messages"].append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    # Agent response
    with st.spinner("Agent is thinking..."):
        try:
            runner = get_runner()
            previous_agent_messages = st.session_state.get("previous_agent_messages", None)

            result = runner.run(
                user_query,
                previous_messages=previous_agent_messages,
            )
            
            st.session_state["previous_agent_messages"] = result.all_messages()

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
            
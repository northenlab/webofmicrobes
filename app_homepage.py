import streamlit as st
from streamlit.components.v1 import html
import os
import json
import uuid
import time
import re
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

from tool_calls import execute_tool_call, parse_tool_calls
from display_chat import display_assistant_message

load_dotenv()

# Add a tracking token
html('<script async defer data-website-id="0b8faebf-f345-4b89-a5d2-7cbe1fd7d88d" src="https://analytics.gnps2.org/umami.js"></script>', width=0, height=0)
html('<script async defer data-website-id="74bc9983-13c4-4da0-89ae-b78209c13aaf" src="https://analytics.gnps2.org/umami.js"></script>', width=0, height=0)


# Write the page label
st.set_page_config(
    page_title="Web of Microbes",
    page_icon="🕸️",
    layout="wide"
)

# Configure Gemini API
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

# Model name
#MODEL_NAME = "gemini-2.0-flash"
#MODEL_NAME = "gemini-2.5-flash"
MODEL_NAME = "gemini-3-flash-preview"
#MODEL_NAME = "gemini-3-pro-preview"

@st.cache_resource
def get_gemini_client():
    """Create and cache the Gemini client."""
    return genai.Client(api_key=GEMINI_KEY)

@st.cache_data
def get_preprompt():
    """Read and cache the preprompt from disk."""
    return open("./prompts/wom_prompt_v2.txt", "r").read()

client = get_gemini_client()
llm_config = types.GenerateContentConfig(
    temperature=0.1,
    seed=42,
    top_k=1,
    system_instruction=get_preprompt(),
    thinking_config=types.ThinkingConfig(thinking_level="high")
)


# Debug mode toggle - set to False in production
DEBUG_MODE = True
DEBUG_LLM = False

def debug_log(label: str, content):
    """Write debug info to the sidebar instead of main area."""
    if DEBUG_MODE:
        with st.sidebar:
            with st.expander(f"🔧 {label}", expanded=False):
                if isinstance(content, (dict, list)):
                    st.json(content)
                else:
                    st.code(str(content), language="text")

# Sidebar disclaimer
st.sidebar.info(
    """This is a general purpose microbiome agent for predicting and interpreting exometabolite mediated dynamics in isolates and communities. Currently, the model is limited to 710 isolates across 114 genera and 347 metabolites, with 186,945 observations. This is an early-stage release so use caution when interpreting and using the results. We provide several illustrative examples and anticipate that the accuracy of the model will continue to increase with additional data."""
)

# Initialize chat_id for history link
if "chat_id" not in st.session_state:
    st.session_state["chat_id"] = str(uuid.uuid4())
if st.sidebar.button("🔄 New Chat", use_container_width=True):
    st.session_state.messages = []
    st.session_state.pop("gemini_chat", None)
    st.session_state["chat_id"] = str(uuid.uuid4())
    st.rerun()
# Add Chat History link to sidebar
chat_id = st.session_state["chat_id"]
st.sidebar.markdown(
    f'<a href="/ChatHistory?chat_id={chat_id}" target="_self" style="text-decoration:none;">'
    f'<button style="width:100%; padding:0.4rem; border-radius:0.5rem; border:1px solid #ccc; cursor:pointer;">📜 Chat History</button>'
    f'</a>',
    unsafe_allow_html=True
)
# st.sidebar.link_button("📜 Chat History", use_container_width=True, url="/ChatHistory?chat_id={}".format(st.session_state["chat_id"]))
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
    st.caption(f"Model: {MODEL_NAME}")
# Sidebar header for debug logs
if DEBUG_MODE:
    st.sidebar.title("🔧 Debug Logs")

# Header and description - use different image based on theme
theme = st.get_option("theme.base")
if theme == "dark":
    st.image("images/wom_dark_transparent.png", width=300)
else:
    st.image("images/wom_light_transparent.png", width=300)

# st.markdown("""
# <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
#             padding: 1.5rem; 
#             border-radius: 12px; 
#             border-left: 4px solid #4CAF50;
#             margin-bottom: 1.5rem;">
#     <p style="color: #e0e0e0; font-size: 1.05rem; line-height: 1.7; margin: 0;">
#         <strong style="color: #4CAF50;">Web of Microbes</strong> predicts which metabolites soil and rhizosphere 
#         microorganisms are likely to consume or secrete, using collaborative filtering models trained on 
#         experimental exometabolomics data. Preference scores reflect the likelihood of metabolite consumption 
#         based on observed patterns and do not account for growth rate or competitive fitness.
#     </p>
#     <p style="color: #b0b0b0; font-size: 0.95rem; line-height: 1.6; margin-top: 1rem; margin-bottom: 0;">
#         Ask the chatbot which compounds a microbe prefers, which microbes consume a given compound, 
#         or how metabolic preferences compare across taxa. You can also ask about predicted growth rates.
#     </p>
#     <p style="color: #888; font-size: 0.85rem; margin-top: 0.75rem; margin-bottom: 0;">
#         <em>The model is trained on exometabolomics data from Lawrence Berkeley National Laboratory.
#         Growth rate predictions are from Phydon.</em>
#     </p>
# </div>
# """, unsafe_allow_html=True)
st.caption("Explore microbial substrate preferences and growth rates using AI-assisted search.  The WoM agent is a collaboration between [northenlab.org](https://www.northenlab.org/) and [cs.ucr.edu/~mingxunw](https://www.cs.ucr.edu/~mingxunw/)")

# ── Tool call patterns ──────────────────────────────────────────────

import json
DEVELOPER_TAG_PATTERN = re.compile(r"<developer>.*?</developer>", re.DOTALL | re.IGNORECASE)

def strip_developer_tag(content: str) -> tuple[str, str, bool]:
    if not content:
        return content, "", False
    # Extract developer tags
    developer_tags = DEVELOPER_TAG_PATTERN.findall(content)
    developer_text = "\n".join(developer_tags)
    # Remove developer tags
    cleaned, count = DEVELOPER_TAG_PATTERN.subn("", content)
    return cleaned.strip(), developer_text, count > 0

def save_developer_snapshot(chat_id: str, messages: list, developer_text: str) -> None:
    developer_dir = os.path.join("/app/chat", "developer")
    os.makedirs(developer_dir, exist_ok=True)
    snapshot_path = os.path.join(developer_dir, f"{chat_id}.json")
    with open(snapshot_path, "w") as file:
        file.write(json.dumps({"messages": messages, "developer": developer_text}))




def execute_all_tool_calls(tool_calls: list) -> tuple[str, bool]:
    """
    Execute a list of tool calls and combine their outputs.

    Returns:
        Tuple of (combined_output_text, any_errors)
    """
    all_outputs = []
    any_errors = False

    for i, tool_info in enumerate(tool_calls):
        tool_name = tool_info.get("tool", "unknown")
        label = tool_info.get("query") or tool_info.get("command") or str(i)
        output, is_error = execute_tool_call(tool_info)

        if is_error:
            any_errors = True

        all_outputs.append(f"[{tool_name}: {label}]\n{output}")

        debug_log(f"Tool {i+1} ({tool_name})", {"input": tool_info, "output": output, "is_error": is_error})

    combined = "\n\n".join(all_outputs)
    return combined, any_errors




def send_message_to_llm(gemini_chat, message: str, stream: bool = True):
    """Send message to LLM and optionally stream the response.
    
    Args:
        gemini_chat: The Gemini chat session
        message: The message to send
        stream: If True, returns a generator that yields chunks. If False, returns complete response.
    
    Returns:
        Generator yielding text chunks if stream=True, otherwise returns response object
    """
    if DEBUG_LLM:
        debug_log("Sending Message to LLM", message)
        from types import SimpleNamespace
        response_obj = SimpleNamespace()
        response_obj.text = "This is a dummy response for debugging purposes. {}".format(str(uuid.uuid4()))
        import time
        time.sleep(3)  # Simulate response time
        return response_obj
    
    if stream:
        # Use streaming API
        response_stream = gemini_chat.send_message_stream(message)
        return response_stream
    else:
        # Use non-streaming API
        response = gemini_chat.send_message(message)
        return response

def gemini_chat():

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize Gemini chat session
    if "gemini_chat" not in st.session_state:
        history = []
        for msg in st.session_state.messages:
            role = "user" if msg["role"] == "user" else "model"
            history.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))
        st.session_state.gemini_chat = client.chats.create(
            model=MODEL_NAME,
            config=llm_config,
            history=history
        )

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
       with st.chat_message(message["role"]):
           if message["role"] == "user":
               st.markdown(message["content"])
           else:
               for part in message["content"]:
                    if part.get("tool_calls", False):
                        display_assistant_message(part["content"], tool_calls=True)
                    else:
                        display_assistant_message(part["content"], tool_calls=False)
           
    first_message = "Ask me about microbial substrate preferences, growth rates and microbiome design"

    # Example questions - only show if no messages yet and no button clicked
    example_container = st.empty()
    if len(st.session_state.messages) == 0 and not st.session_state.get("first_message", ""):
        with example_container.container():
            st.markdown("**Simple queries:**")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                if st.button("🦠 What does Pseudomonas like to eat?", use_container_width=True):
                    st.session_state.first_message = "What does Pseudomonas like to eat?"
                    st.session_state.button_clicked = True
                    example_container.empty()
                    st.rerun()
            with col2:
                if st.button("🧪 Which microbes consume glutamate?", use_container_width=True):
                    st.session_state.first_message = "Which microbes consume glutamate?"
                    st.session_state.button_clicked = True
                    example_container.empty()
                    st.rerun()
            with col3:
                if st.button("⚖️ Compare Burkholderia and Variovorax", use_container_width=True):
                    st.session_state.first_message = "Compare Burkholderia and Variovorax"
                    st.session_state.button_clicked = True
                    example_container.empty()
                    st.rerun()
            with col4:
                if st.button("Tell me about your tools", use_container_width=True):
                    st.session_state.first_message = "Tell me about your tools"
                    st.session_state.button_clicked = True
                    example_container.empty()
                    st.rerun()

            st.markdown("**Exploratory new tools: Predicting microbial interactions and microbiome responses:**")
            col1_adv, col2_adv, col3_adv = st.columns(3)

            with col1_adv:
                if st.button("🔬 Predict Pseudomonas [1]", use_container_width=True):
                    st.session_state.first_message = "For Pseudomonas, rank all 23 of these substrates from highest to lowest preference: phenylalanine, glycine, threonine, isoleucine, leucine, lysine, serine, alanine, proline, arginine, glutamate, adenosine, guanine, hypoxanthine, uridine, 4-guanidinobutanoic acid, glucose, fructose, mannose, arabinose, trehalose, maltose, mannitol"
                    st.session_state.button_clicked = True
                    st.rerun()
            with col2_adv:
                if st.button("🏆 Soil amino acid spike-in [2]", use_container_width=True):
                    st.session_state.first_message = "Make a table where you rank (highest to lowest) which of these bacteria you predict will be most competitive for a mixture of proteogenic amino acids based on having both a high substrate preferences and high growth rates. Acidovorax, Acinetobacter, Afipia, Arthrobacter, Bacillus, Bradyrhizobium, Brevibacillus, Brevundimonas, Burkholderia, Caulobacter, Chitinophaga, Chryseobacterium, Cupriavidus, Curvibacter, Ensifer, Flavobacterium, Herbaspirillum, Janthinobacterium, Lysinibacillus, Lysobacter, Marmoricola, Massilia, Mesorhizobium, Methylobacterium, Modestobacter, Mucilaginibacter, Niastella, Novosphingobium, Paenibacillus, Pedobacter, Phenylobacterium, Pseudomonas, Rhizobium, Rhodococcus, Sphingomonas, Stenotrophomonas, Streptomyces, Undibacterium, Zoogloea?"
                    st.session_state.button_clicked = True
                    st.rerun()
            with col3_adv:
                if st.button("🍬 Soil xylose spike-in [3]", use_container_width=True):
                    st.session_state.first_message = "Make a table where you rank (highest to lowest) of which of these bacteria you predict will be most competitive for aldehydo-D-xylose based on having both a high substrate preferences and high growth rates. Acidovorax, Acinetobacter, Afipia, Arthrobacter, Bacillus, Bradyrhizobium, Brevibacillus, Brevundimonas, Burkholderia, Caulobacter, Chitinophaga, Chryseobacterium, Cupriavidus, Curvibacter, Ensifer, Flavobacterium, Herbaspirillum, Janthinobacterium, Lysinibacillus, Lysobacter, Marmoricola, Massilia, Mesorhizobium, Methylobacterium, Modestobacter, Mucilaginibacter, Niastella, Novosphingobium, Paenibacillus, Pedobacter, Phenylobacterium, Pseudomonas, Rhizobium, Rhodococcus, Sphingomonas, Stenotrophomonas, Streptomyces, Undibacterium, Zoogloea"
                    st.session_state.button_clicked = True
                    st.rerun()

            # Adding in a section of citations
            st.markdown('[1] Jenkins, Stefan, et al. "Construction of viable soil defined media using quantitative metabolomics analysis of soil metabolites." Frontiers in Microbiology 8 (2017): 2618.')
            st.markdown('[2] bioRxiv Manuscript (Experimental result = Pseudomonas, most enriched and Paenibacillus, also enriched) ')
            st.markdown('[3] bioRxiv Manuscript (Experimental result= Novosphingobium)')



    if "first_message" not in st.session_state:
        st.session_state.first_message = ""

    # Auto-submit if button was clicked
    prompt = None
    if st.session_state.get("button_clicked", False):
        prompt = st.session_state.first_message
        st.session_state.button_clicked = False
    
    # Always show chat input
    if user_input := st.chat_input(first_message, key="chat_input"):
        prompt = user_input

    if prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Lets allow a loop of maximum 6 iterations of tool calls and LLM responses to prevent infinite loops in case of errors
            MAX_ITERATIONS = 6
            CURRENT_ITERATION = 0
            all_responses = []
            developer_tag_found = False
            for _ in range(MAX_ITERATIONS):
                CURRENT_ITERATION += 1

                # Send user's message to LLM and get a streaming response
                response_stream = send_message_to_llm(st.session_state.gemini_chat, prompt, stream=True)
                
                # Accumulate the streamed response
                assistant_response = ""
                message_placeholder = st.empty()
                
                for chunk in response_stream:
                    if hasattr(chunk, 'text') and chunk.text:
                        assistant_response += chunk.text
                        # Update the display with accumulated text
                        message_placeholder.markdown(assistant_response + "▌")
                
                assistant_response, developer_text, developer_hit = strip_developer_tag(assistant_response)
                developer_tag_found = developer_tag_found or developer_hit

                # Remove the cursor and show final text
                message_placeholder.markdown(assistant_response)

                # now lets check for tool calls
                tool_calls = parse_tool_calls(assistant_response)

                all_responses.append({"content": assistant_response, "tool_calls": len(tool_calls) > 0})

                if tool_calls:
                    # Clear the placeholder and show formatted version with tool calls
                    message_placeholder.empty()
                    display_assistant_message(assistant_response, tool_calls=True)
                    notice_text = """---

Executing tool calls...
                                
---
                                """
                    display_assistant_message(notice_text)
                    all_responses.append({"content": notice_text, "tool_calls": True})
                else:
                    # For non-tool-call responses, the streaming display is already shown
                    all_responses.append({"content": assistant_response, "tool_calls": False})

                if tool_calls:
                    # Execute all tool calls
                    combined_output, any_errors = execute_all_tool_calls(tool_calls)

                    # there is a tool call, which means we should send the results back to the LLM for interpretation
                    if not any_errors:
                        # Send tool outputs back to LLM for interpretation
                        tool_msg = f"Tool outputs:\n```json\n{combined_output}\n```\n\nPlease interpret all of these results together for the user. Just a note, you are allowed to do recursive tool calls, this is currently {CURRENT_ITERATION} of {MAX_ITERATIONS}."

                        # TODO: If we're on the last one, we should give it a special prompt that says "This is the last iteration, you must not make any more tool calls after this, just give a final answer to the user based on all the information you have, including the tool outputs from this message"

                        prompt = tool_msg
                        
                        continue
                    else:
                        error_msg = f"""One or more tool calls returned errors:
                                ```json
                                {combined_output}
                                ```
                                """
                        prompt = error_msg
                        continue
                else:
                    break  # No tool calls, exit the loop

            st.session_state.messages.append({"role": "assistant", "content": all_responses})

            if developer_tag_found:
                save_developer_snapshot(st.session_state["chat_id"], st.session_state.messages, developer_text)

            # Saving the chat history
            chat_history_filename = os.path.join("/app/chat/{}.json".format(st.session_state["chat_id"]))
            
            with open(chat_history_filename, "w") as file:
                file.write(json.dumps(st.session_state.messages))

## Make a password to block open access
gemini_chat()
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

from tool_calls import execute_tool_call

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
    return open("./prompts/wom_prompt.txt", "r").read()

client = get_gemini_client()
llm_config = types.GenerateContentConfig(
    temperature=0.1,
    seed=42,
    top_k=1,
    system_instruction=get_preprompt()
)


# Debug mode toggle - set to False in production
DEBUG_MODE = False
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
st.caption("Explore microbial substrate preferences and growth rates using AI-assisted search.")

# ── Tool call patterns ──────────────────────────────────────────────

import json
import re

TOOL_CALL_PATTERN = re.compile(r"```tool_call\s*\n(.*?)\n```", re.DOTALL)

def parse_tool_calls(llm_response: str) -> list:
    calls = []
    for match in TOOL_CALL_PATTERN.finditer(llm_response):
        try:
            call = json.loads(match.group(1).strip())
            calls.append(call)
        except json.JSONDecodeError:
            continue
    return calls



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


def display_assistant_message(content: str, tool_calls: bool = False):
    """
    Display assistant message, hiding tool calls in expanders.
    """    
    # Find all tool call matches
    matches = list(TOOL_CALL_PATTERN.finditer(content))
    
    if not matches:
        # No tool calls, display normally
        st.markdown(content)
        return
    
    
    # Track position in content
    last_end = 0

    # Hiding this part
    with st.expander(f"🔧 LLM Response with Tool Call(s)", expanded=False):
        # Process each tool call
        for i, match in enumerate(matches):
            # Display text before this tool call
            text_before = content[last_end:match.start()].strip()
            if text_before:
                st.markdown(text_before)
            
            # Display tool call in expander
            tool_json = match.group(1).strip()
            try:
                tool_data = json.loads(tool_json)
                tool_name = tool_data.get("tool", "unknown")
                tool_label = tool_data.get("query") or tool_data.get("command") or f"call {i+1}"
                
                with st.expander(f"🔧 Tool Call: {tool_name} - {tool_label}", expanded=False):
                    st.code(json.dumps(tool_data, indent=2), language="json")
            except json.JSONDecodeError:
                with st.expander(f"🔧 Tool Call {i+1}", expanded=False):
                    st.code(tool_json, language="json")
            
            last_end = match.end()
        
        # Display any remaining text after the last tool call
        text_after = content[last_end:].strip()
        if text_after:
            st.markdown(text_after)

def send_message_to_llm(gemini_chat, message: str) -> str:
    if DEBUG_LLM:
        debug_log("Sending Message to LLM", message)
        from types import SimpleNamespace
        response_obj = SimpleNamespace()
        response_obj.text = "This is a dummy response for debugging purposes. {}".format(str(uuid.uuid4()))
        import time
        time.sleep(3)  # Simulate response time

        return response_obj
    
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
    if len(st.session_state.messages) == 0 and not st.session_state.get("first_message", ""):
        st.markdown("**Try an example question:**")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🦠 What does Pseudomonas like to eat?", use_container_width=True):
                st.session_state.first_message = "What does Pseudomonas like to eat?"
                st.session_state.button_clicked = True
                st.rerun()
        with col2:
            if st.button("🧪 Which microbes consume glutamate?", use_container_width=True):
                st.session_state.first_message = "Which microbes consume glutamate?"
                st.session_state.button_clicked = True
                st.rerun()
        with col3:
            if st.button("⚖️ Compare Burkholderia and Variovorax", use_container_width=True):
                st.session_state.first_message = "Compare Burkholderia and Variovorax"
                st.session_state.button_clicked = True
                st.rerun()

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
            # Lets allow a loop of maximum 4 iterations of tool calls and LLM responses to prevent infinite loops in case of errors
            all_responses = []
            for _ in range(4):
                # Send user's message to LLM and get a response
                response = send_message_to_llm(st.session_state.gemini_chat, prompt)
                assistant_response = response.text

                # now lets check for tool calls
                tool_calls = parse_tool_calls(assistant_response)

                all_responses.append({"content": assistant_response, "tool_calls": len(tool_calls) > 0})

                if tool_calls:
                    display_assistant_message(assistant_response, tool_calls=True)
                    notice_text = """---

Executing tool calls...
                                
---
                                """
                    display_assistant_message(notice_text)
                    all_responses.append({"content": notice_text, "tool_calls": True})
                else:
                    display_assistant_message(assistant_response)
                    all_responses.append({"content": assistant_response, "tool_calls": False})

                if tool_calls:
                    # Execute all tool calls
                    combined_output, any_errors = execute_all_tool_calls(tool_calls)

                    # there is a tool call, which means we should send the results back to the LLM for interpretation
                    if not any_errors:
                        # Send tool outputs back to LLM for interpretation
                        tool_msg = f"Tool outputs:\n```json\n{combined_output}\n```\n\nPlease interpret all of these results together for the user."
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

            # Saving the chat history
            chat_history_filename = os.path.join("/app/chat/{}.json".format(st.session_state["chat_id"]))
            
            with open(chat_history_filename, "w") as file:
                file.write(json.dumps(st.session_state.messages))


    # Check if an example question was clicked
    # prompt = None


    


    # # Accept user input
    # if user_input := st.chat_input(first_message):
    #     prompt = user_input

    # # Character limit
    # MAX_CHARS = 2000


    # if prompt:
    #     # Check character limit
    #     if len(prompt) > MAX_CHARS:
    #         st.warning(f"⚠️ Your message is too long ({len(prompt)} characters). Please limit to {MAX_CHARS} characters.")
    #         return

    #     # Add user message to chat history
    #     st.session_state.messages.append({"role": "user", "content": prompt})

    #     # Display user message
    #     with st.chat_message("user"):
    #         st.markdown(prompt)

    #     with st.chat_message("assistant"):
    #         # Send user's message to LLM and get a response
    #         response = send_message_to_llm(st.session_state.gemini_chat, prompt)
    #         assistant_response = response.text

    #         # Check for tool calls in the response
    #         #tool_calls = parse_tool_calls(assistant_response)

    #         #debug_log("Parsed Tool Calls", tool_calls)
            
    #         st.session_state.messages.append({"role": "assistant", "content": assistant_response})

    #         #with st.chat_message("assistant"):
    #         display_assistant_message(assistant_response)

        #st.write("here2", str(uuid.uuid4()), tool_calls)

        # if tool_calls:
        #     # Execute all tool calls
        #     combined_output, any_errors = execute_all_tool_calls(tool_calls)

        #     # Add the initial response to history
        #     st.session_state.messages.append({"role": "assistant", "content": assistant_response})

        #     # Display the initial response (with tool calls)
        #     with st.chat_message("assistant"):
        #         display_assistant_message(assistant_response)
        
        #     # No errors - send all tool outputs back to LLM for interpretation
        #     tool_msg = f"Tool outputs:\n```json\n{combined_output}\n```\n\nPlease interpret all of these results together for the user."

        #     debug_log("Tool Message", tool_msg)

        #     # Send to LLM
        #     final_response = send_message_to_llm(st.session_state.gemini_chat, tool_msg)

        #     debug_log("Final Response", final_response.text)

        #     final_text = final_response.text

        #     st.session_state.messages.append({"role": "assistant", "content": final_text})

        #     with st.chat_message("assistant"):
        #         st.markdown(final_text)
        # else:
        #     # No tool calls, just display the response normally
        #     st.session_state.messages.append({"role": "assistant", "content": assistant_response})

        #     with st.chat_message("assistant"):
        #         st.markdown(assistant_response)

        # Saving the chat history
        # chat_history_filename = os.path.join("/app/chat/{}.json".format(st.session_state["chat_id"]))
        
        # with open(chat_history_filename, "w") as file:
        #     file.write(json.dumps(st.session_state.messages))

if True:
    gemini_chat()
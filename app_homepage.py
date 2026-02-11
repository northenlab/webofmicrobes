import streamlit as st
from streamlit.components.v1 import html
import os
import json
import uuid
import time
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Add a tracking token
html('<script async defer data-website-id="0b8faebf-f345-4b89-a5d2-7cbe1fd7d88d" src="https://analytics.gnps2.org/umami.js"></script>', width=0, height=0)
html('<script async defer data-website-id="74bc9983-13c4-4da0-89ae-b78209c13aaf" src="https://analytics.gnps2.org/umami.js"></script>', width=0, height=0)


# Write the page label
st.set_page_config(
    page_title="Web of Microbes Chatbot",
    page_icon="👋",
)

# Configure Gemini API
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

# Model name
#MODEL_NAME = "gemini-2.0-flash"
MODEL_NAME = "gemini-2.5-flash"
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
DEBUG_MODE = True

def debug_log(label: str, content):
    """Write debug info to the sidebar instead of main area."""
    if DEBUG_MODE:
        with st.sidebar:
            with st.expander(f"🔧 {label}", expanded=False):
                if isinstance(content, dict):
                    st.json(content)
                else:
                    st.code(str(content), language="text")

# Initialize chat_id for history link
if "chat_id" not in st.session_state:
    st.session_state["chat_id"] = str(uuid.uuid4())

# Add Chat History link to sidebar
st.sidebar.link_button("📜 Chat History", "/ChatHistory?chat_id={}".format(st.session_state["chat_id"]))

# Sidebar header for debug logs
if DEBUG_MODE:
    st.sidebar.title("🔧 Debug Logs")

# Header and description
st.image("images/wom_logo.png")

st.markdown("""
<div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
            padding: 1.5rem; 
            border-radius: 12px; 
            border-left: 4px solid #4CAF50;
            margin-bottom: 1.5rem;">
    <p style="color: #e0e0e0; font-size: 1.05rem; line-height: 1.7; margin: 0;">
        <strong style="color: #4CAF50;">Web of Microbes</strong> predicts which metabolites soil and rhizosphere 
        microorganisms are likely to consume or secrete, using collaborative filtering models trained on 
        experimental exometabolomics data. Preference scores reflect the likelihood of metabolite consumption 
        based on observed patterns and do not account for growth rate or competitive fitness.
    </p>
    <p style="color: #b0b0b0; font-size: 0.95rem; line-height: 1.6; margin-top: 1rem; margin-bottom: 0;">
        Ask the chatbot which compounds a microbe prefers, which microbes consume a given compound, 
        or how metabolic preferences compare across taxa.
    </p>
    <p style="color: #888; font-size: 0.85rem; margin-top: 0.75rem; margin-bottom: 0;">
        <em>The model is trained on exometabolomics data from Lawrence Berkeley National Laboratory.</em>
    </p>
</div>
""", unsafe_allow_html=True)

import re
import requests

# Match "wom.py" command anywhere in text (with or without "I'll run:")
WOM_PATTERN = re.compile(
    r"wom\.py\s+((?:microbes|compounds|compare|search-compound|search-taxon|list-taxa|list-compounds|info)\s*[^\n]*)",
    re.IGNORECASE
)

def parse_args(cmd_args: str) -> list:
    """
    Parse command arguments, handling quoted strings properly.
    """
    args = []
    current = ""
    in_quotes = False
    quote_char = None
    for char in cmd_args:
        if char in ('"', "'") and not in_quotes:
            in_quotes = True
            quote_char = char
        elif char == quote_char and in_quotes:
            in_quotes = False
            quote_char = None
        elif char == ' ' and not in_quotes:
            if current:
                args.append(current)
                current = ""
        else:
            current += char
    if current:
        args.append(current)
    return args


def parse_tool_call(llm_response: str) -> dict:
    """
    Parse the LLM response to extract tool calls.
    Returns a dict structured for POST requests or None if no tool call found.
    
    Returns:
        {
            "command": "microbes",  # the subcommand
            "compounds": "glutamate,acetate",  # --compounds value
            "level": "genus",  # --level value
            "top": "10",  # --top value
            ...
        }
    """
    # Check for wom.py commands
    wom_match = WOM_PATTERN.search(llm_response)
    if wom_match:
        cmd_args = wom_match.group(1).strip()
        args = parse_args(cmd_args)
        
        if not args:
            return None
        
        # Build the params dict
        params = {
            "command": args[0]  # First arg is the subcommand
        }
        
        # Parse remaining args as --flag value pairs
        i = 1
        while i < len(args):
            arg = args[i]
            if arg.startswith("--"):
                key = arg[2:]  # Remove the --
                # Check if there's a value following
                if i + 1 < len(args) and not args[i + 1].startswith("--"):
                    params[key] = args[i + 1]
                    i += 2
                else:
                    # Flag without value (boolean flag)
                    params[key] = True
                    i += 1
            else:
                # Positional argument after command
                i += 1
        
        return params
    
    
    return None


def execute_tool_call(tool_info: dict) -> tuple[str, bool]:
    """
    Execute a tool call via web API.
    
    Args:
        tool_info: dict from parse_tool_call, ready to be sent as POST JSON
        
    Returns:
        Tuple of (response_text, is_error)
    """
    command = tool_info.get("command")
    
    if command:
        # WoM API call - the tool_info dict is ready to POST directly
        url = "http://wom-flask:5000/api/query"

        try:
            response = requests.post(url, json=tool_info, timeout=30)
            response_text = response.text
            
            # Check for HTTP errors
            if response.status_code != 200:
                return response_text, True
            
            # Check for API-level errors in the response
            try:
                result = json.loads(response_text)
                if result.get("error") or result.get("success") == False:
                    return response_text, True
            except json.JSONDecodeError:
                pass
            
            return response_text, False
            
        except requests.RequestException as e:
            return json.dumps({"error": f"Request failed: {str(e)}"}), True
    
    return json.dumps({"error": "Unknown command"}), True


def gemini_chat():

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize Gemini chat session
    if "gemini_chat" not in st.session_state:
        # Convert existing messages to Gemini format if any
        history = []
        for msg in st.session_state.messages:
            role = "user" if msg["role"] == "user" else "model"
            history.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))
        st.session_state.gemini_chat = client.chats.create(
            model=MODEL_NAME,
            config=llm_config,
            history=history
        )

    first_message = "Hello, I am here to help you explore microbial substrate preferences"
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Example questions - only show if no messages yet
    if len(st.session_state.messages) == 0:
        st.markdown("**Try an example question:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🦠 What does Pseudomonas like to eat?", use_container_width=True):
                st.session_state.example_question = "What does Pseudomonas like to eat?"
                st.rerun()
        with col2:
            if st.button("🧪 Which microbes consume glutamate?", use_container_width=True):
                st.session_state.example_question = "Which microbes consume glutamate?"
                st.rerun()
        with col3:
            if st.button("⚖️ Compare Burkholderia and Variovorax", use_container_width=True):
                st.session_state.example_question = "Compare Burkholderia and Variovorax"
                st.rerun()

    # Check if an example question was clicked
    prompt = None
    if "example_question" in st.session_state:
        prompt = st.session_state.example_question
        del st.session_state.example_question
    
    # Accept user input
    if user_input := st.chat_input(first_message):
        prompt = user_input
    
    # Character limit
    MAX_CHARS = 500
    
    if prompt:
        # Check character limit
        if len(prompt) > MAX_CHARS:
            st.warning(f"⚠️ Your message is too long ({len(prompt)} characters). Please limit to {MAX_CHARS} characters.")
            return
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Send user's message to Gemini and get a response
        response = st.session_state.gemini_chat.send_message(prompt)
        assistant_response = response.text
        
        # Check for tool calls in the response
        tool_info = parse_tool_call(assistant_response)
        #tool_info = None

        debug_log("Parsed Tool Info", tool_info)
        
        if tool_info:
            # Execute the tool call
            tool_output, is_error = execute_tool_call(tool_info)
            
            debug_log("Tool Output", {"output": tool_output, "is_error": is_error})
            
            # Add the initial response to history
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            
            # Display the initial response (with tool call)
            with st.chat_message("assistant"):
                st.markdown(assistant_response)
            
            # If there was an error, give the LLM one chance to fix it
            if is_error:
                debug_log("Error Detected", "Giving LLM one chance to fix")
                
                error_msg = f"""The tool call returned an error:
```json
{tool_output}
```

Please analyze the error and try again with a corrected command. Make sure to use the exact wom.py syntax."""
                
                # Get corrected response from LLM
                retry_response = st.session_state.gemini_chat.send_message(error_msg)
                retry_text = retry_response.text
                
                debug_log("Retry Response", retry_text)
                
                # Check if the retry contains a new tool call
                retry_tool_info = parse_tool_call(retry_text)
                
                if retry_tool_info:
                    debug_log("Retry Tool Info", retry_tool_info)
                    
                    # Execute the retry
                    tool_output, is_error = execute_tool_call(retry_tool_info)
                    
                    debug_log("Retry Tool Output", {"output": tool_output, "is_error": is_error})
                    
                    # Add retry response to history and display
                    st.session_state.messages.append({"role": "assistant", "content": retry_text})
                    with st.chat_message("assistant"):
                        st.markdown(retry_text)
                    
                    if is_error:
                        # Still failed after retry - notify user
                        final_text = f"I tried to fix the command but still encountered an error. The API returned:\n```json\n{tool_output}\n```\n\nPlease check your query and try again with different parameters."
                        st.session_state.messages.append({"role": "assistant", "content": final_text})
                        with st.chat_message("assistant"):
                            st.markdown(final_text)
                    else:
                        # Retry succeeded - interpret results
                        tool_msg = f"Tool output:\n```json\n{tool_output}\n```\n\nPlease interpret these results for the user."
                        final_response = st.session_state.gemini_chat.send_message(tool_msg)
                        final_text = final_response.text
                        
                        st.session_state.messages.append({"role": "assistant", "content": final_text})
                        with st.chat_message("assistant"):
                            st.markdown(final_text)
                else:
                    # LLM didn't provide a corrected command, just show its response
                    st.session_state.messages.append({"role": "assistant", "content": retry_text})
                    with st.chat_message("assistant"):
                        st.markdown(retry_text)
            else:
                # No error - send tool output back to Gemini for interpretation
                tool_msg = f"Tool output:\n```json\n{tool_output}\n```\n\nPlease interpret these results for the user."

                debug_log("Tool Message", tool_msg)
                
                # Get the final interpreted response
                final_response = st.session_state.gemini_chat.send_message(tool_msg)
        
                debug_log("Final Response", final_response.text)

                final_text = final_response.text
                
                # Add the interpretation to history
                st.session_state.messages.append({"role": "assistant", "content": final_text})
                
                # Display the final interpretation
                with st.chat_message("assistant"):
                    st.markdown(final_text)
        else:
            # No tool call, just display the response normally
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            
            # Display Gemini's response
            with st.chat_message("assistant"):
                st.markdown(assistant_response)

        # Saving the chat history
        chat_history_filename = os.path.join("/app/chat/{}.json".format(st.session_state["chat_id"]))
        with open(chat_history_filename, "w") as file:
            file.write(json.dumps(st.session_state.messages))


# password question
# password = st.text_input("Access Password:", value="", type="password")

#if password == os.getenv("ACCESSPW"):
if True:

    # Draw a vertical line
    #st.markdown("---")

    #openaiassistantchat()
    gemini_chat()
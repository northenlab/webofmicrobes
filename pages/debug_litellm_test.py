import streamlit as st
import os
from litellm import completion
from dotenv import load_dotenv

# 2. Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# 1. Set up the Streamlit UI
st.title("✨ LiteLLM + Streamlit Hello World (Gemini)")
st.write("A simple app to demonstrate streaming responses using LiteLLM and Google Gemini.")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display all previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Handle the chat interaction
if prompt := st.chat_input("Say hello to the AI!"):
        
    # Temporarily set the environment variable for LiteLLM
    # NOTE: Changed from OPENAI_API_KEY to GEMINI_API_KEY
    os.environ["GEMINI_API_KEY"] = api_key

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user's prompt in the chat
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display the assistant's streamed response
    with st.chat_message("assistant"):
        
        # Call LiteLLM with stream=True, passing the entire chat history
        # NOTE: Added the "gemini/" prefix to route to Google AI Studio
        response = completion(
            model="gemini/gemini-3-flash-preview", 
            messages=st.session_state.messages,
            stream=True
        )

        # Streamlit 1.31+ allows you to use st.write_stream with a generator
        def stream_parser(stream_response):
            for chunk in stream_response:
                # Extract the text delta from the LiteLLM chunk
                content = chunk.choices[0].delta.content
                if content:
                    yield content

        # Render the streaming text in real-time and collect the full response
        full_response = st.write_stream(stream_parser(response))
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
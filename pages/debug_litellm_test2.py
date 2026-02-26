import streamlit as st
import os
from litellm import completion
from dotenv import load_dotenv

# 2. Load environment variables
load_dotenv()
proxy_api_key = os.getenv("LITELLM_PROXY_API_KEY", None)  # Optional: API key for proxy auth
proxy_url = os.getenv("LITELLM_PROXY_URL", "http://localhost:4000")  # Default proxy URL

# 1. Set up the Streamlit UI
st.title("✨ LiteLLM Proxy + Streamlit Hello World")
st.write("A simple app to demonstrate streaming responses using LiteLLM Proxy.")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display all previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Handle the chat interaction
if prompt := st.chat_input("Say hello to the AI!"):
        
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user's prompt in the chat
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display the assistant's streamed response
    with st.chat_message("assistant"):
        
        # Call LiteLLM Proxy with stream=True, passing the entire chat history
        # The proxy handles routing to the appropriate model
        response = completion(
            model="litellm_proxy/gptoss120b",  # Model name as configured in proxy
            messages=st.session_state.messages,
            api_base=proxy_url,
            api_key=proxy_api_key,
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
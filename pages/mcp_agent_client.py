import streamlit as st
from agents.ollama_agent import TestAgent
from pydantic import BaseModel
from pydantic_ai.messages import ModelResponse, ToolCallPart, TextPart
import asyncio 

st.set_page_config(
    page_title="Web of Microbes Agent",
    page_icon="🕸️", # spider web icon
    layout="wide",
)

# Password gate
st.write("Please enter the password to access the Web of Microbes Agent:")
password = st.text_input("Password", type="password")
if password != "webofmicrobes":
    st.error("Incorrect password. Please try again.")
    st.stop()  # Stop execution if the password is incorrect


def display_assistant_message(agent_message_list):
    """
    Display assistant message, hiding tool calls in expanders.
    """

    for msg in agent_message_list:
        print("Agent message part:", msg)  # Debug print to check each part of the agent message

        if msg["type"] == "text":
            st.markdown(msg["content"])
        elif msg["type"] == "tool_call":
            with st.expander(f"Tool Call: {msg['tool_name']}", expanded=False):
                st.code(f"Tool Call: {msg['tool_name']} --- Args: {msg['args']}", language="json")
        
        


# Initialize the Ollama agent
agent = TestAgent(
    model_name="gpt-oss:120b",
    #model_name="gpt-oss:20b",
    base_url="http://10.10.10.238:11434/v1",
)

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Streamlit UI
st.title("🕸️ Web of Microbes")
st.caption("Explore microbial substrate preferences and growth rates using AI-assisted search. The WoM agent is a collaboration between northenlab.org and cs.ucr.edu/~mingxunw")

# Display conversation history
for msg in st.session_state["messages"]:
    if msg["role"] == "assistant":
        with st.chat_message("assistant"):
            display_assistant_message(msg["content"])
    else:
        st.chat_message("user").write(msg["content"])

# Input box for user query at the bottom
if user_query := st.chat_input("Ask me about microbial substrate preferences, growth rates, or microbiome design..."):
    # Add user message to session state
    st.session_state["messages"].append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    # Agent response
    with st.spinner("Agent is thinking..."):
        try:
            previous_agent_result = st.session_state.get("previous_agent_result", None)
            #if previous_result:
            #    print("Previous result:", previous_result.new_messages())  # Debug print to check the previous result
            #    print(len(previous_result.all_messages()))  # Debug print to check the number of messages in the previous result
            #    print(len(previous_result.new_messages()))  # Debug print to check the content of the first message in the previous result

            # Run the agent and get the result
            #result = agent.run_sync(user_query, previous_result=previous_result)


            #with st.empty():
            #    result = st.streaming(agent.run_streaming_sync(user_query, previous_result=previous_result))
                
            result = agent.run_sync(user_query, previous_result=previous_agent_result)

            #for msg in result.new_messages():
            #    print("New message:", msg)  # Debug print to check each new message in the result
            


            #print("Agent result:", result)  # Debug print to check the raw result

            #print("Agent result output:", result.all_messages()) # Debug print to check the output field of the result
            
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
            with st.chat_message("assistant"):
                display_assistant_message(model_response_list)

        except Exception as e:
            raise
            error_message = f"An error occurred: {e}"
            st.session_state["messages"].append({"role": "assistant", "content": error_message})
            st.chat_message("assistant").write(error_message)
            
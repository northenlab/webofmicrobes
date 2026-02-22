import json
import re
import streamlit as st

from tool_calls import TOOL_CALL_PATTERN

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
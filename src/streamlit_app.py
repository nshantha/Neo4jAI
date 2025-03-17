import streamlit as st
import asyncio
import logging
from typing import Dict, List, Any
import json
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import STREAMLIT_TITLE
from src.neo4j_client import Neo4jClient
from src.llm_engine import LLMEngine
from src.langgraph_agent import Neo4jReActAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main function for the Streamlit app."""
    # Set page title and icon
    st.set_page_config(
        page_title=STREAMLIT_TITLE,
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Set page title
    st.title(STREAMLIT_TITLE)
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize components
    if "neo4j_client" not in st.session_state:
        st.session_state.neo4j_client = Neo4jClient()
    
    if "llm_engine" not in st.session_state:
        st.session_state.llm_engine = LLMEngine()
    
    if "agent" not in st.session_state:
        st.session_state.agent = Neo4jReActAgent(
            neo4j_client=st.session_state.neo4j_client,
            llm_engine=st.session_state.llm_engine
        )
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This app allows you to query a Neo4j database using natural language.
        
        Simply type your question, and the app will:
        1. Translate it to a Cypher query
        2. Execute the query against the Neo4j database
        3. Explain the results in natural language
        
        For debugging, start your query with "debug:" to see the generated Cypher query and raw results.
        """)
        
        st.header("Examples")
        example_queries = [
            "What are all the node labels in the database?",
            "Show me the database schema",
            "Find all Promotion nodes with status 'Active'",
            "What relationship types exist in the database?",
            "Show me 10 Promotion nodes"
        ]
        
        for query in example_queries:
            if st.button(query):
                st.session_state.query = query
        
        st.header("Schema Options")
        if st.button("Get Simplified Schema"):
            # Add user message to chat
            query = "Show me a simplified schema with only the main nodes and relationships"
            st.session_state.messages.append({"role": "user", "content": query})
            
            # Display user message in main area
            with st.chat_message("user"):
                st.markdown(query)
            
            # Display assistant response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("Retrieving simplified schema...")
                
                try:
                    # Get simplified schema
                    simplified_schema = asyncio.run(st.session_state.neo4j_client.get_simplified_schema())
                    
                    # Format the response
                    schema_response = "### Simplified Database Schema\n\n"
                    schema_response += "This shows only the main nodes and their relationships:\n\n"
                    
                    # Add nodes
                    schema_response += "#### Main Node Types\n"
                    for node in simplified_schema:
                        schema_response += f"- **{node['label']}**\n"
                        if node['properties']:
                            schema_response += f"  - Properties: {', '.join(node['properties'][:5])}\n"
                    
                    # Add relationships
                    schema_response += "\n#### Relationships\n"
                    for node in simplified_schema:
                        if node['relationships']:
                            for rel in node['relationships']:
                                schema_response += f"- **{node['label']}** --[{rel['type']}]--> **{rel['target']}**\n"
                    
                    # Update the message
                    message_placeholder.markdown(schema_response)
                    
                    # Add to chat history
                    st.session_state.messages.append({"role": "assistant", "content": schema_response})
                except Exception as e:
                    error_message = f"Error retrieving simplified schema: {str(e)}"
                    message_placeholder.markdown(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if query := st.chat_input("Ask a question about your Neo4j database..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(query)
        
        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            
            try:
                # Process the query
                response = asyncio.run(st.session_state.agent.process_query(query))
                
                # Check if it's a debug query
                if query.lower().startswith("debug:"):
                    # Get the state for debugging
                    debug_state = asyncio.run(st.session_state.agent.get_debug_state(query[6:].strip()))
                    
                    # Format the debug response
                    debug_response = f"{response}\n\n**Debug Info:**\n```cypher\n{debug_state.get('cypher_query', 'No Cypher query generated')}\n```\n\n**Results:**\n```json\n{json.dumps(debug_state.get('query_results', []), indent=2)}\n```"
                    
                    # Update the message
                    message_placeholder.markdown(debug_response)
                    
                    # Add to chat history
                    st.session_state.messages.append({"role": "assistant", "content": debug_response})
                else:
                    # Update the message
                    message_placeholder.markdown(response)
                    
                    # Add to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_message = f"Error: {str(e)}"
                message_placeholder.markdown(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

if __name__ == "__main__":
    main()

import logging
from typing import Any, Dict, List, Optional, Union, TypedDict, Annotated, Literal, Callable
import json
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, create_react_agent
import openai
import sys
import os
import subprocess
import asyncio

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import OPENAI_API_KEY, LLM_MODEL
from src.neo4j_client import Neo4jClient
from src.llm_engine import LLMEngine
from src.prompts import GENERIC_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Define the state schema
class AgentState(TypedDict):
    """State for the ReAct agent."""
    query: str
    schema: Optional[List[Dict[str, Any]]]
    cypher_query: Optional[str]
    query_results: Optional[List[Dict[str, Any]]]
    response: Optional[str]
    error: Optional[str]
    attempts: int
    max_attempts: int
    research_results: Optional[str]
    messages: List[Dict[str, Any]]

class Neo4jReActAgent:
    """ReAct agent for Neo4j query translation and execution."""
    
    def __init__(self, neo4j_client: Neo4jClient, llm_engine: LLMEngine):
        """Initialize the Neo4j ReAct agent."""
        self.neo4j_client = neo4j_client
        self.llm_engine = llm_engine
        self.graph = self.build_graph()
        self.react_agent = self._create_react_agent()
    
    def get_empty_state(self) -> AgentState:
        """Get an empty state for the agent."""
        return {
            "query": "",
            "schema": None,
            "cypher_query": None,
            "query_results": None,
            "response": None,
            "error": None,
            "attempts": 0,
            "max_attempts": 3,
            "research_results": None,
            "messages": []
        }
    
    def _create_react_agent(self):
        """Create a ReAct agent with tools."""
        # Define tools
        tools = [
            self._neo4j_query_tool,
            self._neo4j_schema_tool,
            self._neo4j_research_tool
        ]
        
        # Create the agent using the prebuilt ReAct agent
        agent = create_react_agent(
            llm=self.llm_engine.client,
            tools=tools,
            system_prompt="""You are an expert Neo4j database assistant. You can help users query a Neo4j database using natural language.
            You have access to tools that can help you translate natural language to Cypher queries, execute those queries, and research the database.
            Always use the appropriate tool to help the user. If you're not sure about the database schema, use the neo4j_schema_tool first.
            If a query fails, try to research the database using the neo4j_research_tool to find relevant information.
            """
        )
        
        return agent
    
    async def _neo4j_query_tool(self, query: str) -> str:
        """Tool to translate natural language to Cypher and execute it."""
        try:
            # Get the schema
            schema = await self.neo4j_client.get_schema()
            
            # Translate to Cypher
            cypher_query = await self.llm_engine.translate_to_cypher(query, schema)
            
            # Execute the query
            if cypher_query.strip().upper().startswith(("CREATE", "DELETE", "MERGE", "REMOVE", "SET")):
                results = await self.neo4j_client.execute_write_query(cypher_query)
            else:
                results = await self.neo4j_client.execute_read_query(cypher_query)
            
            # Format the results
            return json.dumps({
                "cypher_query": cypher_query,
                "results": results
            }, indent=2)
        except Exception as e:
            return f"Error executing query: {str(e)}"
    
    async def _neo4j_schema_tool(self, query: str = "") -> str:
        """Tool to get the database schema."""
        try:
            # Get a simplified schema
            schema = await self.neo4j_client.get_simplified_schema()
            
            # Format the schema
            formatted_schema = []
            for node in schema:
                node_info = {
                    "label": node.get("label", "Unknown"),
                    "properties": node.get("properties", []),
                    "relationships": []
                }
                
                for rel in node.get("relationships", []):
                    node_info["relationships"].append({
                        "type": rel.get("type", "Unknown"),
                        "target": rel.get("target", "Unknown")
                    })
                
                formatted_schema.append(node_info)
            
            return json.dumps(formatted_schema, indent=2)
        except Exception as e:
            return f"Error getting schema: {str(e)}"
    
    async def _neo4j_research_tool(self, search_term: str) -> str:
        """Tool to research the database for relevant information."""
        try:
            # Try to find nodes with this term in any property
            query = f"""
            MATCH (n)
            WHERE any(prop in keys(n) WHERE toString(n[prop]) CONTAINS '{search_term}')
            RETURN DISTINCT labels(n) as labels, n
            LIMIT 10
            """
            
            results = await self.neo4j_client.execute_read_query(query)
            
            if results and len(results) > 0:
                return json.dumps({
                    "search_term": search_term,
                    "results": results
                }, indent=2)
            else:
                return f"No results found for search term: {search_term}"
        except Exception as e:
            return f"Error researching database: {str(e)}"
    
    async def get_schema(self, state: AgentState) -> AgentState:
        """Get the database schema."""
        try:
            # Check if the query is asking for a simplified schema
            if "simplified schema" in state["query"].lower() or "simple schema" in state["query"].lower():
                schema = await self.neo4j_client.get_simplified_schema()
                logger.info("Retrieved simplified schema")
            else:
                schema = await self.neo4j_client.get_schema()
                logger.info("Retrieved full schema")
            return {**state, "schema": schema}
        except Exception as e:
            logger.error(f"Error getting schema: {e}")
            return {**state, "error": str(e)}
    
    async def translate_to_cypher(self, state: AgentState) -> AgentState:
        """Translate natural language to Cypher."""
        try:
            # Check if the query is asking for a schema
            query_lower = state["query"].lower()
            
            # Intercept schema-related queries to avoid token limit issues
            if any(term in query_lower for term in ["schema", "visualization", "structure", "model", "diagram"]):
                logger.info("Intercepted schema query, using simplified schema query instead")
                # Use a more efficient query instead of db.schema.visualization()
                cypher_query = """
                MATCH (n)
                WITH DISTINCT labels(n)[0] AS label
                MATCH (n)
                WHERE label IN labels(n)
                WITH label, count(n) AS count
                ORDER BY count DESC
                LIMIT 10
                MATCH (n)
                WHERE label IN labels(n)
                WITH label, n LIMIT 1
                OPTIONAL MATCH (n)-[r]->(m)
                WITH label, collect(DISTINCT type(r)) AS relationshipTypes, 
                     collect(DISTINCT labels(m)[0]) AS connectedLabels,
                     keys(n) AS propertyKeys
                RETURN label, propertyKeys, 
                       [rel IN RANGE(0, size(relationshipTypes)-1) | 
                        {type: relationshipTypes[rel], target: connectedLabels[rel]}] AS relationships
                """
                return {**state, "cypher_query": cypher_query}
            # Check if the query is asking for a simplified schema
            elif "simplified schema" in query_lower or "simple schema" in query_lower:
                # Create a Cypher query that will show the schema in a readable format
                cypher_query = """
                MATCH (n)
                WITH DISTINCT labels(n)[0] AS label
                MATCH (n)
                WHERE label IN labels(n)
                WITH label, count(n) AS count
                ORDER BY count DESC
                LIMIT 10
                MATCH (n)
                WHERE label IN labels(n)
                WITH label, n LIMIT 1
                OPTIONAL MATCH (n)-[r]->(m)
                WITH label, collect(DISTINCT type(r)) AS relationshipTypes, 
                     collect(DISTINCT labels(m)[0]) AS connectedLabels,
                     keys(n) AS propertyKeys
                RETURN label, propertyKeys, 
                       [rel IN RANGE(0, size(relationshipTypes)-1) | 
                        {type: relationshipTypes[rel], target: connectedLabels[rel]}] AS relationships
                """
                return {**state, "cypher_query": cypher_query}
            else:
                cypher_query = await self.llm_engine.translate_to_cypher(
                    state["query"], state["schema"]
                )
                return {**state, "cypher_query": cypher_query}
        except Exception as e:
            logger.error(f"Error translating to Cypher: {e}")
            return {**state, "error": str(e)}
    
    async def execute_cypher_query(self, state: AgentState) -> AgentState:
        """Execute the Cypher query."""
        try:
            cypher_query = state["cypher_query"]
            
            # Determine if it's a read or write query
            if cypher_query.strip().upper().startswith(("CREATE", "DELETE", "MERGE", "REMOVE", "SET")):
                results = await self.neo4j_client.execute_write_query(cypher_query)
            else:
                results = await self.neo4j_client.execute_read_query(cypher_query)
            
            return {**state, "query_results": results}
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return {**state, "error": str(e), "attempts": state["attempts"] + 1}
    
    async def research_database(self, state: AgentState) -> AgentState:
        """Research the database to find relevant information when initial query fails."""
        try:
            # If we've already tried a few times, let's do some research
            logger.info(f"Researching database for query: {state['query']}")
            
            # Extract key entities or concepts from the query
            system_prompt = """You are an expert at extracting key entities and concepts from natural language queries.
            Given a user's query that failed to return results, identify the main entities, labels, or relationships 
            that might be present in a Neo4j database."""
            
            user_prompt = f"""
            User query: {state['query']}
            
            Extract 2-3 key entities, labels, or relationships that might be relevant for searching in a Neo4j database.
            Format your response as a comma-separated list of terms.
            """
            
            key_terms = await self.llm_engine.generate_text(system_prompt, user_prompt)
            
            # Use these terms to construct a more general query
            research_queries = []
            for term in key_terms.split(','):
                term = term.strip()
                if term:
                    # Try to find nodes with this term in any property
                    research_queries.append(f"""
                    MATCH (n)
                    WHERE any(prop in keys(n) WHERE toString(n[prop]) CONTAINS '{term}')
                    RETURN DISTINCT labels(n) as labels, n
                    LIMIT 5
                    """)
            
            # Execute the research queries
            all_results = []
            for query in research_queries:
                try:
                    results = await self.neo4j_client.execute_read_query(query)
                    if results and len(results) > 0:
                        all_results.extend(results)
                except Exception as e:
                    logger.warning(f"Error executing research query: {e}")
            
            # Summarize the research results
            if all_results:
                research_summary = f"Found {len(all_results)} potentially relevant nodes in the database."
                
                # Get a better Cypher query based on research
                system_prompt = """You are an expert in Neo4j and Cypher queries.
                Based on the research results and the original query, suggest a better Cypher query that might answer the user's question."""
                
                user_prompt = f"""
                Original query: {state['query']}
                Original Cypher query that failed: {state['cypher_query']}
                Research results: {json.dumps(all_results[:5], indent=2)}
                
                Suggest a better Cypher query that might answer the user's question.
                """
                
                suggested_query = await self.llm_engine.generate_text(system_prompt, user_prompt)
                
                return {
                    **state, 
                    "research_results": research_summary,
                    "cypher_query": suggested_query,
                    "attempts": state["attempts"] + 1
                }
            else:
                return {
                    **state, 
                    "research_results": "No relevant information found in the database.",
                    "attempts": state["attempts"] + 1
                }
        except Exception as e:
            logger.error(f"Error researching database: {e}")
            return {**state, "error": str(e), "attempts": state["attempts"] + 1}
    
    async def generate_response(self, state: AgentState) -> AgentState:
        """Generate a natural language response."""
        try:
            # If there's an error, use it as the response
            if state.get("error"):
                return {**state, "response": f"Error: {state['error']}"}
            
            query_lower = state["query"].lower()
            
            # Check if the query is asking for a schema (either simplified or regular)
            if any(term in query_lower for term in ["schema", "visualization", "structure", "model", "diagram", "simplified schema", "simple schema"]):
                # Format the schema results as a response
                results = state.get("query_results", [])
                
                if not results:
                    return {**state, "response": "No schema information was found in the database."}
                
                response = "### Database Schema\n\n"
                response += "This shows the main node types and their relationships:\n\n"
                
                # Add nodes
                response += "#### Node Types\n"
                for record in results:
                    label = record.get("label", "Unknown")
                    properties = record.get("propertyKeys", [])
                    
                    response += f"- **{label}**\n"
                    if properties:
                        prop_list = ", ".join(properties[:10])  # Limit to 10 properties
                        response += f"  - Properties: {prop_list}\n"
                
                # Add relationships
                response += "\n#### Relationships\n"
                for record in results:
                    label = record.get("label", "Unknown")
                    relationships = record.get("relationships", [])
                    
                    for rel in relationships:
                        if rel.get("type") and rel.get("target"):
                            response += f"- **{label}** --[{rel['type']}]--> **{rel['target']}**\n"
                
                return {**state, "response": response}
            
            # If there are no results, provide a simple response
            if not state.get("query_results") or len(state["query_results"]) == 0:
                if state.get("research_results"):
                    # If we did research but still no results, provide a more helpful response
                    system_prompt = """You are an expert at explaining database query results in natural language.
                    Given a user's query that didn't return direct results but had some research done, provide a helpful explanation."""
                    
                    user_prompt = f"""
                    User query: {state['query']}
                    
                    Cypher query executed: {state['cypher_query']}
                    
                    Research results: {state['research_results']}
                    
                    Please provide a helpful explanation of why we couldn't find exact results and what the user might try instead.
                    """
                    
                    response = await self.llm_engine.generate_text(system_prompt, user_prompt)
                    return {**state, "response": response}
                else:
                    return {**state, "response": "No results were found for your query."}
            
            # Generate a response based on the query and results
            system_prompt = """You are an expert at explaining database query results in natural language.
            Given a user's query and the results of a Neo4j Cypher query, provide a clear and concise explanation of the results.
            Focus on the key information and insights from the data."""
            
            user_prompt = f"""
            User query: {state['query']}
            
            Cypher query executed: {state['cypher_query']}
            
            Query results: {json.dumps(state['query_results'], indent=2)}
            
            Please provide a clear and concise explanation of these results.
            """
            
            response = await self.llm_engine.generate_text(system_prompt, user_prompt)
            return {**state, "response": response}
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {**state, "error": str(e), "response": f"Error: {str(e)}"}
    
    def should_retry(self, state: AgentState) -> Literal["retry", "respond"]:
        """Determine if we should retry the query or respond with what we have."""
        # If we have results, no need to retry
        if state.get("query_results") and len(state["query_results"]) > 0:
            return "respond"
        
        # If we've hit an error but haven't exceeded max attempts, retry
        if state.get("error") and state["attempts"] < state["max_attempts"]:
            logger.info(f"Retrying query (attempt {state['attempts']+1}/{state['max_attempts']})")
            return "retry"
        
        # If we have no results but haven't exceeded max attempts, retry
        if (not state.get("query_results") or len(state["query_results"]) == 0) and state["attempts"] < state["max_attempts"]:
            logger.info(f"No results found, retrying query (attempt {state['attempts']+1}/{state['max_attempts']})")
            return "retry"
        
        # Otherwise, just respond with what we have
        return "respond"
    
    async def use_react_agent(self, state: AgentState) -> AgentState:
        """Use the ReAct agent to process the query."""
        try:
            # Initialize messages if empty
            if not state.get("messages"):
                state["messages"] = [{"role": "user", "content": state["query"]}]
            
            # Invoke the ReAct agent
            result = await asyncio.to_thread(
                self.react_agent.invoke,
                {"messages": state["messages"]}
            )
            
            # Extract the response
            if result and "messages" in result:
                messages = result["messages"]
                # Get the last assistant message
                for message in reversed(messages):
                    if message["role"] == "assistant":
                        return {**state, "response": message["content"], "messages": messages}
            
            return {**state, "error": "No response from ReAct agent"}
        except Exception as e:
            logger.error(f"Error using ReAct agent: {e}")
            return {**state, "error": str(e)}
    
    def build_graph(self):
        """Build the LangGraph for the ReAct agent."""
        # Create the workflow
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("get_schema", self.get_schema)
        workflow.add_node("translate_to_cypher", self.translate_to_cypher)
        workflow.add_node("execute_cypher_query", self.execute_cypher_query)
        workflow.add_node("research_database", self.research_database)
        workflow.add_node("generate_response", self.generate_response)
        workflow.add_node("use_react_agent", self.use_react_agent)
        
        # Add edges
        workflow.add_edge("get_schema", "use_react_agent")
        
        # Set the entry point
        workflow.set_entry_point("get_schema")
        
        # Compile the workflow
        return workflow.compile()
    
    async def process_query(self, query: str) -> str:
        """Process a natural language query.
        
        Args:
            query: Natural language query
            
        Returns:
            Response to the query
        """
        try:
            # Initialize the state
            initial_state = self.get_empty_state()
            initial_state["query"] = query
            
            # Run the graph
            final_state = await self.graph.ainvoke(initial_state)
            
            # Return the response or error
            if final_state.get("response"):
                return final_state["response"]
            elif final_state.get("error"):
                return f"Error: {final_state['error']}"
            else:
                return "I couldn't process your query. Please try again."
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"Error: {str(e)}"
    
    async def get_debug_state(self, query: str) -> Dict[str, Any]:
        """Get the debug state for a query.
        
        Args:
            query: Natural language query
            
        Returns:
            Debug state with cypher query and results
        """
        try:
            # Initialize the state
            initial_state = self.get_empty_state()
            initial_state["query"] = query
            
            # Get the schema
            schema_state = await self.get_schema(initial_state)
            
            # Translate to Cypher
            cypher_state = await self.translate_to_cypher(schema_state)
            
            # Execute the query
            result_state = await self.execute_cypher_query(cypher_state)
            
            # Return the debug state
            return {
                "query": query,
                "schema": schema_state.get("schema"),
                "cypher_query": cypher_state.get("cypher_query"),
                "query_results": result_state.get("query_results"),
                "error": result_state.get("error")
            }
        except Exception as e:
            logger.error(f"Error getting debug state: {e}")
            return {
                "query": query,
                "error": str(e)
            } 
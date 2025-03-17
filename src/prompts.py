"""
This file contains public prompts used in the application.
Database-specific prompts and examples are stored in prompts_db.py
"""
import os
import sys

# Try to import database-specific prompts, use defaults if not available
try:
    from src.prompts_db import (
        CYPHER_TRANSLATION_PROMPT,
        RESULTS_EXPLANATION_PROMPT,
        SIMPLIFIED_SCHEMA_PROMPT,
        GENERIC_SYSTEM_PROMPT,
        GENERIC_EXAMPLES,
        CYPHER_EXAMPLES
    )
    print("Loaded database-specific prompts")
except ImportError:
    # Generic system prompt for text generation
    GENERIC_SYSTEM_PROMPT = """You are a helpful AI assistant with expertise in Neo4j databases and Cypher queries."""

    # System prompt for explaining Neo4j query results
    RESULTS_EXPLANATION_PROMPT = """You are an expert in explaining Neo4j query results in natural language."""

    # Prompt for simplified schema retrieval
    SIMPLIFIED_SCHEMA_PROMPT = """You are an expert in Neo4j databases. Your task is to retrieve a simplified schema that shows only the main nodes and their relationships.

    Return a Cypher query that will:
    1. Identify the main node labels in the database
    2. Show the relationships between these nodes
    3. Include only essential information (avoid retrieving all properties)
    4. Limit the results to prevent timeouts

    The query should be efficient and avoid retrieving excessive data. Focus on structure rather than content."""

    # Define generic examples
    GENERIC_EXAMPLES = [
        {
            "natural_language": "What are all the nodes in the database?",
            "cypher": "MATCH (n) RETURN n LIMIT 25"
        },
        {
            "natural_language": "Show me the database schema",
            "cypher": """MATCH (n)
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
        {type: relationshipTypes[rel], target: connectedLabels[rel]}] AS relationships"""
        },
        {
            "natural_language": "What node labels exist in the database?",
            "cypher": "CALL db.labels()"
        },
        {
            "natural_language": "What relationship types exist in the database?",
            "cypher": "CALL db.relationshipTypes()"
        },
        {
            "natural_language": "What properties do nodes have?",
            "cypher": "CALL db.propertyKeys()"
        }
    ]
    
    # No database-specific examples
    CYPHER_EXAMPLES = []
    
    print("Using default prompts (database-specific prompts not found)")
    
    # Build examples text
    examples_text = "\n\nHere are some examples of natural language queries and their corresponding Cypher translations:\n\n"
    
    # Add generic examples
    for i, example in enumerate(GENERIC_EXAMPLES, 1):
        examples_text += f"Example {i}:\nNatural Language: \"{example['natural_language']}\"\nCypher: {example['cypher']}\n\n"
    
    # System prompt for translating natural language to Cypher queries
    CYPHER_TRANSLATION_PROMPT = """You are an expert in translating natural language queries to Neo4j Cypher queries.
    Your task is to convert the user's question into a valid Cypher query that can be executed against a Neo4j database.
    Return ONLY the Cypher query without any explanations or markdown formatting.
    Be concise and focus on creating an efficient query.
    Don't Try to Match everything, the db is huge and you will get a timeout.
    
    IMPORTANT: ALWAYS include a LIMIT clause in every query to prevent timeouts and excessive data retrieval. If the user doesn't specify a limit, use a reasonable default like LIMIT 25.
    """ + examples_text + """
    Remember to use the schema information provided to create accurate queries. For schema visualization, always use db.schema.visualization() instead of db.schema() as the latter is not available in this Neo4j instance.
    
    IMPORTANT REMINDER: ALWAYS include a LIMIT clause in every query to prevent timeouts and excessive data retrieval.""" 
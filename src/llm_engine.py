import logging
from typing import Any, Dict, List, Optional, Union
import openai
import sys
import os
import json

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS
from src.prompts import CYPHER_TRANSLATION_PROMPT, RESULTS_EXPLANATION_PROMPT, GENERIC_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

class LLMEngine:
    """Engine for translating natural language to Cypher queries using LLMs."""
    
    def __init__(self, api_key: str = OPENAI_API_KEY, model: str = LLM_MODEL):
        """Initialize the LLM engine.
        
        Args:
            api_key: OpenAI API key
            model: LLM model to use
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.temperature = LLM_TEMPERATURE
        self.max_tokens = LLM_MAX_TOKENS
        
    async def translate_to_cypher(self, query: str, schema: List[Dict[str, Any]]) -> str:
        """Translate a natural language query to a Cypher query.
        
        Args:
            query: Natural language query
            schema: Database schema
            
        Returns:
            Cypher query
        """
        try:
            # Limit schema size to avoid context length issues
            if len(schema) > 10:
                logger.warning(f"Schema has {len(schema)} items, limiting to 10 to avoid context length issues")
                schema = schema[:10]
            
            # Create a more compact schema representation for the prompt
            compact_schema = []
            for item in schema:
                # Only include essential information
                compact_item = {
                    "label": item.get("label", "Unknown"),
                    "properties": item.get("properties", [])[:5],  # Limit to 5 properties
                }
                
                # Only include relationships if they exist and limit to 5
                relationships = item.get("relationships", [])
                if relationships and len(relationships) > 0:
                    compact_item["relationships"] = relationships[:5]
                
                compact_schema.append(compact_item)
            
            # Create a prompt for the LLM
            messages = [
                {"role": "system", "content": CYPHER_TRANSLATION_PROMPT},
                {"role": "user", "content": f"Schema: {json.dumps(compact_schema, indent=None)}\n\nQuery: {query}"}
            ]
            
            # Call the LLM (without await since OpenAI client is synchronous)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Extract the Cypher query from the response
            cypher_query = response.choices[0].message.content.strip()
            
            # Remove markdown code block formatting if present
            if cypher_query.startswith("```cypher"):
                cypher_query = cypher_query.replace("```cypher", "").replace("```", "").strip()
            elif cypher_query.startswith("```"):
                cypher_query = cypher_query.replace("```", "").strip()
            
            logger.info(f"Translated '{query}' to Cypher: {cypher_query}")
            return cypher_query
        except Exception as e:
            logger.error(f"Error translating query to Cypher: {e}")
            raise
    
    async def explain_results(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Explain the results of a Cypher query in natural language.
        
        Args:
            query: Original natural language query
            results: Results from the Cypher query
            
        Returns:
            Natural language explanation of the results
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                messages=[
                    {"role": "system", "content": RESULTS_EXPLANATION_PROMPT},
                    {"role": "user", "content": f"Original question: {query}\n\nQuery results: {results}\n\nPlease explain these results in a clear, concise way."}
                ]
            )
            
            explanation = response.choices[0].message.content.strip()
            return explanation
            
        except Exception as e:
            logger.error(f"Error explaining results: {e}")
            raise

    async def generate_text(self, system_prompt: str = GENERIC_SYSTEM_PROMPT, user_prompt: str = "") -> str:
        """Generate text using the LLM.
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            
        Returns:
            Generated text
        """
        try:
            # Create messages for the LLM
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Call the LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Extract the generated text
            generated_text = response.choices[0].message.content.strip()
            
            return generated_text
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise
            
    # Add methods to make the client compatible with the ReAct agent
    def bind_tools(self, tools):
        """Make the client compatible with the ReAct agent by adding a bind_tools method."""
        # This is a no-op since we're using the OpenAI client directly
        return self

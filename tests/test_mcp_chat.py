import pytest
import asyncio
from src.neo4j_client import Neo4jClient
from src.llm_engine import LLMEngine
from src.langgraph_agent import Neo4jReActAgent
import os

# Skip tests if API key is not set
requires_api_key = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY environment variable not set"
)

# Skip tests if Neo4j is not available
requires_neo4j = pytest.mark.skipif(
    not os.environ.get("NEO4J_URI"),
    reason="NEO4J_URI environment variable not set"
)

@requires_neo4j
class TestNeo4jClient:
    """Tests for the Neo4jClient class."""
    
    def test_init(self):
        """Test initialization of Neo4jClient."""
        client = Neo4jClient()
        assert client.uri is not None
        assert client.user is not None
        assert client.password is not None
    
    @pytest.mark.asyncio
    async def test_get_schema(self):
        """Test getting the database schema."""
        client = Neo4jClient()
        schema = await client.get_schema()
        assert isinstance(schema, list)
    
    @pytest.mark.asyncio
    async def test_execute_read_query(self):
        """Test executing a read query."""
        client = Neo4jClient()
        results = await client.execute_read_query("MATCH (n) RETURN count(n) as count LIMIT 1")
        assert isinstance(results, list)
        assert len(results) > 0
        assert "count" in results[0]

@requires_api_key
class TestLLMEngine:
    """Tests for the LLMEngine class."""
    
    def test_init(self):
        """Test initialization of LLMEngine."""
        engine = LLMEngine()
        assert engine.model is not None
    
    @pytest.mark.asyncio
    async def test_translate_to_cypher(self):
        """Test translating natural language to Cypher."""
        engine = LLMEngine()
        cypher = await engine.translate_to_cypher("Find all nodes labeled Person")
        assert isinstance(cypher, str)
        assert "MATCH" in cypher.upper()
    
    @pytest.mark.asyncio
    async def test_explain_results(self):
        """Test explaining results in natural language."""
        engine = LLMEngine()
        explanation = await engine.explain_results(
            "How many nodes are in the database?",
            [{"count": 42}]
        )
        assert isinstance(explanation, str)
        assert len(explanation) > 0

@requires_api_key
@requires_neo4j
class TestNeo4jReActAgent:
    """Tests for the Neo4jReActAgent class."""
    
    def test_init(self):
        """Test initialization of Neo4jReActAgent."""
        agent = Neo4jReActAgent()
        assert agent.neo4j_client is not None
        assert agent.llm_engine is not None
        assert agent.graph is not None
    
    @pytest.mark.asyncio
    async def test_run(self):
        """Test running the ReAct agent."""
        agent = Neo4jReActAgent()
        result = await agent.run("Count all nodes in the database")
        assert isinstance(result, dict)
        assert "final_answer" in result

import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# OpenAI API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Neo4j SSL Configuration
NEO4J_SSL_ENABLED = os.getenv("NEO4J_SSL_ENABLED", "True").lower() in ("true", "1", "t")
NEO4J_SSL_VERIFY = os.getenv("NEO4J_SSL_VERIFY", "False").lower() in ("true", "1", "t")
NEO4J_SSL_CERT_PATH = os.getenv("NEO4J_SSL_CERT_PATH", None)
NEO4J_SSL_KEY_PATH = os.getenv("NEO4J_SSL_KEY_PATH", None)
NEO4J_SSL_CA_CERT_PATH = os.getenv("NEO4J_SSL_CA_CERT_PATH", None)

# MCP Server Configuration
MCP_SERVER_HOST = os.getenv("MCP_SERVER_HOST", "localhost")
MCP_SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "8000"))

# Streamlit Configuration
STREAMLIT_TITLE = "Neo4j LLM Query Engine"
STREAMLIT_THEME = "light"

# LLM Configuration
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1000"))

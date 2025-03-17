# Neo4j LLM Query Engine

This project provides a natural language interface to Neo4j databases using the MCP (Model Context Protocol) server. It combines LLM-powered query translation with a ReAct agent built on LangGraph to create a powerful and intuitive way to interact with graph databases.

## Features

- 🔍 **Natural Language to Cypher Translation**: Translate natural language questions into Neo4j Cypher queries
- 🤖 **ReAct Agent Architecture**: Uses LangGraph to implement a reasoning and acting agent
- 🔄 **MCP Server Integration**: Leverages the Neo4j MCP server for database interactions
- 💬 **Streamlit Chat Interface**: User-friendly chat interface for querying the database
- 🧠 **Contextual Understanding**: Provides the database schema to the LLM for better query translation
- 🔁 **Retry Mechanism**: Automatically retries failed queries with research-based improvements
- 🔎 **Simplified Schema**: Provides a simplified database schema view to avoid token limits

## Project Structure

```
neo4j-llm-agent/
├── config.py              # Configuration file for API keys and server URLs
├── run.py                 # Main entry point to run the application
├── requirements.txt       # Python dependencies
├── .env.template          # Template for environment variables
├── .gitignore             # Git ignore file
├── README.md              # Project overview and setup instructions
├── src/                   # Source code directory
│   ├── __init__.py        # Makes src a package
│   ├── neo4j_client.py    # Module to interact with the Neo4j database
│   ├── llm_engine.py      # Module that integrates LLMs for query translation
│   ├── langgraph_agent.py # ReAct agent implementation using LangGraph
│   ├── prompts.py         # Public prompts used by the LLM
│   ├── prompts_db.py      # Database-specific prompts and examples (gitignored)
│   └── streamlit_app.py   # Streamlit-based chat interface
└── tests/                 # Directory for unit and integration tests
    └── test_neo4j_llm.py  # Tests for the project components
```

## Setup

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/neo4j-llm-agent.git
cd neo4j-llm-agent
```

2. **Create a virtual environment**

```bash
python -m venv .nenv
source .nenv/bin/activate  # On Windows: .nenv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure environment variables**

Copy the template environment file and edit it with your credentials:

```bash
cp .env.template .env
# Edit .env with your actual credentials
```

Required variables:
```
OPENAI_API_KEY=your-openai-api-key
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password
```

5. **Start the Neo4j database**

Make sure your Neo4j database is running and accessible with the credentials specified in the `.env` file.

6. **Run the application**

```bash
python run.py
```

This will:
- Start the MCP server for Neo4j
- Launch the Streamlit app

## Usage

1. Open the Streamlit app in your browser (typically at http://localhost:8501)
2. Type a natural language question about your Neo4j database
3. The app will translate your question to Cypher, execute it, and explain the results

Example queries:
- "Show me the database schema"
- "Get a simplified schema with only the main nodes and relationships"
- "Find all Promotion nodes with Approved status"
- "Count promotions by status"

For debugging, start your query with "debug:" to see the generated Cypher query and raw results.

## Database-Specific Prompts

The project supports separating database-specific prompts and examples:

1. The `src/prompts.py` file contains public prompts and will try to import from `src/prompts_db.py`
2. If `prompts_db.py` is not found, default prompts and generic examples will be used
3. The `prompts_db.py` file is gitignored to prevent committing sensitive data

To create your own database-specific prompts:
1. Copy the template from `src/prompts_db.py`
2. Customize the prompts and examples to match your database schema
3. Add domain-specific knowledge to the system prompts
4. Add relevant examples that demonstrate common query patterns for your database

The combined approach allows you to:
- Keep all database-specific content in one file
- Customize system prompts with domain knowledge
- Provide tailored examples for your specific database schema
- Maintain privacy by keeping sensitive data out of version control

## How It Works

1. **Query Translation**: The LLM engine translates natural language to Cypher queries
2. **Database Interaction**: The Neo4j client executes the queries against the database
3. **Result Explanation**: The LLM engine explains the results in natural language
4. **ReAct Agent**: The LangGraph-based agent orchestrates the entire process, including:
   - Retrieving the database schema
   - Translating queries to Cypher
   - Executing queries
   - Researching the database when queries fail
   - Generating helpful responses

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

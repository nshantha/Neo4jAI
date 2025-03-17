#!/usr/bin/env python
"""
Run script for the Neo4j LLM Query Engine.
This script starts the Streamlit app.
"""

import os
import sys
import subprocess
import time
import signal
import atexit
from dotenv import load_dotenv

# Global variable to store the MCP server process
mcp_server_process = None

def start_mcp_server():
    """Start the MCP server as a subprocess."""
    global mcp_server_process
    
    print("Starting MCP server...")
    
    # Get the current Python executable
    python_executable = sys.executable
    
    # Start the MCP server
    mcp_server_process = subprocess.Popen(
        [
            python_executable,
            "-m",
            "mcp_neo4j_cypher",
            "--db-url", os.environ.get("NEO4J_URI", "bolt://localhost:7687").replace("bolt+ssc://", "bolt://"),
            "--username", os.environ.get("NEO4J_USER", "neo4j"),
            "--password", os.environ.get("NEO4J_PASSWORD", "password")
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ.copy()
    )
    
    # Give the server a moment to start
    time.sleep(2)
    
    # Check if the server started successfully
    if mcp_server_process.poll() is not None:
        print("Error: Failed to start MCP server.")
        stdout, stderr = mcp_server_process.communicate()
        print(f"STDOUT: {stdout.decode('utf-8')}")
        print(f"STDERR: {stderr.decode('utf-8')}")
        sys.exit(1)
    
    print(f"MCP server started with PID: {mcp_server_process.pid}")
    
    # Register a function to stop the server when the script exits
    atexit.register(stop_mcp_server)

def stop_mcp_server():
    """Stop the MCP server."""
    global mcp_server_process
    
    if mcp_server_process is not None:
        print("Stopping MCP server...")
        
        # Try to terminate gracefully
        mcp_server_process.terminate()
        
        # Wait for a moment
        try:
            mcp_server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # If it doesn't terminate, kill it
            print("MCP server did not terminate gracefully, killing...")
            mcp_server_process.kill()
        
        print("MCP server stopped.")

def main():
    """Main entry point for the run script."""
    # Load environment variables
    load_dotenv()
    
    # Check if OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set it in your .env file or environment variables.")
        sys.exit(1)
    
    # Check if Neo4j URI is set
    if not os.environ.get("NEO4J_URI"):
        print("Warning: NEO4J_URI environment variable not set.")
        print("Using default: bolt://localhost:7687")
    
    # Start the MCP server
    start_mcp_server()
    
    # Set up signal handlers to gracefully shut down
    signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))
    signal.signal(signal.SIGTERM, lambda sig, frame: sys.exit(0))
    
    # Start the Streamlit app
    print("Starting Neo4j LLM Query Engine...")
    subprocess.run(["streamlit", "run", "src/streamlit_app.py"])

if __name__ == "__main__":
    main() 
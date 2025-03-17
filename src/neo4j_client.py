import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
import json
import subprocess
import sys
import os
import ssl

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    NEO4J_SSL_ENABLED, NEO4J_SSL_VERIFY, 
    NEO4J_SSL_CERT_PATH, NEO4J_SSL_KEY_PATH, NEO4J_SSL_CA_CERT_PATH
)

logger = logging.getLogger(__name__)

class Neo4jClient:
    """Client for interacting with Neo4j database through MCP server."""
    
    def __init__(self, uri: str = NEO4J_URI, user: str = NEO4J_USER, password: str = NEO4J_PASSWORD):
        """Initialize the Neo4j client.
        
        Args:
            uri: Neo4j database URI
            user: Neo4j username
            password: Neo4j password
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.mcp_process = None
        
    def start_mcp_server(self) -> None:
        """Start the MCP server as a subprocess."""
        try:
            # Start the MCP server as a subprocess
            cmd = [
                sys.executable, "-m", "mcp_neo4j_cypher",
                "--db-url", self.uri.replace("bolt+ssc://", "bolt://"),  # Remove SSL from URI for MCP server
                "--username", self.user,
                "--password", self.password
            ]
            
            # Add environment variables to disable SSL verification if needed
            env = os.environ.copy()
            if not NEO4J_SSL_VERIFY:
                env["PYTHONWARNINGS"] = "ignore:Unverified HTTPS request"
                env["PYTHONHTTPSVERIFY"] = "0"
                env["NEO4J_PYTHON_DRIVER_TRUST"] = "TRUST_ALL_CERTIFICATES"
            
            self.mcp_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env
            )
            
            logger.info(f"Started MCP server with PID {self.mcp_process.pid}")
        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            raise
    
    def stop_mcp_server(self) -> None:
        """Stop the MCP server subprocess."""
        if self.mcp_process:
            self.mcp_process.terminate()
            try:
                self.mcp_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.mcp_process.kill()
            logger.info("MCP server stopped")
            self.mcp_process = None
    
    def _get_ssl_context(self) -> Optional[ssl.SSLContext]:
        """Create an SSL context for Neo4j connection.
        
        Returns:
            SSL context or None if SSL is disabled
        """
        if not NEO4J_SSL_ENABLED:
            return None
        
        # Create SSL context
        ssl_context = ssl.create_default_context()
        
        # Configure SSL verification
        if not NEO4J_SSL_VERIFY:
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            logger.warning("SSL certificate verification is disabled. This is insecure!")
        
        # Load client certificate and key if provided
        if NEO4J_SSL_CERT_PATH and NEO4J_SSL_KEY_PATH:
            try:
                ssl_context.load_cert_chain(
                    certfile=NEO4J_SSL_CERT_PATH,
                    keyfile=NEO4J_SSL_KEY_PATH
                )
                logger.info("Loaded client certificate and key")
            except Exception as e:
                logger.error(f"Error loading client certificate and key: {e}")
        
        # Load CA certificate if provided
        if NEO4J_SSL_CA_CERT_PATH:
            try:
                ssl_context.load_verify_locations(cafile=NEO4J_SSL_CA_CERT_PATH)
                logger.info("Loaded CA certificate")
            except Exception as e:
                logger.error(f"Error loading CA certificate: {e}")
        
        return ssl_context
    
    def _create_driver(self):
        """Create a Neo4j driver with proper SSL configuration.
        
        Returns:
            Neo4j driver instance
        """
        from neo4j import GraphDatabase, TrustAll, TrustSystemCAs
        
        # For bolt+s:// or neo4j+s:// URIs, we need to handle SSL differently
        if self.uri.startswith(("bolt+s://", "neo4j+s://", "bolt+ssc://", "neo4j+ssc://")):
            # These URI schemes already include SSL settings, so we don't need to specify additional SSL parameters
            driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
        else:
            # For other URI schemes (bolt:// or neo4j://)
            ssl_config = {}
            if NEO4J_SSL_ENABLED:
                ssl_config["encrypted"] = True
                if not NEO4J_SSL_VERIFY:
                    ssl_config["trusted_certificates"] = TrustAll()
                else:
                    ssl_config["trusted_certificates"] = TrustSystemCAs()
            
            driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                **ssl_config
            )
        
        return driver
    
    async def execute_read_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a read query against the Neo4j database.
        
        Args:
            query: Cypher query
            params: Query parameters
            
        Returns:
            List of dictionaries representing the query results
        """
        try:
            with self._create_driver().session() as session:
                result = session.run(query, params or {})
                records = list(result)
                
                # Convert Neo4j objects to serializable dictionaries
                serializable_records = [
                    {key: self.neo4j_to_dict(record[key]) for key in record.keys()}
                    for record in records
                ]
                
                return serializable_records
        except Exception as e:
            logger.error(f"Error executing read query: {e}")
            raise
    
    async def execute_write_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a write query against the Neo4j database.
        
        Args:
            query: Cypher query
            params: Query parameters
            
        Returns:
            List of dictionaries representing the query results
        """
        try:
            with self._create_driver().session() as session:
                result = session.run(query, params or {})
                records = list(result)
                
                # Convert Neo4j objects to serializable dictionaries
                serializable_records = [
                    {key: self.neo4j_to_dict(record[key]) for key in record.keys()}
                    for record in records
                ]
                
                return serializable_records
        except Exception as e:
            logger.error(f"Error executing write query: {e}")
            raise
    
    def neo4j_to_dict(self, obj):
        """Convert Neo4j objects to serializable dictionaries.
        
        Args:
            obj: Neo4j object (Node, Relationship, Path, etc.)
            
        Returns:
            Dictionary representation of the object
        """
        from neo4j.graph import Node, Relationship, Path
        
        if isinstance(obj, Node):
            return {
                "id": obj.id,
                "labels": list(obj.labels),
                "properties": dict(obj)
            }
        elif isinstance(obj, Relationship):
            return {
                "id": obj.id,
                "type": obj.type,
                "properties": dict(obj),
                "start_node": self.neo4j_to_dict(obj.start_node),
                "end_node": self.neo4j_to_dict(obj.end_node)
            }
        elif isinstance(obj, Path):
            return {
                "nodes": [self.neo4j_to_dict(node) for node in obj.nodes],
                "relationships": [self.neo4j_to_dict(rel) for rel in obj.relationships]
            }
        elif isinstance(obj, list):
            return [self.neo4j_to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self.neo4j_to_dict(value) for key, value in obj.items()}
        else:
            return obj

    async def get_schema(self) -> List[Dict[str, Any]]:
        """Get the database schema.
        
        Returns:
            List of dictionaries representing the schema
        """
        try:
            # Use a more efficient query that returns a compact schema representation
            query = """
            MATCH (n)
            WITH DISTINCT labels(n) AS nodeLabels
            UNWIND nodeLabels AS label
            OPTIONAL MATCH (n)
            WHERE ANY(l IN labels(n) WHERE l = label)
            WITH label, n LIMIT 1
            OPTIONAL MATCH (n)-[r]->(m)
            WITH label, collect(DISTINCT type(r)) AS relationshipTypes, 
                 collect(DISTINCT labels(m)[0]) AS connectedLabels,
                 keys(n) AS propertyKeys
            RETURN {
                label: label,
                properties: propertyKeys,
                relationships: [rel IN RANGE(0, size(relationshipTypes)-1) | {
                    type: relationshipTypes[rel],
                    target: connectedLabels[rel]
                }]
            } AS schema
            LIMIT 20
            """
            
            result = await self.execute_read_query(query)
            if result and len(result) > 0:
                # Extract the schema from the result
                schema = []
                for record in result:
                    if 'schema' in record:
                        schema.append(record['schema'])
                
                logger.info(f"Got schema using custom query: {len(schema)} labels")
                return schema
        except Exception as e:
            logger.warning(f"Error getting schema with custom query: {e}")
            
            try:
                # Fallback to a very simple query that just gets node labels
                query = """
                MATCH (n)
                WITH DISTINCT labels(n) AS labels
                UNWIND labels AS label
                RETURN DISTINCT {
                    label: label,
                    properties: [],
                    relationships: []
                } AS schema
                LIMIT 20
                """
                result = await self.execute_read_query(query)
                if result and len(result) > 0:
                    # Extract the schema from the result
                    schema = []
                    for record in result:
                        if 'schema' in record:
                            schema.append(record['schema'])
                    
                    logger.info(f"Got schema using fallback query: {len(schema)} labels")
                    return schema
            except Exception as e2:
                logger.error(f"Error getting schema with fallback query: {e2}")
        
        # If all else fails, return a minimal mock schema
        logger.warning("Returning mock schema as fallback")
        return [{"label": "Node", "properties": ["id", "name"], "relationships": []}]

    async def get_simplified_schema(self) -> List[Dict[str, Any]]:
        """Get a simplified database schema showing only main nodes and relationships.
        
        Returns:
            List of dictionaries representing the simplified schema
        """
        try:
            # Query to get node counts to identify main nodes
            count_query = """
            MATCH (n)
            WITH DISTINCT labels(n)[0] AS label
            MATCH (n)
            WHERE label IN labels(n)
            WITH label, count(n) AS count
            ORDER BY count DESC
            LIMIT 10
            RETURN label, count
            """
            
            count_result = await self.execute_read_query(count_query)
            
            # Get the top labels (most frequent nodes)
            top_labels = [record['label'] for record in count_result]
            
            if not top_labels:
                logger.warning("No labels found in the database")
                return [{"label": "Node", "properties": ["id", "name"], "relationships": []}]
            
            # Query to get relationships between main nodes
            relationship_query = """
            MATCH (n)-[r]->(m)
            WHERE any(label IN labels(n) WHERE label IN $labels)
              AND any(label IN labels(m) WHERE label IN $labels)
            WITH DISTINCT labels(n)[0] AS sourceLabel, 
                         type(r) AS relType, 
                         labels(m)[0] AS targetLabel,
                         count(*) AS relCount
            ORDER BY relCount DESC
            LIMIT 20
            RETURN sourceLabel, relType, targetLabel, relCount
            """
            
            rel_result = await self.execute_read_query(relationship_query, {"labels": top_labels})
            
            # Build the simplified schema
            schema = []
            label_dict = {}
            
            # First add all top labels as nodes
            for label in top_labels:
                node_info = {
                    "label": label,
                    "properties": [],
                    "relationships": []
                }
                schema.append(node_info)
                label_dict[label] = node_info
            
            # Then add relationships
            for rel in rel_result:
                source_label = rel['sourceLabel']
                if source_label in label_dict:
                    label_dict[source_label]['relationships'].append({
                        "type": rel['relType'],
                        "target": rel['targetLabel']
                    })
            
            # Get a sample of properties for each label
            for node_info in schema:
                label = node_info['label']
                prop_query = f"""
                MATCH (n)
                WHERE $label IN labels(n)
                WITH n LIMIT 1
                RETURN keys(n) AS properties
                """
                try:
                    prop_result = await self.execute_read_query(prop_query, {"label": label})
                    if prop_result and len(prop_result) > 0 and 'properties' in prop_result[0]:
                        node_info['properties'] = prop_result[0]['properties'][:5]  # Limit to 5 properties
                except Exception as e:
                    logger.warning(f"Error getting properties for label {label}: {e}")
            
            logger.info(f"Got simplified schema: {len(schema)} labels, {len(rel_result)} relationships")
            return schema
            
        except Exception as e:
            logger.error(f"Error getting simplified schema: {e}")
            return [{"label": "Node", "properties": ["id", "name"], "relationships": []}]

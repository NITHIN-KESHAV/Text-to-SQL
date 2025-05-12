import os
import json
import re
from typing import Dict, List, Any, Tuple, Union

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client
from langsmith.run_helpers import trace

# Local imports
from .DatabaseManager import DatabaseManager
from .LLMManager import LLMManager
from .DataFormatter import DataFormatter
from .ResultFormatter import ResultFormatter
from .LangGraphAgent import LangGraphAgent

class SQLAgent:
    """
    SQL Agent for converting natural language queries to SQL and executing them.
    Uses LangGraph for workflow management.
    """
    
    def __init__(self, debug=False):
        """Initialize the SQL agent with required components."""
        # Initialize Database Manager for database operations
        self.db_manager = DatabaseManager()
        
        # Initialize LLM Manager for language model access
        hf_endpoint = os.environ.get("HF_ENDPOINT", "https://xk5cgx7gjgga5a7e.us-east-1.aws.endpoints.huggingface.cloud")
        hf_token = os.environ.get("HF_TOKEN")
        
        # Ensure we always use HuggingFace
        self.llm_manager = LLMManager(
            use_huggingface=True,  # Always use HuggingFace
            hf_endpoint=hf_endpoint,
            hf_token=hf_token,
            debug=debug
        )
        
        # Initialize data formatter for processing results
        self.data_formatter = DataFormatter()
        
        # Initialize result formatter for proper output formatting
        self.result_formatter = ResultFormatter()
        
        # Initialize LangGraph agent
        self.lang_graph_agent = LangGraphAgent(debug=debug)
        
        # Store debug flag
        self.debug = debug
        
        # Common term mappings for preprocessing
        self.term_mappings = {
            "movies": "films",
            "movie": "film",
            "genres": "categories",
            "genre": "category",
            "actors": "actor",
            "customers": "customer",
            "rentals": "rental",
            "payments": "payment"
        }

    def preprocess_query(self, question):
        """
        Preprocess the user's question to normalize terms and improve understanding.
        
        Args:
            question (str): The original user question
            
        Returns:
            str: The preprocessed question
        """
        processed_question = question
        
        # Convert to lowercase for easier matching
        lower_q = question.lower()
        
        # Replace common term variations with standard forms
        for term, replacement in self.term_mappings.items():
            # Only replace whole words, not parts of words
            processed_question = processed_question.replace(f" {term} ", f" {replacement} ")
            
            # Handle terms at the beginning of the question
            if lower_q.startswith(term + " "):
                processed_question = replacement + processed_question[len(term):]
                
            # Handle terms at the end of the question
            if lower_q.endswith(" " + term):
                processed_question = processed_question[:-len(term)] + replacement
        
        # Add preprocessing info if changes were made
        if processed_question != question and self.debug:
            print(f"[Preprocessing] Modified question: '{processed_question}'")
        
        return processed_question

    def nl_to_sql(self, nl_query: str) -> str:
        """
        Convert natural language query to SQL.
        This is a wrapper around the LangGraph workflow's SQL generation.
        
        Args:
            nl_query: Natural language question
            
        Returns:
            SQL query
        """
        # Preprocess the query to normalize terms
        processed_query = self.preprocess_query(nl_query)
        
        # Use the LangGraphAgent to process the query and extract just the SQL
        result = self.lang_graph_agent.process_query(processed_query)
        
        # Return the generated SQL
        return result.get("generated_sql", "")
    
    def process_nl_query(self, nl_query: str) -> Dict[str, Any]:
        """
        Process a natural language query using the LangGraph workflow.
        
        Args:
            nl_query: Natural language question
            
        Returns:
            Dict containing results, SQL, and other information
        """
        try:
            # Preprocess the query to normalize terms
            processed_query = self.preprocess_query(nl_query)
            
            # Process using the LangGraph workflow
            state = self.lang_graph_agent.process_query(processed_query)
            
            # Format the results for presentation
            results = state.get("results", [])
            formatted_results = self.result_formatter.format_results(
                sql=state.get("generated_sql", ""),
                results=results,
                visualization_type=state.get("visualization_type", "table")
            )
            
            # Return both raw state and formatted results
            return {
                **state,
                "formatted_results": formatted_results
            }
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            if self.debug:
                print(f"[PROCESS_NL_QUERY] {error_msg}")
            return {
                "user_query": nl_query,
                "error": error_msg
            }
    
    def execute_sql_query(self, sql_query: str) -> Tuple[List[Any], str]:
        """
        Execute an SQL query directly.
        
        Args:
            sql_query: SQL query to execute
            
        Returns:
            Tuple of (results, error message)
        """
        try:
            results = self.db_manager.execute_query(sql_query)
            return results, ""
        except Exception as e:
            error_msg = f"SQL execution failed: {str(e)}"
            if self.debug:
                print(f"[EXECUTE_SQL] {error_msg}")
            return [], error_msg
    
    def get_schema(self) -> str:
        """
        Get the database schema.
        
        Returns:
            Database schema as a string
        """
        return self.db_manager.get_schema()
    
    def close(self):
        """Close database connections and other resources."""
        self.db_manager.close_connection() 
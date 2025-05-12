import os
import sys
from typing import Dict, Any, List, Optional
import traceback

from .SQLAgent import SQLAgent
from .LangGraphAgent import LangGraphAgent
from langsmith import Client
from langsmith.run_helpers import trace
from dotenv import load_dotenv

class WorkflowManager:
    """
    WorkflowManager class to manage the workflow of SQL generation and execution.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the WorkflowManager with an SQLAgent.
        
        Args:
            verbose (bool): Whether to print verbose output
        """
        # Load environment variables
        load_dotenv()
        
        self.verbose = verbose
        self.sql_agent = SQLAgent(debug=verbose)
        self.database_connected = True
        
        # LangSmith configuration
        self.langsmith_api_key = os.environ.get("LANGCHAIN_API_KEY")
        self.langsmith_project = os.environ.get("LANGCHAIN_PROJECT", "sql_agent")
        self.langsmith_enabled = bool(self.langsmith_api_key)
        
        if self.langsmith_enabled and verbose:
            print(f"LangSmith tracing enabled for project: {self.langsmith_project}")
    
    def process_query(self, question: str) -> Dict[str, Any]:
        """
        Process a natural language query through the SQL Agent pipeline.
        
        Args:
            question (str): The natural language question to process
            
        Returns:
            Dict[str, Any]: The state object with query results, SQL, visualization info, etc.
        """
        # Just use regular processing for now instead of trying to use LangSmith tracing
        return self._regular_process_query(question)
    
    # Commenting out for now due to issues with trace decorator
    # @trace(name="SQL_Agent_Workflow")
    # def _traced_process_query(self, question: str) -> Dict[str, Any]:
    #     """
    #     LangSmith traced version of the query processing workflow.
    #     """
    #     return self._regular_process_query(question)
    
    def _regular_process_query(self, question: str) -> Dict[str, Any]:
        """
        Regular version of the query processing workflow.
        Processes a natural language query to SQL and executes it.
        
        Args:
            question (str): Natural language question
            
        Returns:
            Dict[str, Any]: Results and metadata
        """
        if self.verbose:
            print(f"Processing question: {question}")
        
        try:
            # Process the query using our SQLAgent's LangGraph workflow
            result = self.sql_agent.process_nl_query(question)
            
            if self.verbose and "error" in result:
                print(f"Error: {result['error']}")
            elif self.verbose:
                print(f"Successfully processed query")
                
            return result
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            if self.verbose:
                print(f"Error: {error_msg}")
                traceback.print_exc()
            
            return {
                "user_query": question,
                "error": error_msg
            } 
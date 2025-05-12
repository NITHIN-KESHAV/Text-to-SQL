"""
State definitions for the LangGraph workflow.
"""
from typing import TypedDict, Optional, List, Dict, Any, Union

class GraphState(TypedDict):
    """
    TypedDict to maintain state across nodes in the LangGraph workflow.
    This is a more explicit version of the AgentState type defined in LangGraphAgent.py.
    """
    # Core inputs and outputs
    user_query: str
    schema: Optional[str]
    reasoning: Optional[str]
    generated_sql: Optional[str]
    is_valid: Optional[bool]
    validation_message: Optional[str]
    results: Optional[List[Any]]
    
    # Metadata
    column_types: Optional[List[Any]]
    error: Optional[str]
    
    # Visualization info
    visualization_type: Optional[str]
    visualization_data: Optional[Dict[str, Any]]
    
    # Additional optional fields
    reasoning_time: Optional[float]
    generation_time: Optional[float]
    execution_time: Optional[float]
    processed_query: Optional[str]

    # New fields for enhanced reasoning
    clarification_needed: Optional[bool]  # Flag for ambiguous questions
    clarification_question: Optional[str]  # Question to ask for clarification
    entities: Optional[List[str]]  # Key entities identified in the question
    relationships: Optional[List[str]]  # Relationships between entities
    constraints: Optional[List[str]]  # Constraints or filters in the question
    suggested_followups: Optional[List[str]]  # Suggestions for follow-up questions 
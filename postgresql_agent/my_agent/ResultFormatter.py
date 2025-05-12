from langchain_core.prompts import ChatPromptTemplate
from my_agent.LLMManager import LLMManager
from typing import List, Dict, Any, Tuple, Optional, Union

class ResultFormatter:
    """
    Formats SQL query results for presentation, including visualization recommendations.
    """
    
    def __init__(self):
        """Initialize the ResultFormatter."""
        self.llm_manager = LLMManager()
    
    def format_results(self, sql: str, results: List[Any], visualization_type: str = "table") -> Dict[str, Any]:
        """
        Format SQL query results for presentation.
        
        Args:
            sql: The executed SQL query
            results: The query results (typically list of tuples)
            visualization_type: The recommended visualization type
            
        Returns:
            Dictionary with formatted results and metadata
        """
        # Handle empty results
        if not results:
            return {
                "sql": sql,
                "results": "No results found",
                "visualization": {
                    "type": "none",
                    "data": None
                }
            }
        
        # Format results based on type
        if isinstance(results[0], str) and results[0].startswith("❌"):
            # Handle error messages
            return {
                "sql": sql,
                "results": results[0],
                "visualization": {
                    "type": "none",
                    "data": None
                }
            }
        
        # Format data for visualization
        visualization_data = self._format_for_visualization(results, visualization_type)
        
        return {
            "sql": sql,
            "results": results,
            "result_count": len(results),
            "visualization": {
                "type": visualization_type,
                "data": visualization_data
            }
        }
    
    def _format_for_visualization(self, results: List[Any], vis_type: str) -> Dict[str, Any]:
        """
        Format data for the specified visualization type.
        
        Args:
            results: Query results
            vis_type: Type of visualization
            
        Returns:
            Formatted data for visualization
        """
        # Provide appropriate format based on visualization type
        if vis_type == "table":
            # Table visualization is just the raw data
            return {"rows": results}
        
        elif vis_type == "bar_chart":
            # Format for bar chart - assumes first column is label, second is value
            try:
                if len(results) == 0 or len(results[0]) < 2:
                    return {"rows": results}
                
                labels = [row[0] for row in results]
                values = [row[1] for row in results]
                
                return {
                    "labels": labels,
                    "values": values,
                    "title": "Bar Chart"
                }
            except Exception:
                # Fall back to raw data if formatting fails
                return {"rows": results}
        
        elif vis_type == "pie_chart":
            # Format for pie chart - assumes first column is label, second is value
            try:
                if len(results) == 0 or len(results[0]) < 2:
                    return {"rows": results}
                
                labels = [str(row[0]) for row in results]
                values = [float(row[1]) for row in results]
                
                return {
                    "labels": labels,
                    "values": values,
                    "title": "Pie Chart"
                }
            except Exception:
                # Fall back to raw data if formatting fails
                return {"rows": results}
        
        elif vis_type == "line_chart":
            # Format for line chart - assumes first column is x-axis (e.g., date), others are y-values
            try:
                if len(results) == 0 or len(results[0]) < 2:
                    return {"rows": results}
                
                x_values = [row[0] for row in results]
                series = []
                
                # If there are multiple y columns, create multiple series
                for i in range(1, len(results[0])):
                    series.append({
                        "name": f"Series {i}",
                        "values": [row[i] for row in results]
                    })
                
                return {
                    "x_values": x_values,
                    "series": series,
                    "title": "Line Chart"
                }
            except Exception:
                # Fall back to raw data if formatting fails
                return {"rows": results}
        
        else:
            # Default fallback
            return {"rows": results}

    def format_results_for_human(self, state: dict) -> dict:
        """Format query results into a human-readable response."""
        question = state['question']
        sql_query = state.get('sql_query', '')
        
        # Get reasoning and follow-up suggestions if available
        reasoning = state.get('reasoning', '')
        suggested_followups = state.get('suggested_followups', [])
        
        # Handle cases where results are missing
        if 'result' not in state or not state.get('result'):
            if state.get('error'):
                return {"formatted_results": f"I encountered an error while processing your query: {state['error']}. Please try rephrasing your question or providing more specific details."}
            return {"formatted_results": "I couldn't find any results for your question. The SQL query might be valid but returned no matching data."}
        
        results = state['result']
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", '''
You are an AI assistant that explains database query results in a natural, conversational way.
Analyze the results and provide a concise, helpful summary that answers the user's original question.

Guidelines:
1. Focus on the key insights from the data
2. Include relevant statistics or trends
3. Use clear, simple language
4. Keep your answer concise and direct
5. Do not include the SQL query in your answer
6. Do not use the word "query" in your response

Your response should read as if you're directly answering the user's question, not describing query results.
'''),
            ("human", '''
User question: {question}
SQL query: {sql_query}
Query results: {results}
My reasoning: {reasoning}

Create a concise, human-readable answer:
'''),
        ])

        response = self.llm_manager.invoke(
            prompt, 
            question=question, 
            sql_query=sql_query, 
            results=results,
            reasoning=reasoning
        )
        
        # Add follow-up suggestions if available
        formatted_response = response
        if suggested_followups:
            formatted_response += "\n\nYou might also want to know:"
            for i, followup in enumerate(suggested_followups[:3], 1):  # Limit to top 3
                formatted_response += f"\n{i}. {followup}"
        
        return {
            "formatted_results": formatted_response
        } 
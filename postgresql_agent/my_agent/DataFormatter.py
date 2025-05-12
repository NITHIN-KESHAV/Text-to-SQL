import json
from langchain_core.prompts import ChatPromptTemplate
from my_agent.LLMManager import LLMManager
from typing import List, Dict, Any, Tuple, Optional, Union

class DataFormatter:
    def __init__(self):
        self.llm_manager = LLMManager()

    def format_data_for_visualization(self, state: dict) -> dict:
        """Format the data for the chosen visualization type."""
        visualization = state['visualization']
        results = state['result']
        question = state['question']
        sql_query = state['sql_query']

        if visualization == "none" or not results:
            return {"formatted_data_for_visualization": None}
        
        try:
            # Check if we need to adjust the visualization type based on result structure
            visualization = self._adjust_visualization_type(visualization, results)
            
            # Format data based on visualization type
            if visualization == "bar" or visualization == "horizontal_bar":
                return self._format_bar_data(results, question)
            elif visualization == "scatter":
                return self._format_scatter_data(results)
            elif visualization == "line":
                return self._format_line_data(results, question)
            elif visualization == "pie":
                return self._format_pie_data(results, question)
            else:
                # Fallback to LLM formatting if the visualization type is not recognized
                return self._format_with_llm(visualization, question, sql_query, results)
        except Exception as e:
            print(f"Error formatting data: {str(e)}")
            # Provide a simple fallback for emergency situations
            return self._create_fallback_visualization(results, str(e))

    def _adjust_visualization_type(self, visualization, results):
        """
        Adjust the visualization type based on the structure of the results
        to ensure compatibility.
        """
        # If results is a string, try to convert to a list
        if isinstance(results, str):
            try:
                results = eval(results)
            except:
                # If conversion fails, leave as is
                pass
        
        # For empty results, no visualization is needed
        if not results or not isinstance(results, list) or len(results) == 0:
            return "none"
        
        # Handle single column results
        if len(results[0]) == 1:
            # For a single column of text values, bar chart is usually best
            if visualization in ["line", "scatter"]:
                print(f"Adjusting visualization from {visualization} to 'bar' for single column results")
                return "bar"
        
        # Handle two column results
        if len(results[0]) == 2:
            # If second column isn't numeric and line/scatter was chosen, switch to bar
            if visualization in ["line", "scatter"]:
                try:
                    # Check if second column can be converted to float
                    float(results[0][1])
                except (ValueError, TypeError):
                    print(f"Adjusting visualization from {visualization} to 'bar' because second column isn't numeric")
                    return "bar"
        
        return visualization
    
    def _create_fallback_visualization(self, results, error_message):
        """Create a simple fallback visualization when normal formatting fails"""
        try:
            # If results is a list of tuples/lists
            if isinstance(results, list) and len(results) > 0:
                if len(results[0]) == 1:
                    # Single column - return a simple bar chart
                    return {
                        "formatted_data_for_visualization": {
                            "labels": [str(i+1) for i in range(min(10, len(results)))],
                            "values": [{
                                "data": [1] * min(10, len(results)),
                                "label": "Count"
                            }],
                            "fallback": True,
                            "error": error_message
                        }
                    }
                elif len(results[0]) == 2:
                    # Two columns - try to make a basic bar chart
                    labels = [str(row[0]) for row in results[:10]]
                    try:
                        data = [float(row[1]) for row in results[:10]]
                    except (ValueError, TypeError):
                        data = [1 for _ in results[:10]]
                    
                    return {
                        "formatted_data_for_visualization": {
                            "labels": labels,
                            "values": [{
                                "data": data,
                                "label": "Value"
                            }],
                            "fallback": True,
                            "error": error_message
                        }
                    }
            
            # Default fallback
            return {
                "formatted_data_for_visualization": {
                    "error": error_message,
                    "fallback": True,
                    "message": "Unable to format data for visualization"
                }
            }
        except Exception as e:
            # Last resort fallback
            return {
                "formatted_data_for_visualization": {
                    "error": f"{error_message} (Fallback error: {str(e)})",
                    "fallback": True
                }
            }

    def _format_bar_data(self, results, question):
        """Format data for bar chart visualization."""
        if isinstance(results, str):
            results = eval(results)
        
        # Handle results with 2 columns (label, value)
        if len(results[0]) == 2:
            labels = [str(row[0]) for row in results]
            data = [float(row[1]) for row in results]
            
            # Use LLM to get a relevant label
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a data labeling expert. Given a question and some data, provide a concise and relevant label for the data series."),
                ("human", "Question: {question}\nData: {data}\n\nProvide a concise label for this data series (e.g., 'Sales', 'Revenue', 'Count'):")
            ])
            y_label = self.llm_manager.invoke(prompt, question=question, data=str(results[:3]))
            
            formatted_data = {
                "labels": labels,
                "values": [{"data": data, "label": y_label.strip()}]
            }
        # Handle results with 3 columns (category, series, value)
        elif len(results[0]) == 3:
            # Group by categories
            categories = list(set(row[0] for row in results))
            series = list(set(row[1] for row in results))
            
            formatted_data = {
                "labels": categories,
                "values": []
            }
            
            # Create series data
            for s in series:
                series_data = []
                for category in categories:
                    # Find the value for this category and series
                    value = next((float(row[2]) for row in results if row[0] == category and row[1] == s), 0)
                    series_data.append(value)
                
                formatted_data["values"].append({
                    "data": series_data,
                    "label": str(s)
                })
        else:
            raise ValueError(f"Unexpected data format for bar chart: {len(results[0])} columns")
            
        return {"formatted_data_for_visualization": formatted_data}

    def _format_line_data(self, results, question):
        """Format data for line chart visualization."""
        if isinstance(results, str):
            results = eval(results)
            
        # Handle results with 2 columns (x, y)
        if len(results[0]) == 2:
            x_values = [str(row[0]) for row in results]
            y_values = [float(row[1]) for row in results]
            
            # Use LLM to get a relevant label
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a data labeling expert. Given a question and some data, provide a concise and relevant label for the y-axis."),
                ("human", "Question: {question}\nData: {data}\n\nProvide a concise label for the y-axis (e.g., 'Sales', 'Revenue', 'Temperature'):")
            ])
            y_label = self.llm_manager.invoke(prompt, question=question, data=str(results[:3]))
            
            formatted_data = {
                "xValues": x_values,
                "yValues": [
                    {
                        "data": y_values,
                        "label": y_label.strip()
                    }
                ]
            }
        # Handle results with 3 columns (x, series, y)
        elif len(results[0]) == 3:
            # Extract unique x values and series
            x_values = sorted(list(set(str(row[0]) for row in results)))
            series = list(set(str(row[1]) for row in results))
            
            formatted_data = {
                "xValues": x_values,
                "yValues": []
            }
            
            # Create data for each series
            for s in series:
                series_data = []
                for x in x_values:
                    # Find the value for this x and series
                    value = next((float(row[2]) for row in results if str(row[0]) == x and str(row[1]) == s), None)
                    series_data.append(value)
                
                formatted_data["yValues"].append({
                    "data": series_data,
                    "label": s
                })
                
            # Use LLM to get a relevant label for the y-axis
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a data labeling expert. Given a question and some data, provide a concise and relevant label for the y-axis."),
                ("human", "Question: {question}\nData: {data}\n\nProvide a concise label for the y-axis:")
            ])
            y_axis_label = self.llm_manager.invoke(prompt, question=question, data=str(results[:3]))
            formatted_data["yAxisLabel"] = y_axis_label.strip()
        else:
            raise ValueError(f"Unexpected data format for line chart: {len(results[0])} columns")
            
        return {"formatted_data_for_visualization": formatted_data}

    def _format_scatter_data(self, results):
        """Format data for scatter plot visualization."""
        if isinstance(results, str):
            results = eval(results)
            
        formatted_data = {"series": []}
        
        # Handle results with 2 columns (x, y)
        if len(results[0]) == 2:
            data_points = [
                {"x": float(row[0]), "y": float(row[1]), "id": i+1}
                for i, row in enumerate(results)
            ]
            
            formatted_data["series"].append({
                "data": data_points,
                "label": "Data Points"
            })
        # Handle results with 3 columns (x, y, series)
        elif len(results[0]) == 3:
            # Group by series
            series_groups = {}
            
            for i, row in enumerate(results):
                # Determine which column is the series label
                # Typically it would be a string in one of the columns
                if isinstance(row[2], str) and not self._is_numeric(row[2]):
                    x, y, series = float(row[0]), float(row[1]), row[2]
                elif isinstance(row[1], str) and not self._is_numeric(row[1]):
                    x, series, y = float(row[0]), row[1], float(row[2])
                else:
                    series, x, y = row[0], float(row[1]), float(row[2])
                
                if series not in series_groups:
                    series_groups[series] = []
                
                series_groups[series].append({
                    "x": x,
                    "y": y,
                    "id": len(series_groups[series])+1
                })
            
            # Add each series to the formatted data
            for series, data in series_groups.items():
                formatted_data["series"].append({
                    "data": data,
                    "label": str(series)
                })
        else:
            raise ValueError(f"Unexpected data format for scatter plot: {len(results[0])} columns")
            
        return {"formatted_data_for_visualization": formatted_data}
        
    def _format_pie_data(self, results, question):
        """Format data for pie chart visualization."""
        if isinstance(results, str):
            results = eval(results)
        
        # Pie charts typically have 2 columns: label and value
        if len(results[0]) == 2:
            labels = [str(row[0]) for row in results]
            values = [float(row[1]) for row in results]
            
            formatted_data = {
                "labels": labels,
                "values": values
            }
            
            # Get a title for the chart
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a data visualization expert. Given a question and some data, provide a concise and relevant title for a pie chart."),
                ("human", "Question: {question}\nData: {data}\n\nProvide a concise title for this pie chart:")
            ])
            title = self.llm_manager.invoke(prompt, question=question, data=str(results[:3]))
            
            formatted_data["title"] = title.strip()
            
            return {"formatted_data_for_visualization": formatted_data}
        else:
            raise ValueError(f"Unexpected data format for pie chart: {len(results[0])} columns")
    
    def _format_with_llm(self, visualization, question, sql_query, results):
        """Use LLM to format data for visualization when standard formatting fails."""
        visualization_instructions = {
            "bar": """Format the data for a bar chart. The expected format is:
            {
                "labels": ["category1", "category2", ...],
                "values": [
                    {
                        "data": [value1, value2, ...],
                        "label": "Series Label"
                    }
                ]
            }""",
            
            "line": """Format the data for a line chart. The expected format is:
            {
                "xValues": ["x1", "x2", ...],
                "yValues": [
                    {
                        "data": [y1, y2, ...],
                        "label": "Series Label"
                    }
                ]
            }""",
            
            "scatter": """Format the data for a scatter plot. The expected format is:
            {
                "series": [
                    {
                        "data": [
                            {"x": x1, "y": y1, "id": 1},
                            {"x": x2, "y": y2, "id": 2},
                            ...
                        ],
                        "label": "Series Label"
                    }
                ]
            }""",
            
            "pie": """Format the data for a pie chart. The expected format is:
            {
                "labels": ["category1", "category2", ...],
                "values": [value1, value2, ...],
                "title": "Chart Title"
            }"""
        }
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a Data expert who formats data for visualizations. Return valid JSON data structured according to the provided format."),
            ("human", '''
Question: {question}
SQL query: {sql_query}
Results: {results}

Format the data for a {visualization} chart according to this structure:
{instructions}

Return ONLY valid JSON without any explanation, code blocks or markdown formatting.'''),
        ])
        
        instructions = visualization_instructions.get(visualization, "Format the data appropriately for visualization.")
        response = self.llm_manager.invoke(
            prompt, 
            question=question, 
            sql_query=sql_query, 
            results=results, 
            visualization=visualization,
            instructions=instructions
        )
        
        try:
            # Strip any extra text and extract just the JSON
            json_str = response.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            if json_str.startswith("```"):
                json_str = json_str[3:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]
                
            formatted_data = json.loads(json_str)
            return {"formatted_data_for_visualization": formatted_data}
        except json.JSONDecodeError as e:
            return {"error": f"Failed to parse LLM response as JSON: {str(e)}", "raw_response": response}
    
    def _is_numeric(self, value):
        """Check if a value is numeric (can be converted to float)."""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    def format_results_as_json(self, results: List[Tuple], column_names: List[str]) -> str:
        """
        Format query results as JSON.
        
        Args:
            results: List of result rows (tuples)
            column_names: Names of the columns
            
        Returns:
            JSON string representation of the results
        """
        if not results or not column_names:
            return json.dumps([])
        
        # Convert to list of dictionaries with column names as keys
        result_dicts = []
        for row in results:
            row_dict = {}
            for i, value in enumerate(row):
                if i < len(column_names):
                    # Handle special types that need conversion
                    if hasattr(value, 'isoformat'):  # For dates and timestamps
                        row_dict[column_names[i]] = value.isoformat()
                    else:
                        row_dict[column_names[i]] = value
            result_dicts.append(row_dict)
            
        return json.dumps(result_dicts, indent=2, default=str)
    
    def format_results_as_csv(self, results: List[Tuple], column_names: List[str]) -> str:
        """
        Format query results as CSV.
        
        Args:
            results: List of result rows (tuples)
            column_names: Names of the columns
            
        Returns:
            CSV string representation of the results
        """
        if not results or not column_names:
            return ""
        
        # Create CSV header
        csv_lines = [",".join([f'"{col}"' for col in column_names])]
        
        # Add data rows
        for row in results:
            # Handle special types and escape commas in values
            formatted_values = []
            for value in row:
                if value is None:
                    formatted_values.append('""')
                elif hasattr(value, 'isoformat'):  # For dates and timestamps
                    formatted_values.append(f'"{value.isoformat()}"')
                elif isinstance(value, str):
                    # Escape quotes in strings
                    escaped_value = value.replace('"', '""')
                    formatted_values.append(f'"{escaped_value}"')
                else:
                    formatted_values.append(f'"{value}"')
            
            csv_lines.append(",".join(formatted_values))
        
        return "\n".join(csv_lines)
    
    def truncate_results(self, results: List[Any], max_rows: int = 100) -> Tuple[List[Any], bool]:
        """
        Truncate results to a maximum number of rows.
        
        Args:
            results: List of result rows
            max_rows: Maximum number of rows to keep
            
        Returns:
            Tuple of (truncated_results, was_truncated)
        """
        if not results or len(results) <= max_rows:
            return results, False
        
        return results[:max_rows], True 
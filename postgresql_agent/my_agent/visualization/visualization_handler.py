def determine_visualization_type(data, query):
    """
    Automatically determine the best visualization type based on the data and query.
    """
    # Check if data is empty
    if not data or len(data) == 0:
        return 'table'  # Default to table for empty data
    
    # Get number of columns and rows
    row_count = len(data)
    
    # Check for single-column results (typically lists)
    if isinstance(data[0], tuple) and len(data[0]) == 1:
        print(f"Detected single column result with {row_count} rows")
        # If we have many rows, a list visualization or simple table works best
        if row_count > 20:
            return 'table'  # Table works well for longer lists
        else:
            return 'list'  # Custom list view for shorter lists
    
    # Extract column count from first row
    if isinstance(data[0], tuple) or isinstance(data[0], list):
        col_count = len(data[0])
    else:
        # Handle scalar values
        return 'text'
    
    # Keywords suggesting specific visualization types
    bar_keywords = ['count', 'total', 'sum', 'average', 'mean', 'compare', 'comparison', 'distribution', 'popular', 'top', 'bottom', 'most', 'least']
    line_keywords = ['time', 'trend', 'year', 'month', 'day', 'date', 'period', 'growth', 'change', 'increase', 'decrease']
    pie_keywords = ['percentage', 'proportion', 'share', 'ratio', 'distribution', 'breakdown']
    
    query_lower = query.lower()
    
    # Rule-based logic for visualization type
    if col_count == 2:
        # Two columns could be label-value pairs good for bar charts
        if any(kw in query_lower for kw in bar_keywords):
            return 'bar'
        # Could be time series data good for line charts
        elif any(kw in query_lower for kw in line_keywords):
            return 'line'
        # Small datasets with proportional data good for pie charts
        elif any(kw in query_lower for kw in pie_keywords) and row_count <= 10:
            return 'pie'
        else:
            return 'bar'  # Default to bar for 2 columns
            
    elif col_count >= 3:
        # Multi-column data is usually best as a table
        # But if two columns could form x-y data and others are grouping, try visualizing
        if any(kw in query_lower for kw in line_keywords):
            return 'line'
        else:
            return 'table'
    
    # Default to table for anything else
    return 'table'

def format_data_for_visualization(data, viz_type):
    """
    Format the data appropriately for the specified visualization type.
    """
    if not data or len(data) == 0:
        return {"error": "No data to visualize"}
    
    try:
        # Handle single column results (common case)
        is_single_column = False
        
        # Check if this is a single-column result
        if isinstance(data[0], tuple) and len(data[0]) == 1:
            is_single_column = True
            # Extract values from the tuples
            values = [item[0] for item in data]
            
            # Force simpler visualization types for single column
            if viz_type in ['bar', 'line', 'pie']:
                print(f"Converting single-column data to table visualization instead of {viz_type}")
                viz_type = 'table'
            
            if viz_type == 'list':
                # Special handling for list visualization
                return {
                    "items": values,
                    "type": "list"
                }
            elif viz_type == 'table':
                # For other types, convert to simple table format
                return {
                    "headers": ["Value"],
                    "rows": [[item[0]] for item in data]
                }
        
        # Regular multi-column data
        if isinstance(data[0], tuple) or isinstance(data[0], list):
            # For table visualization
            if viz_type == 'table':
                # Generate generic headers if needed
                headers = [f"Column {i+1}" for i in range(len(data[0]))]
                return {
                    "headers": headers,
                    "rows": data
                }
            
            # For chart visualizations
            elif viz_type in ['bar', 'line', 'pie']:
                # Basic chart data structure - first column as labels, second as values
                if len(data[0]) >= 2:
                    labels = [str(row[0]) for row in data]
                    datasets = [{
                        "label": f"Column {i+1}",
                        "data": [row[i] for row in data]
                    } for i in range(1, min(4, len(data[0])))]  # Limit to first 3 value columns
                    
                    return {
                        "labels": labels[:30],  # Limit to 30 categories
                        "datasets": datasets
                    }
                else:
                    # Fallback for single column in the results
                    if is_single_column:
                        return {
                            "headers": ["Value"],
                            "rows": [[item] for item in values]
                        }
                    else:
                        print(f"Unexpected data format for {viz_type} chart: {len(data[0])} columns")
                        return {
                            "headers": [f"Column {i+1}" for i in range(len(data[0]))],
                            "rows": data
                        }
            
        # Scalar or other data types
        return {
            "text": str(data)
        }
        
    except Exception as e:
        print(f"Error formatting data: {str(e)}")
        # Return the error and fall back to table
        try:
            if isinstance(data[0], (tuple, list)):
                cols = len(data[0])
                return {
                    "headers": [f"Column {i+1}" for i in range(cols)],
                    "rows": data
                }
            else:
                return {
                    "headers": ["Value"],
                    "rows": [[str(d)] for d in data]
                }
        except:
            # Last resort fallback
            return {
                "text": str(data)
            } 
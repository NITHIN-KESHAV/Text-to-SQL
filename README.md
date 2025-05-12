# Text to SQL with LangGraph

This project implements a natural language to SQL query system for a PostgreSQL database containing the Pagila/IMDB dataset. It uses LangGraph to define an agentic workflow that processes natural language questions, converts them to SQL, and returns meaningful results with visualizations.

## Features

- **Natural Language Understanding**: Ask questions in plain English
- **SQL Generation**: Translates questions to appropriate SQL queries
- **Schema Analysis**: Automatically analyzes database schema to understand tables and relationships
- **Pattern Recognition**: Identifies common query patterns for reliable SQL generation
- **Error Handling**: Robust validation and error correction
- **Visualization**: Automatically selects appropriate visualizations based on result data
- **IMDB-themed UI**: Dark mode interface with IMDB-style design
- **Query History**: Track past questions and reuse them with a single click

## Project Structure

- `app/`: Streamlit frontend for the SQL agent
- `postgresql_agent/`: Core agent implementation
  - `my_agent/`: LangGraph agent modules
    - `LangGraphAgent.py`: Main LangGraph workflow implementation
    - `DatabaseManager.py`: PostgreSQL connection and execution
    - `LLMManager.py`: LLM integration for SQL generation
    - `visualization/`: Result visualization utilities

## Setup and Installation

### Prerequisites

- Python 3.9+
- PostgreSQL database with Pagila/IMDB dataset
- Hugging Face API token (optional)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/datavisualization_langgraph.git
cd datavisualization_langgraph
```

2. Install required packages:
```bash
pip install -r postgresql_agent/requirements.txt
```

3. Configure environment variables in `postgresql_agent/.env`:
```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=pagila
DB_USER=your_user
DB_PASSWORD=your_password
HF_TOKEN=your_huggingface_token  # Optional
```

### Running the Application

Run the application with the provided script:
```bash
./run_app.sh
```

Alternatively, run directly with Streamlit:
```bash
streamlit run app/app.py
```

## Usage

1. Enter your natural language question in the text area
2. Click "Execute Query" to process your request
3. View the generated SQL and results
4. Explore the visualizations generated from your data
5. Access previous queries from the history section in the sidebar

## Example Queries

Try these example queries:

- "Show me 5 comedy films"
- "Which actor has appeared in the most films?"
- "Show me the customer who spent the most money on rentals"
- "What is the average length of horror films?"
- "Find films that have never been rented"
- "Show me rental counts by month in 2005"

## Architecture

The system uses a directed graph workflow with the following nodes:

1. **Schema Reasoning**: Extracts and analyzes the database schema
2. **SQL Generation**: Converts the natural language to SQL
3. **SQL Validation**: Validates the generated SQL for correctness
4. **SQL Execution**: Executes the query against the database
5. **Column Type Extraction**: Extracts column metadata
6. **Visualization Selection**: Determines appropriate visualization
7. **Error Handling**: Handles and recovers from errors

## Optimizations

- **Schema Extraction**: Extracts only relevant portions of the schema to reduce prompt size
- **Query Preprocessing**: Normalizes common terms (e.g., "movies" to "films") to improve accuracy
- **Fallback Mechanisms**: Includes rule-based fallbacks for common query types
- **Timeout Handling**: Robust timeout mechanisms to handle LLM latency
- **Error Recovery**: Sophisticated error handling with automatic correction attempts

## Troubleshooting

If you encounter any issues:

1. Check your database connection in the `.env` file
2. Ensure all required packages are installed
3. Check the console for any error messages
4. Try simpler queries first to verify functionality

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- LangGraph for workflow orchestration
- PostgreSQL for the database backend
- Streamlit for the user interface
- Pagila dataset for the sample data

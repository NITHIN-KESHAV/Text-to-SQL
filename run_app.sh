#!/bin/bash

# Activate the virtual environment if it exists
if [ -d "postgresql_agent/venv" ]; then
    echo "Activating virtual environment..."
    source postgresql_agent/venv/bin/activate
fi

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "Streamlit not found. Installing required packages..."
    pip install -r postgresql_agent/requirements.txt
fi

# Run the Streamlit app
echo "Starting SQL Query Assistant..."
streamlit run app/app.py 
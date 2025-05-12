#!/usr/bin/env python3
"""
Environment setup script for the SQL Agent

This script helps set up the environment variables needed for the SQL Agent.
"""

import os
import sys
import getpass
from pathlib import Path
from dotenv import load_dotenv

def save_env_variables(env_vars, filepath=".env"):
    """Save environment variables to a .env file."""
    with open(filepath, "w") as f:
        for key, value in env_vars.items():
            if value:  # Only write if value is not empty
                f.write(f"{key}={value}\n")
    
    print(f"Environment variables saved to {filepath}")

def setup_environment():
    """
    Set up environment variables for the application.
    """
    # Get the current directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path for .env file
    env_path = os.path.join(script_dir, '.env')
    
    # Check if .env file exists
    if os.path.exists(env_path):
        print(f"Loading existing environment from {env_path}")
        load_dotenv(env_path)
    
    # Dictionary to store environment variables
    env_vars = {}
    
    # Database configuration
    print("Configure database connection:")
    
    # Default values
    default_db_host = os.environ.get("DB_HOST", "localhost")
    default_db_port = os.environ.get("DB_PORT", "5432")
    default_db_name = os.environ.get("DB_NAME", "pagila")
    default_db_user = os.environ.get("DB_USER", "postgres")
    
    # Get user input for database configuration
    db_host = input(f"Database host [{default_db_host}]: ") or default_db_host
    db_port = input(f"Database port [{default_db_port}]: ") or default_db_port
    db_name = input(f"Database name [{default_db_name}]: ") or default_db_name
    db_user = input(f"Database user [{default_db_user}]: ") or default_db_user
    db_password = getpass.getpass(f"Database password: ") or os.environ.get("DB_PASSWORD", "")
    
    # Store database configuration
    env_vars["DB_HOST"] = db_host
    env_vars["DB_PORT"] = db_port
    env_vars["DB_NAME"] = db_name
    env_vars["DB_USER"] = db_user
    env_vars["DB_PASSWORD"] = db_password
    
    # Hugging Face configuration
    print("\nConfigure Hugging Face API (required):")
    
    # Default values
    default_hf_endpoint = os.environ.get("HF_ENDPOINT", "https://xk5cgx7gjgga5a7e.us-east-1.aws.endpoints.huggingface.cloud")
    default_hf_token = os.environ.get("HF_TOKEN", "")
    
    # Get user input for Hugging Face configuration
    hf_endpoint = input(f"Hugging Face endpoint URL [{default_hf_endpoint}]: ") or default_hf_endpoint
    
    if default_hf_token:
        use_existing = input(f"Use existing Hugging Face token? (y/n) [y]: ") or "y"
        if use_existing.lower() == "y":
            hf_token = default_hf_token
        else:
            hf_token = getpass.getpass("Hugging Face token: ")
    else:
        hf_token = getpass.getpass("Hugging Face token: ")
    
    # Store Hugging Face configuration
    env_vars["HF_ENDPOINT"] = hf_endpoint
    env_vars["HF_TOKEN"] = hf_token
    
    # LangSmith configuration
    print("\nConfigure LangSmith for tracing (optional):")
    
    # Default values
    default_langchain_api_key = os.environ.get("LANGCHAIN_API_KEY", "")
    default_langchain_project = os.environ.get("LANGCHAIN_PROJECT", "postgresql_agent")
    default_langchain_tracing_id = os.environ.get("LANGCHAIN_TRACING_ID", "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")
    
    # Get user input for LangSmith configuration
    use_langsmith = input("Do you want to use LangSmith for tracing? (y/n) [y]: ") or "y"
    
    if use_langsmith.lower() == "y":
        if default_langchain_api_key:
            use_existing = input(f"Use existing LangSmith API key? (y/n) [y]: ") or "y"
            if use_existing.lower() == "y":
                env_vars["LANGCHAIN_API_KEY"] = default_langchain_api_key
            else:
                env_vars["LANGCHAIN_API_KEY"] = getpass.getpass("LangSmith API key: ")
        else:
            env_vars["LANGCHAIN_API_KEY"] = getpass.getpass("LangSmith API key: ")
        
        env_vars["LANGCHAIN_PROJECT"] = input(f"LangSmith project name [{default_langchain_project}]: ") or default_langchain_project
        env_vars["LANGCHAIN_TRACING_ID"] = input(f"LangSmith tracing ID [{default_langchain_tracing_id}]: ") or default_langchain_tracing_id
        env_vars["LANGCHAIN_TRACING_V2"] = "true"
        env_vars["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    
    # Write environment variables to .env file
    with open(env_path, 'w') as f:
        for key, value in env_vars.items():
            f.write(f"{key}={value}\n")
    
    print(f"\nEnvironment variables saved to {env_path}")
    print("Setup complete!")

if __name__ == "__main__":
    setup_environment() 
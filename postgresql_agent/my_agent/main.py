import os
from dotenv import load_dotenv
from my_agent.WorkflowManager import WorkflowManager

# Load environment variables from .env file (if it exists)
load_dotenv()

# Set environment variables directly if they are not already set
if not os.environ.get("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_API_KEY"] = "your-langsmith-api-key-here"
    print("Set LANGSMITH_API_KEY from hardcoded value")

if not os.environ.get("HF_TOKEN"):
    os.environ["HF_TOKEN"] = "your-huggingface-token-here"
    print("Set HF_TOKEN from hardcoded value")

# Initialize the workflow manager
workflow_manager = WorkflowManager()

def process_question(question: str):
    """Process a question and return the result."""
    return workflow_manager.run_workflow(question)

def stream_processing(question: str):
    """Stream the processing of a question."""
    return workflow_manager.stream_workflow(question)

if __name__ == "__main__":
    # Example usage
    question = "What are the top 5 most rented films and how many times was each rented?"
    print(f"Processing question: {question}")
    result = process_question(question)
    
    print("\nFinal Result:")
    print("Question:", result["question"])
    print("Generated SQL:", result.get("sql_query", "Not generated"))
    print("Answer:", result.get("answer", "No answer generated"))
    print("Visualization Type:", result.get("visualization", "None"))
    print("Visualization Reason:", result.get("visualization_reason", ""))
    
    if result.get("error"):
        print("Error:", result["error"]) 
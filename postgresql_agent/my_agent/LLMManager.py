import os
import time
import signal
import threading
import traceback
from contextlib import contextmanager
from typing import List, Dict, Any, Optional, Union
import openai
from langchain_openai import ChatOpenAI
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from dotenv import load_dotenv
from langsmith import Client
from langsmith.run_helpers import trace

class LLMManager:
    def __init__(self, use_huggingface=True, hf_endpoint=None, hf_token=None, openai_api_key=None, debug=False):
        """
        Initialize the LLM manager with the appropriate client based on configuration.
        """
        # Load environment variables
        load_dotenv()
        
        self.debug = debug
        # Always use HuggingFace
        self.use_huggingface = True  
        self.hf_endpoint = hf_endpoint or os.environ.get("HF_ENDPOINT")
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        
        # LangSmith configuration
        self.langsmith_api_key = os.environ.get("LANGCHAIN_API_KEY")
        self.langsmith_project = os.environ.get("LANGCHAIN_PROJECT", "postgresql_agent")
        self.langsmith_enabled = bool(self.langsmith_api_key)
        
        if self.langsmith_enabled:
            # Initialize LangSmith client
            try:
                # Set up environment variables for LangSmith
                os.environ["LANGCHAIN_TRACING_V2"] = "true"
                os.environ["LANGCHAIN_PROJECT"] = self.langsmith_project
                os.environ["LANGCHAIN_API_KEY"] = self.langsmith_api_key
                
                # Initialize without explicit arguments to use environment variables
                self.langsmith_client = Client()
                print(f"LangSmith tracing enabled for project: {self.langsmith_project}")
            except Exception as e:
                print(f"Error initializing LangSmith client: {str(e)}")
                self.langsmith_enabled = False
        
        # Default timeout (in seconds)
        self.timeout = 30
        
        # Maximum prompt length (characters)
        self.max_prompt_length = 12000
        
        # Different models for different tasks
        self.sql_model = "default"  # For SQL generation
        self.reasoning_model = "default"  # For reasoning/analysis
        self.validation_model = "default"  # For validation
        
        # Initialize clients
        self.hf_client = None  # HuggingFace client
        
        # Initialize the HuggingFace client
        self._init_huggingface_client()
    
    def _init_huggingface_client(self):
        """
        Initialize the HuggingFace client with the provided token and endpoint.
        """
        if not self.hf_token:
            print("HF_TOKEN environment variable not set. Using demo mode.")
        
        if not self.hf_endpoint:
            print("HF_ENDPOINT environment variable not set. Using default endpoint.")
            self.hf_endpoint = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
        
        print(f"Initializing HuggingFace InferenceClient with endpoint: {self.hf_endpoint}")
        
        try:
            from huggingface_hub import InferenceClient
            self.hf_client = InferenceClient(model=self.hf_endpoint, token=self.hf_token)
            
            # Test the connection if in debug mode
            if self.debug:
                print("Testing connection to Hugging Face API...")
                self.hf_client.token_count("Hello world")
                print("Connection test successful.")
            
            print("InferenceClient initialized successfully")
        except Exception as e:
            print(f"Error initializing HuggingFace client: {str(e)}")
            # Still continue, as we might want to use fallbacks
    
    def _format_for_hf(self, formatted_messages):
        """
        Formats the messages for the HuggingFace API.
        """
        formatted_text = ""
        for message in formatted_messages:
            if message.type == "system":
                formatted_text += f"<s>[SYSTEM] {message.content} </s>\n"
            elif message.type == "human":
                formatted_text += f"<s>[HUMAN] {message.content} </s>\n"
            elif message.type == "ai":
                formatted_text += f"<s>[AI] {message.content} </s>\n"
            else:
                formatted_text += f"<s>{message.content}</s>\n"
        
        # Add the final assistant prompt
        formatted_text += "<s>[AI] "
        
        return formatted_text
    
    def invoke(self, prompt, model_name=None, reasoning=False, run_name="llm_invoke", **kwargs):
        """
        Invoke the LLM with the given prompt and return the response.
        """
        # Just use regular invoke instead of trying LangSmith tracing
        return self._regular_invoke(prompt, model_name, reasoning, **kwargs)
    
    # Commenting out for now due to issues with trace decorator
    # @trace
    # def _traced_invoke(self, prompt, model_name=None, reasoning=False, run_name="llm_invoke", **kwargs):
    #     """
    #     Traced version of invoke that logs to LangSmith
    #     """
    #     return self._regular_invoke(prompt, model_name, reasoning, **kwargs)
    
    def _regular_invoke(self, prompt, model_name=None, reasoning=False, **kwargs):
        """
        Regular invoke implementation
        """
        try:
            # Use a thread-based timeout approach instead of signals
            class TimeoutError(Exception):
                pass
                
            def timeout_handler(timeout_duration, func, *args, **kwargs):
                result = [None]
                error = [None]
                
                def target():
                    try:
                        result[0] = func(*args, **kwargs)
                    except Exception as e:
                        error[0] = e
                
                thread = threading.Thread(target=target)
                thread.daemon = True
                thread.start()
                thread.join(timeout_duration)
                
                if thread.is_alive():
                    raise TimeoutError(f"Request timed out after {timeout_duration} seconds")
                if error[0]:
                    raise error[0]
                return result[0]
            
            # Set the timeout depending on operation type
            max_duration = 25 if reasoning else 30  # reasoning needs shorter timeout
            
            # Select the right model based on the operation
            if not model_name:
                model_name = self.sql_model if not reasoning else self.reasoning_model

            # Format the prompt and get the messages
            formatted_prompt = prompt.format_messages(**kwargs)
            
            # Check the prompt length and truncate if necessary
            prompt_text = str(formatted_prompt)
            if len(prompt_text) > self.max_prompt_length:
                print(f"Warning: Prompt exceeds maximum length. Truncating to {self.max_prompt_length} characters.")
                # Truncate the prompt content in the user section
                for i, message in enumerate(formatted_prompt):
                    if message.type == "human":
                        content = message.content
                        if len(content) > 10000:  # Leave room for system prompt
                            formatted_prompt[i].content = content[:10000] + "...(truncated for length)"
            
            # Time the request
            start_time = time.time()
            
            # Define the function that makes the API call
            def make_api_call():
                # Check if HuggingFace client is initialized
                if not self.hf_client:
                    raise ValueError("HuggingFace client not initialized. Please check your configuration.")
                    
                # Convert the formatted prompt to a single string for HuggingFace
                prompt_str = self._format_for_hf(formatted_prompt)
                response = self.hf_client.text_generation(
                    prompt_str,
                    max_new_tokens=1000,
                    do_sample=True,
                    temperature=0.7,
                    repetition_penalty=1.2,
                    return_full_text=False
                )
                return response[0]["generated_text"] if isinstance(response, list) else response
            
            # Make the API call with timeout
            try:
                response_text = timeout_handler(max_duration, make_api_call)
            except TimeoutError as e:
                print(f"TimeoutError: {str(e)}")
                return f"Request timed out after {max_duration} seconds. Please try again with a more focused query."
            
            # Check if the request took too long
            elapsed_time = time.time() - start_time
            if elapsed_time > max_duration:
                print(f"Warning: Request took {elapsed_time:.2f} seconds, which exceeds the timeout of {max_duration} seconds")
            
            return response_text
            
        except Exception as e:
            print(f"Error in LLM invocation: {str(e)}")
            traceback.print_exc()
            return f"Error in generating response: {str(e)}. Please try a different query." 
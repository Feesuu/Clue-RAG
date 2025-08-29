from filelock import FileLock
import sqlite3
import functools
import hashlib
import json
from utils.logging import logger

def cache_response(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # get messages from args or kwargs
        if args:
            messages = args[0]
        else:
            messages = kwargs.get("messages")
        if messages is None:
            raise ValueError("Missing required 'messages' parameter for caching.")

        # get model and temperature from kwargs or self attributes
        model = kwargs.get("model", getattr(self, "model_name", "llama3.1:8b4k"))
        temperature = kwargs.get("temperature", getattr(self, "temperature", 0))
        max_tokens = kwargs.get("max_tokens", getattr(self, "max_tokens", 4000))
        seed = kwargs.get("seed", getattr(self, "seed", 0))

        # build key data, convert to JSON string and hash to generate key_hash
        key_data = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()

        # the file name of lock, ensure mutual exclusion when accessing concurrently
        lock_file = self.cache_file_name + ".lock"

        # Try to read from SQLite cache
        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            # if the table does not exist, create it
            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
            """)
            conn.commit()  # commit to save the table creation
            c.execute("SELECT message, metadata FROM cache WHERE key = ?", (key_hash,))
            row = c.fetchone()
            conn.close()
            if row is not None:
                message, metadata_str = row
                metadata = json.loads(metadata_str)
                # Log cache hit metadata
                logger.info(f"Cache hit - Metadata: {metadata}")
                return message, metadata

        # if cache miss, call the original function to get the result
        result = func(self, *args, **kwargs)
        message, metadata = result
        
        # Log new result metadata
        logger.info(f"Cache miss - New result metadata: {metadata}")
        # insert new result into cache
        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            # make sure the table exists again (if it doesn't exist, it would be created)
            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
            """)
            metadata_str = json.dumps(metadata)
            c.execute("INSERT OR REPLACE INTO cache (key, message, metadata) VALUES (?, ?, ?)",
                    (key_hash, message, metadata_str))
            conn.commit()
            conn.close()
            
        return message, metadata

    return wrapper

# from typing import Tuple, List, Optional, TypeVar, Type, Any
# from pydantic import BaseModel, ValidationError
# import json
# from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, RetryCallState
# from functools import wraps

# T = TypeVar('T', bound=BaseModel)

# def validate_and_retry(
#     max_retries: int = 3,
#     initial_wait: float = 1,
#     max_wait: float = 3,
#     backoff_factor: float = 2
# ):
#     """
#     Decorator to validate output against a Pydantic model and retry if validation fails.
#     """
#     def decorator(func):
#         @wraps(func)
#         @retry(
#             stop=stop_after_attempt(max_retries),
#             wait=wait_exponential(multiplier=backoff_factor, min=initial_wait, max=max_wait),
#             retry=retry_if_exception_type(ValidationError),
#             before_sleep=before_sleep_log
#         )
#         def wrapper(*args, **kwargs):
#             # Extract js_format from kwargs if present
#             js_format = kwargs.pop('js_format', None)
            
#             # Call the original function
#             response_content, metadata = func(*args, **kwargs)
            
#             # If an output model is specified, validate against it
#             if js_format is not None:
#                 try:
#                     # Try to parse the response
#                     parsed = js_format.model_validate_json(response_content)
#                     # If successful, return the parsed object
#                     return parsed, metadata
#                 except ValidationError as e:
#                     # Log the validation error
#                     logger.error(f"Validation error: {e}")
#                     raise
                    
#             return response_content, metadata
        
#         return wrapper
#     return decorator

# def before_sleep_log(retry_state: RetryCallState):
#     if retry_state.outcome is not None and retry_state.outcome.failed:
#         exc = retry_state.outcome.exception()
#         logger.warning(
#             f"Retrying {retry_state.fn.__name__} after attempt {retry_state.attempt_number} "
#             f"due to: {str(exc)}"
#         )
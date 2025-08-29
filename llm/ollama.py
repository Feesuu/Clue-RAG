import os
from typing import Tuple, List, Optional, Any
from ollama import Client
from llm.base import BaseLLM
from utils.config import BaseConfig
from utils.llm import TextChatMessage
from utils.logging import logger
from llm.cache import cache_response
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
from pydantic import BaseModel


class OllamaLLM(BaseLLM):
    def __init__(self, global_config: BaseConfig):
        super().__init__()
        self.llm_name = global_config.llm_name
        self.llm_base_url = global_config.llm_base_url
        self.temperature = global_config.temperature
        self.llm_max_tokens = global_config.llm_max_tokens
        self.llm_num_processes = global_config.llm_num_processes
        self.seed = global_config.seed
        
        # Initialize cache information
        self.cache_dir = os.path.join(
            os.path.join(
                global_config.save_dir, 
                f"{global_config.select_metric}_{global_config.alpha:.2f}"
            ), 
            "llm_cache"
        )
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file_name = os.path.join(self.cache_dir, f"{self.llm_name.replace('/', '_')}_cache.sqlite")

    def _create_client(self) -> Client:
        return Client(host=self.llm_base_url)

    def _format_messages(self, messages: List[TextChatMessage]) -> List[dict]:
        return [{'role': "system", 'content': "You are a helpful AI assistant."}] + [{'role': msg.role, 'content': msg.content} for msg in messages]

    def _prepare_options(self) -> dict:
        return {
            'num_predict': self.llm_max_tokens,
            'temperature': self.temperature,
            'seed': self.seed
        }
        
    def _create_showbar(self, messages_list):
        return tqdm(
            messages_list,
            desc="Sequential processing via ollama: ",
            unit="req",
            ncols=100
        )
    
    @cache_response
    def infer(
        self,
        messages: List[TextChatMessage],
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        max_retries: Optional[int] = 3,
        format: Optional[str] = None,
    ) -> Tuple[Any, dict]:
        """
        Enhanced inference with format validation and automatic retry.
        
        Args:
            messages: List of chat messages
            max_tokens: Maximum tokens to generate
            seed: Random seed for generation
            max_retries: Maximum number of retry attempts (default: 3)
            initial_delay: Initial delay between retries in seconds (default: 1.0)
            backoff_factor: Multiplier for increasing delay between retries (default: 2.0)
            
        Returns:
            Tuple of (response, metadata) where response is either:
            - The parsed js_format instance if validation succeeds
        """
        logger.info(f"Initiating Ollama request with {len(messages)} messages")
        
        max_tokens = max_tokens or self.llm_max_tokens
        seed = seed or self.seed
        client = self._create_client()
        
        metadata = {
            'model': self.llm_name,
            'max_tokens': max_tokens,
            'temperature': self.temperature,
            'seed': self.seed,
            "source": messages[0].source,
        }
        
        for attempt in range(max_retries+1):
            try:
                response = client.chat(
                    model=self.llm_name,
                    messages=self._format_messages(messages),
                    options=self._prepare_options(),
                    format=format,
                )
                
                content = response['message']['content']

                if "prompt_eval_count" in response:
                    pt = response.get('prompt_eval_count', 0)
                else:
                    pt = 0
                if "eval_count" in response:
                    ct = response.get('eval_count', 0)
                else:
                    ct = 0

                metadata.update({
                    'prompt_tokens': pt,
                    'completion_tokens': ct,
                    'total_tokens': pt + ct,
                })

                return content, metadata
                
            except Exception as e:
                logger.warning(f"Attempt {attempt+1}/{max_retries} failed: {str(e)}")
                
        logger.error(f"All {max_retries} retry attempts failed")
        metadata.update({
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
        })
        return "", metadata

    def multiprocess_batch_infer(
        self,
        messages_list: List[List[TextChatMessage]],
        max_tokens: Optional[int] = None,
        num_processes: Optional[int] = None,
        format: Optional[str] = None,
    ) -> Tuple[List[str], dict]:
        logger.info(f"Processing batch of {len(messages_list)} requests")
        
        max_tokens = max_tokens or self.llm_max_tokens
        num_processes = min(cpu_count(), num_processes or self.llm_num_processes)
        
        # Partial function 
        infer_partial = partial(self._infer_task, max_tokens=max_tokens, format=format)
        
        # Process batch with progress bar
        with Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.imap(infer_partial, messages_list),
                total=len(messages_list),
                desc="Processing batch requests via ollama",
                unit="req",
                ncols=100  # Adjust width for better visibility
            ))
        
        # Aggregate results and metadata
        all_responses = []
        combined_metadata = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
            'num_requests': len(messages_list),
            'model': self.llm_name,
            'max_tokens': max_tokens,
            'temperature': self.temperature,
            'num_processes': num_processes,
            'processing_mode': "multiprocessing",
        }
        
        for response_text, meta in results:
            all_responses.append((meta["source"], response_text))
            combined_metadata['prompt_tokens'] += meta['prompt_tokens']
            combined_metadata['completion_tokens'] += meta['completion_tokens']
            combined_metadata['total_tokens'] += meta['total_tokens']
        
        logger.info(
            f"Multiprocess batch completed."
        )
        return all_responses, combined_metadata
    
    def _infer_task(self, messages: List[TextChatMessage], max_tokens: int, format: str= None):
        """Helper method that can be pickled for multiprocessing"""
        return self.infer(messages=messages, max_tokens=max_tokens, format=format)
    
    
    def sequential_batch_infer(
        self,
        messages_list: List[List[TextChatMessage]],
        max_tokens: Optional[int] = None,
        format: Optional[str] = None,
    ) -> Tuple[List[str], dict]:
        """
            Process a batch of messages sequentially using the existing infer method
            
            Args:
                messages_list: List of message lists to process
                max_tokens: Maximum tokens to generate per request
                show_progress: Whether to display tqdm progress bar
                
            Returns:
                Tuple containing:
                - List of response texts in input order
                - Aggregated metadata dictionary
        """
        logger.info(f"Processing {len(messages_list)} requests sequentially")
        
        max_tokens = max_tokens or self.llm_max_tokens
        all_responses = []
        combined_metadata = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
            'num_requests': len(messages_list),
            'completed_requests': 0,
            'failed_requests': 0,
            'model': self.llm_name,
            'max_tokens': max_tokens,
            'temperature': self.temperature,
            'processing_mode': 'sequential'
        }

        # Create iterator with optional progress bar
        iterator = self._create_showbar(messages_list=messages_list)
        for messages in iterator:
            try:
                response_text, meta = self.infer(messages=messages, max_tokens=max_tokens)
                all_responses.append((meta["source"], response_text))
                combined_metadata['prompt_tokens'] += meta['prompt_tokens']
                combined_metadata['completion_tokens'] += meta['completion_tokens']
                combined_metadata['total_tokens'] += meta['total_tokens']
                combined_metadata['completed_requests'] += 1
            except Exception as e:
                logger.warning(f"Request failed: {str(e)}")
                combined_metadata['failed_requests'] += 1

        logger.info(
            f"Sequential batch completed: {combined_metadata.get('completed_requests', 0)} success, "
            f"{combined_metadata.get('failed_requests', 0)} failures, "
        )
        return all_responses, combined_metadata
    
if __name__ == "__main__":
    # print(test_ollama())
    config = BaseConfig()
    llm = OllamaLLM(config)
    
    from utils.prompt import EXTRACTION_PROMPT
    from utils.llm import KnowledgeUnits
    content = "Vaada Poda Nanbargal is a 2011 Indian Tamil-language romantic comedy film directed by Manikai. P. Arumaichandran has produced this movie under the banner 8 Point Entertainments. The film stars newcomers Nanda, Sharran Kumar and Yashika in the lead roles. The lead actor Nanda happens to be one of the strong contender of a popular television series \"Yaar Adutha Prabhu Deva\" aired on Vijay TV."
    
    single_message: List[TextChatMessage] = [
        TextChatMessage(role="user", content=EXTRACTION_PROMPT.format(passage=content), source="hi")
    ]
    response, metadata = llm.infer(single_message)
    print("Response:", response)
    print("Metadata:", metadata)
import os
from typing import Tuple, List, Optional, Any
from openai import OpenAI
from llm.base import BaseLLM
from utils.config import BaseConfig
from utils.llm import TextChatMessage
from utils.logging import logger
from llm.cache import cache_response
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
from pydantic import BaseModel


class OpenAIGPTLLM(BaseLLM):
    def __init__(self, global_config: BaseConfig):
        super().__init__()
        self.llm_name = global_config.llm_name
        self.api_key = global_config.api_key
        self.base_url = global_config.llm_base_url or "https://api.openai.com/v1"
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

    def _create_client(self) -> OpenAI:
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def _format_messages(self, messages: List[TextChatMessage]) -> List[dict]:
        return [{'role': "system", 'content': "You are a helpful AI assistant."}] + [{'role': msg.role, 'content': msg.content} for msg in messages]

    def _prepare_options(self, max_tokens: int, seed: Optional[int] = None) -> dict:
        options = {
            'max_tokens': max_tokens,
            'temperature': self.temperature,
        }
        if seed is not None:
            options['seed'] = seed
        return options
        
    def _create_showbar(self, messages_list):
        return tqdm(
            messages_list,
            desc="Sequential processing via OpenAI: ",
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
            
        Returns:
            Tuple of (response, metadata)
        """
        logger.info(f"Initiating OpenAI request with {len(messages)} messages")
        
        max_tokens = max_tokens or self.llm_max_tokens
        seed = seed or self.seed
        client = self._create_client()
        
        metadata = {
            'model': self.llm_name,
            'max_tokens': max_tokens,
            'temperature': self.temperature,
            'seed': self.seed,
            "source": messages[0].source if messages else "unknown",
        }
        
        for attempt in range(max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=self.llm_name,
                    messages=self._format_messages(messages),
                    **self._prepare_options(max_tokens, seed),
                    response_format={"type": "json_object"} if format == "json" else None,
                )
                
                content = response.choices[0].message.content
                usage = response.usage

                metadata.update({
                    'prompt_tokens': usage.prompt_tokens,
                    'completion_tokens': usage.completion_tokens,
                    'total_tokens': usage.total_tokens,
                })

                return content, metadata
                
            except Exception as e:
                logger.warning(f"Attempt {attempt+1}/{max_retries} failed: {str(e)}")
                if attempt == max_retries:
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
                desc="Processing batch requests via OpenAI",
                unit="req",
                ncols=100
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
            all_responses.append((meta.get("source", "unknown"), response_text))
            combined_metadata['prompt_tokens'] += meta['prompt_tokens']
            combined_metadata['completion_tokens'] += meta['completion_tokens']
            combined_metadata['total_tokens'] += meta['total_tokens']
        
        logger.info(
            f"Multiprocess batch completed."
        )
        return all_responses, combined_metadata
    
    def _infer_task(self, messages: List[TextChatMessage], max_tokens: int, format: str = None):
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
                response_text, meta = self.infer(messages=messages, max_tokens=max_tokens, format=format)
                all_responses.append((meta.get("source", "unknown"), response_text))
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
    config = BaseConfig()
    llm = OpenAIGPTLLM(config)
    
    from utils.prompt import EXTRACTION_PROMPT
    content = "Vaada Poda Nanbargal is a 2011 Indian Tamil-language romantic comedy film directed by Manikai. P. Arumaichandran has produced this movie under the banner 8 Point Entertainments. The film stars newcomers Nanda, Sharran Kumar and Yashika in the lead roles. The lead actor Nanda happens to be one of the strong contender of a popular television series \"Yaar Adutha Prabhu Deva\" aired on Vijay TV."
    
    single_message: List[TextChatMessage] = [
        TextChatMessage(role="user", content=EXTRACTION_PROMPT.format(passage=content), source="hi")
    ]
    response, metadata = llm.infer(single_message)
    print("Response:", response)
    print("Metadata:", metadata)
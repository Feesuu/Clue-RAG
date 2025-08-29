from utils.config import BaseConfig
from llm.ollama import OllamaLLM
from llm.openai import OpenAIGPTLLM
from typing import List, Dict, Any
from tqdm import tqdm
from utils.prompt import GENERATION_PROMPT
from utils.llm import TextChatMessage
from utils.logging import logger
from utils.utils import mdhash_id

class Generation:
    
    def __init__(self, global_config: BaseConfig):
        if global_config.use_ollama:
            self.llm = OllamaLLM(global_config)
        else:
            self.llm = OpenAIGPTLLM(global_config)
        self.multiprocess = global_config.multiprocess
        
        logger.info("Generation component initialized")
        logger.info(f"Using {'multiprocess' if self.multiprocess else 'sequential'} generation mode")
    
    def augmentated_generation(self, retrieval_results: List[Dict[str, Any]]):
        logger.info(f"Starting augmented generation for {len(retrieval_results)} retrieval results")
        
        generations = []
        
        for item in retrieval_results:
            question = item["question"]
            context = "\n\n".join(item["chunks"])
            
            prompt = GENERATION_PROMPT.format(context_data=context, question=question)
            generations.append([TextChatMessage(role="user", content=prompt, source=mdhash_id(prompt))])
        
        try:
            if self.multiprocess:
                results, res_metadata = self.llm.multiprocess_batch_infer(messages_list=generations)
            else:
                results, res_metadata = self.llm.sequential_batch_infer(messages_list=generations)
            
            self.res_metadata = res_metadata
            # self._log_generation_metrics(res_metadata)
            
            for idx, item in enumerate(retrieval_results):
                item["generation"] = results[idx]
                
            # breakpoint()
            
            logger.info("Completed all generations successfully")
            return retrieval_results
            
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            raise
    
    
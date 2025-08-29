import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import (
    Optional,
    Tuple,
    Any, 
    Dict,
    List
)

from utils.logging import logger
from utils.config import BaseConfig
from utils.llm import TextChatMessage

class BaseLLM(ABC):
    """Abstract base class for LLMs."""
    global_config: BaseConfig
    llm_name: str # Class name indicating which LLM model to use.
    
    def __init__(self, global_config: Optional[BaseConfig] = None) -> None:
        if global_config is None: 
            logger.debug("global config is not given. Using the default ExperimentConfig instance.")
            self.global_config = BaseConfig()
        else: self.global_config = global_config
        logger.debug(f"Loading {self.__class__.__name__} with global_config: {asdict(self.global_config)}")
        
        self.llm_name = self.global_config.llm_name
        logger.debug(f"Init {self.__class__.__name__}'s llm_name with: {self.llm_name}")
    
    def batch_upsert_llm_config(self, updates: Dict[str, Any]) -> None:
        """
        Upsert self.llm_config with attribute-value pairs specified by a dict. 
        
        Args:
            updates (Dict[str, Any]): a dict to be integrated into self.llm_config.
            
        Returns: 
            None
        """
        
        self.llm_config.batch_upsert(updates=updates)
        logger.debug(f"Updated {self.__class__.__name__}'s llm_config with {updates} to eventually obtain llm_config as: {self.llm_config}")
    
    
    def ainfer(self, chat: List[TextChatMessage]) -> Tuple[List[TextChatMessage], dict]:
        """
        Perform asynchronous inference using the LLM.
        
        Args:
            chat (List[TextChatMessage]): Input chat history for the LLM.

        Returns:
            Tuple[List[TextChatMessage], dict]: The list of n (number of choices) LLM response message (a single dict of role + content), and additional metadata (all input params including input chat) as a dictionary.
        """
        pass
    
 
    def infer(self, chat: List[TextChatMessage]) -> Tuple[List[TextChatMessage], dict]:
        """
        Perform synchronous inference using the LLM.
        
        Args:
            chat (List[TextChatMessage]): Input chat history for the LLM.

        Returns:
            Tuple[List[TextChatMessage], dict]: The list of n (number of choices) LLM response message (a single dict of role + content), and additional metadata (all input params including input chat) as a dictionary.
        """
        pass
    


    def batch_infer(self, batch_chat: List[List[TextChatMessage]]) -> Tuple[List[List[TextChatMessage]], List[dict]]:
        """
        Perform batched synchronous inference using the LLM.
        
        Args:
            batch_chat (List[List[TextChatMessage]]): Input chat history batch for the LLM.

        Returns:
            Tuple[List[List[TextChatMessage]], List[dict]]: The batch list of length-n (number of choices) list of LLM response message (a single dict of role + content), and corresponding batch of additional metadata (all input params including input chat) as a list of dictionaries.
        """
        
        pass
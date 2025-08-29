from utils.logging import logger
import json
import tiktoken
from utils.config import BaseConfig
from utils.utils import mdhash_id
from typing import List, Dict, Any

class Dataset:
    def __init__(self, global_config: BaseConfig):
        self.dataset_name = global_config.dataset_name
        self.encoder = tiktoken.get_encoding(global_config.token_encoder)
        self.max_text_token_size = global_config.max_text_token_size
        self.overlap_text_token_size = global_config.overlap_text_token_size
        
        self.load_corpora()
        self.load_questions()
        self._process_data = self.process_data(self._data)
        
        # Log final dataset statistics
        logger.info(f"Dataset initialization complete. Corpus items: {len(self._data)}, "
                   f"Processed chunks: {len(self._process_data)}, "
                   f"Questions: {len(self._questions_data)}")
        
    def load_corpora(self):
        """Load the JSON data from the specified dataset file."""
        try:
            with open(f"data/{self.dataset_name}_corpus.json", 'r', encoding='utf-8') as file:
                data = json.load(file)
                for item in data:
                    try:
                        item["content"] = f"Title: {item['title']}\nContent: {item['text']}"
                        #item["content"] = f"{item['text']}"
                    except:
                        item["content"] = f"Content: {item['text']}"
                    item["source_id"] = mdhash_id(item["content"])
                    
                self._data = data  
                logger.info(f"Loaded corpus: {len(data)} documents")
        except FileNotFoundError:
            logger.error(f"File '{self.dataset_name}_corpus.json' not found.")
            self._data = []
        except json.JSONDecodeError:
            logger.error(f"File '{self.dataset_name}_corpus.json' is not a valid JSON.")
            raw_data = []
            with open(f"data/{self.dataset_name}_corpus.json", 'r', encoding='utf-8') as file:
                for line in file:
                    raw_data.append(json.loads(line))
                    
            data = []
            for item in raw_data:
                try:
                    item["content"] = f"Title: {item['title']}\nContent: {item['context']}"
                except:
                    item["content"] = f"Content: {item['context']}"
                item["source_id"] = mdhash_id(item["content"])
                data.append(item)
    
            self._data = data
            
    def encode_text(self, text: str) -> List[int]:
        """Encode text to tokens."""
        return self.encoder.encode(text)
    
    def decode_tokens(self, tokens: List[int]) -> str:
        """Decode tokens back to text."""
        return self.encoder.decode(tokens)
    
    def process_data(self, data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Process JSON data into chunks with token size constraints.
        
        Args:
            data: List of dictionary items containing "text" field
            
        Returns:
            List of chunk dictionaries with metadata
        """
        chunks = {}
        total_chunks = 0
        
        for item in data:
            tokens = self.encode_text(item["content"])
            text_length = len(tokens)
            
            for start in range(0, text_length, self.max_text_token_size - self.overlap_text_token_size):
                end = start + self.max_text_token_size
                chunk_tokens = tokens[start:end]
                chunk_content = self.decode_tokens(chunk_tokens).strip()
                md5_id = mdhash_id(chunk_content)
                
                chunks[md5_id] = {
                    "source_id": item["source_id"],
                    "tokens": len(chunk_tokens),
                    "content": chunk_content
                }
                total_chunks += 1
                
        logger.info(f"Processed {len(data)} documents into {total_chunks} chunks")
        return chunks
    
    def load_questions(self):
        """Load the JSON data from the specified dataset file."""
        try:
            with open(f"data/{self.dataset_name}.json", 'r', encoding='utf-8') as file:
                data = json.load(file)
                questions_dict = {}
                for item in data:
                    try: id = item["id"] 
                    except: id = item["_id"]
                    questions_dict[id] = {
                        "question": item["question"],
                        "answer": item["answer"]
                    }
                    
                self._questions_data = questions_dict
                logger.info(f"Loaded {len(questions_dict)} questions")
        except FileNotFoundError:
            logger.error(f"File '{self.dataset_name}.json' not found.")
            self._questions_data = {}
        except json.JSONDecodeError:
            logger.error(f"File '{self.dataset_name}.json' is not a valid JSON.")
            self._questions_data = {}
    
    @property
    def corpora_data(self):
        """Return the loaded data."""
        return self._process_data
    
    @property
    def questions_data(self):
        """Return the loaded data."""
        return self._questions_data

if __name__ == "__main__":
    # Example usage
    config = BaseConfig()
    dataset = Dataset(config)
    
    print(dataset.corpora_data["08476336ad6f8e1f6142eb39ea05c73e"])
    print(dataset.questions_data["2hop__13548_13529"])
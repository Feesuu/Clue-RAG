import os
from dataclasses import dataclass, field
from typing import Literal, Union, Optional

@dataclass
class BaseConfig:
    """One and only configuration."""
    # LLM specific attributes 
    llm_name: str = field(
        default="llama3.1:8b4k", # qwen3:8b4k llama3.1:8b4k
        metadata={"help": "Class name indicating which LLM model to use."}
    )
    llm_base_url: str = field(
        default="http://localhost:5001/forward", #'http://localhost:5001/forward'
        metadata={"help": "Base URL for the LLM model"}
    )
    llm_max_tokens: int = field(
        default=4000,
        metadata={"help": "Max new tokens to generate in each inference."}
    )
    llm_num_processes: int = field(
        default=10,
        metadata={"help": "Number of parallel processes to use for batch inference operations in the LLM"}
    )
    seed: int = field(
        default=0,
        metadata={"help": "Random seed."}
    )
    temperature: float = field(
        default=0.0,
        metadata={"help": "Temperature for sampling in each inference."}
    )
    response_format: Union[dict, None] = field(
        default_factory=lambda: { "type": "json_object" },
        metadata={"help": "Specifying the format that the model must output."}
    )
    multiprocess: bool = field(
        default=True,
        metadata={"help": "Whether to use multiprocesses in the LLM or not"}
    )
    api_key: str = field(
        default="",
        metadata={"help": "API key to access your gpt."}
    )
    use_ollama: bool = field(
        default=True,
        metadata={"help": "Whether to use ollama or not"}
    )
    
    # Storage specific attributes 
    force_index_from_scratch: bool = field(
        default=False,
        metadata={"help": "If set to True, will ignore all existing storage files and graph data and will rebuild from scratch."}
    )
    
    db_name: str = field(
        default="milvus.db",
        metadata={"help": "Name to store the database."}
    )
    insert_batch_size: int = field(
        default=1024,
        metadata={"help": "Batch size to insert into the database."}
    )
    
    # Select chunks:
    alpha: float = field(
        default=1.0,
        metadata={"help": "Token constraint coefficient between 0 and 1, meaning the total percentage that can use in indexing stage."}
    )
    select_metric: Optional[Literal['COSINE', 'BLEU']] = field(
        default="COSINE",
        metadata={"help": "Metric used to measure the similarity between text chunks."}
    )
    
    num_processes: int = field(
        default=8,
        metadata={"help": "Number of parallel processes to use for batch processing"}
    )
    
    # Embedding specific attributes
    embedding_model_name: str = field(
        default="BAAI/bge-m3",
        metadata={"help": "Class name indicating which embedding model to use."}
    )
    embedding_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size of calling embedding model."}
    )
    embedding_return_as_normalized: bool = field(
        default=True,
        metadata={"help": "Whether to normalize encoded embeddings not."}
    )
    
    # Rerank
    rerank_model_name: str = field(
        default="BAAI/bge-reranker-v2-m3",
        metadata={"help": "Reranking model name"}
    )
    
    # Retrieval specific attributes
    top_K: int = field(
        default=3,
        metadata={"help": "The number of most relevent objects at retrieval step"}
    )
    top_M: int = field(
        default=5,
        metadata={"help": "Maintain top-M paths in each depth (Beam Size)"}
    )
    depth_D: int = field(
        default=3,
        metadata={"help": "The traverse depth of iterative retrival"}
    )
    top_N: int = field(
        default=5,
        metadata={"help": "Size of returned final chunks"}
    )
    use_entity_linking: bool = field(
        default=True,
        metadata={"help": "Whether use entity linking or not during retrieval stage"}
    )
    use_knowledge_anchoring: bool = field(
        default=True,
        metadata={"help": "Whether use knowledge anchoring or not during retrieval stage"}
    )
    use_dynamic: bool = field(
        default=True,
        metadata={"help": "Whether update query embedding or not during retrieval stage"}
    )
    
    # QA specific attributes
    max_qa_steps: int = field(
        default=1,
        metadata={"help": "For answering a single question, the max steps that we use to interleave retrieval and reasoning."}
    )
    qa_top_k: int = field(
        default=5,
        metadata={"help": "Feeding top k documents to the QA model for reading."}
    )
    
    # Save dir (highest level directory)
    save_dir: str = field(
        default=None,
        metadata={"help": "Directory to save all related information. If it's given, will overwrite all default save_dir setups. If it's not given, then if we're not running specific datasets, default to `outputs`, otherwise, default to a dataset-customized output dir."}
    )
    
    # Dataset
    dataset_name: Optional[Literal['hotpotqa', 'musique', '2wikimultihopqa',"test"]] = field(
        default="musique",
        metadata={"help": "Dataset to use. If specified, it means we will run specific datasets. If not specified, it means we're running freely."}
    )
    
    # Text encoder
    token_encoder: str = field(
        default="cl100k_base",
        metadata={"help": "Specifies the tokenizer encoding to use for text processing."}
    )
    max_text_token_size: int = field(
        default=1200,
        metadata={"help": "Maximum number of tokens allowed per text chunk."}
    )
    overlap_text_token_size: int = field(
        default=100,
        metadata={"help": "Number of overlapping tokens between consecutive chunks."}
    )
    
    def __post_init__(self):
        if self.save_dir is None: # If save_dir not given
            if self.dataset_name is None: self.save_dir = 'outputs' # running freely
            else: self.save_dir = os.path.join('outputs', self.dataset_name) # customize your dataset's output dir here

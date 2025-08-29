
from utils.config import BaseConfig
from dataset.dataclass import Dataset
from index.hybrid_extraction import HybridExtraction
from index.construction import MultiLayerGraph
from retrieval.retrieval import IterativeRetrieval
from generation.generation import Generation
from utils.utils import calculate_metric_scores, log_tokens
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    config = BaseConfig()
    dataset = Dataset(global_config=config)
    # offline indexing
    hybrid = HybridExtraction(global_config=config, dataset=dataset)
    graph = MultiLayerGraph(config, hybrid, dataset)
    
    # online retrieval
    retriever = IterativeRetrieval(global_config=config, graph=graph, dataset=dataset)
    retrieval_results = retriever.query()
    
    # generation
    generator= Generation(global_config=config)
    generation_results = generator.augmentated_generation(retrieval_results=retrieval_results)
    
    # evalutation
    results_overall, results_examples = calculate_metric_scores(config, generation_results)
    
    # Token statistics
    log_tokens(mode="HybridExtraction", res_metadata=hybrid.res_metadata)
    log_tokens(mode="Retrieval", res_metadata=retriever.res_metadata)
    log_tokens(mode="Generation", res_metadata=generator.res_metadata)
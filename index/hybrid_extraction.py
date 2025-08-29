from utils.config import BaseConfig
from dataset.dataclass import Dataset
from multiprocessing import Pool
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import os
from math import ceil
import torch
from index.embedding import EmbeddingModel
import pickle
from utils.logging import logger
from tqdm import tqdm
from llm.ollama import OllamaLLM
from llm.openai import OpenAIGPTLLM
from utils.llm import TextChatMessage
from pydantic import BaseModel
from sacrebleu import sentence_bleu
from utils.utils import clean_text

class HybridExtraction:
    
    def __init__(self, global_config: BaseConfig, dataset: Dataset):
        """Initialize hybrid extraction with caching capability."""
        self.select_metric = global_config.select_metric
        self.token_constraint = global_config.alpha
        
        self.dir = os.path.join(global_config.save_dir, f"{self.select_metric}_{self.token_constraint:.2f}")
        os.makedirs(self.dir, exist_ok=True)
        
        self.selection_cache_dir = os.path.join(self.dir, "selection_cache.pkl")
        self.extraction_cache_dir = os.path.join(self.dir, f"extraction_cache.pkl")
        
        logger.info(f"Initializing HybridExtraction with:")
        logger.info(f"Selection metric: {self.select_metric}")
        logger.info(f"Token constraint (alpha): {self.token_constraint:.2f}")
        logger.info(f"Cache location: {self.selection_cache_dir}")
        
        self.results = self._load_cache()
        
        if not self.results:
            logger.info("No valid cache found, computing from scratch...")
            self.dataset_name = global_config.dataset_name
            self.num_processes = global_config.num_processes
            self.embedder = EmbeddingModel(global_config)
            self.results = self.select_chunk(candidates=dataset.corpora_data)
        else:
            logger.info("Loaded results from cache")
            selected_items, max_value = self.results[-2], self.results[-1]
            logger.info(f"Selection completed. Selected {len(selected_items)} chunks with max value {max_value:.2f}")
            
        # LLM
        self.multiprocess = global_config.multiprocess
        if global_config.use_ollama:
            self.llm = OllamaLLM(global_config)
        else:
            self.llm = OpenAIGPTLLM(global_config)
    
        # Try to load extraction results from cache
        extraction_results = self._load_extraction_cache()
        
        if extraction_results:
            logger.info("Loaded extraction results from cache")
            self.knowledge_units, self.res_metadata, self.chunkid2ku2entity = extraction_results
        else:
            logger.info("No valid extraction cache found, computing from scratch...")
            # knowledge units extraction
            self.knowledge_units, self.res_metadata = self.knowledge_units_extraction(candidates=dataset.corpora_data)
            # entity extraction
            self.chunkid2ku2entity = self.ner_by_spacy(self.knowledge_units)
            # Save to cache
            self._save_extraction_cache((self.knowledge_units, self.res_metadata, self.chunkid2ku2entity))
        
    def _load_extraction_cache(self) -> Optional[Tuple]:
        """Attempt to load cached extraction results from disk.
        
        Returns:
            Tuple of (knowledge_units, res_metadata, chunkid2ku2entity) if cache exists and is valid,
            None otherwise.
        """
        if not os.path.exists(self.extraction_cache_dir):
            logger.debug("Extraction cache file does not exist")
            return None
        try:
            with open(self.extraction_cache_dir, 'rb') as f:
                cache_data = pickle.load(f)
                if isinstance(cache_data, tuple) and len(cache_data) == 3:
                    logger.debug("Successfully loaded valid extraction cache")
                    return cache_data
                else:
                    logger.warning("Invalid extraction cache format")
        except (pickle.PickleError, EOFError, FileNotFoundError) as e:
            logger.warning(f"Extraction cache loading failed: {str(e)}")
        return None
    
    def _save_extraction_cache(self, results: Tuple):
        """Save extraction results to disk cache.
        
        Args:
            results: Tuple of (knowledge_units, res_metadata, chunkid2ku2entity) to be cached
        """
        try:
            os.makedirs(os.path.dirname(self.extraction_cache_dir), exist_ok=True)
            with open(self.extraction_cache_dir, 'wb') as f:
                pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Extraction results successfully cached at {self.extraction_cache_dir}")
        except (IOError, pickle.PickleError) as e:
            logger.error(f"Failed to cache extraction results: {str(e)}")
        
    def _load_cache(self) -> Optional[Tuple]:
        """Attempt to load cached results from disk.
        
        Returns:
            Tuple of (scores, costs, selected_items, max_value) if cache exists and is valid,
            None otherwise.
        """
        if not os.path.exists(self.selection_cache_dir):
            logger.debug("Cache file does not exist")
            return None
            
        try:
            with open(self.selection_cache_dir, 'rb') as f:
                cache_data = pickle.load(f)
                if isinstance(cache_data, tuple) and len(cache_data) == 4:
                    logger.debug("Successfully loaded valid cache")
                    return cache_data
                else:
                    logger.warning("Invalid cache format")
        except (pickle.PickleError, EOFError, FileNotFoundError) as e:
            logger.warning(f"Cache loading failed: {str(e)}")
        return None
    
    def _save_cache(self, results: Tuple):
        """Save computation results to disk cache.
        
        Args:
            results: Tuple of (scores, costs, selected_items, max_value) to be cached
        """
        try:
            os.makedirs(os.path.dirname(self.selection_cache_dir), exist_ok=True)
            with open(self.selection_cache_dir, 'wb') as f:
                pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Results successfully cached at {self.selection_cache_dir}")
        except (IOError, pickle.PickleError) as e:
            logger.error(f"Failed to cache results: {str(e)}")
    
    @staticmethod
    def _calc_bleu_single(id, references, hypothesis):
        """Calculate BLEU score for a single hypothesis against multiple references.
        
        Args:
            id: Index of the current hypothesis
            references: List of reference texts
            hypothesis: Current hypothesis text
            
        Returns:
            Tuple of (id, bleu_score)
        """
        score = sentence_bleu(hypothesis, references, lowercase=True, smooth_method="floor")
        return id, score.score * 0.01
    
    @staticmethod
    def _calc_cosine_similarity(matrix):
        """Calculate mean cosine similarity for each row against all others.
        
        Args:
            matrix: Tensor of shape (n_samples, embedding_dim)
            
        Returns:
            Tensor of shape (n_samples,) containing mean cosine similarities
        """
        norm_matrix = matrix / torch.norm(matrix, dim=1, keepdim=True)
        cos_sim = torch.mm(norm_matrix, norm_matrix.T)
        cos_sim.fill_diagonal_(0)  # Exclude self-similarity
        cos_sim_mean = torch.sum(cos_sim, dim=1) / len(matrix)
        return cos_sim_mean
    
    @staticmethod
    def _knapsack_problem(number: int, weight: int, w: np.ndarray, v: np.ndarray) -> Tuple:
        """Solve 0/1 knapsack problem using dynamic programming with memory-mapped arrays.
        
        Args:
            number: Number of items
            weight: Maximum weight capacity
            w: Array of item weights (shape: [number])
            v: Array of item values (shape: [number])
            
        Returns:
            tuple: (max_value, selected_items) where:
                - max_value: Maximum achievable value
                - selected_items: List of selected item indices (0-based)
        """
        temp_file = "temp_knapsack.npy"
        
        # Clean up previous temp file if exists
        if os.path.exists(temp_file):
            os.remove(temp_file)

        # Initialize memory-mapped array for DP table
        dp_table = np.memmap(temp_file, dtype=np.float32, mode="w+", shape=(number+1, weight+1))
        dp_table[0, :] = 0  # Base case: 0 items have 0 value

        # Dynamic programming with vectorized operations
        for i in tqdm(range(1, number+1), total=number, desc="DP computation"):
            prev_row = dp_table[i-1, :]
            current_row = prev_row.copy()  # Default: don't take current item

            # Vectorized update for positions where current item can fit
            j_values = np.arange(w[i-1], weight+1)
            if len(j_values) > 0:
                # Compute potential new values if taking current item
                new_values = prev_row[j_values - w[i-1]] + v[i-1]
                # Take maximum between taking and not taking the item
                current_row[j_values] = np.maximum(prev_row[j_values], new_values)

            dp_table[i, :] = current_row
            dp_table.flush()  # Ensure writes to disk

        # Backtrack to find selected items
        selected_items = []
        remaining_weight = weight
        for i in tqdm(range(number, 0, -1), total=number, desc="Backtracking"):
            if dp_table[i, remaining_weight] != dp_table[i-1, remaining_weight]:
                selected_items.append(i-1)  # Item was selected
                remaining_weight -= w[i-1]

        max_value = dp_table[number, weight]
        
        # Clean up resources
        del dp_table
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
        return float(max_value), selected_items

    def _calculate_selfbleu(self, candidates_text: list[str]) -> list[float]:
        """Calculate BLEU scores for a list of candidate texts.
        
        Args:
            candidates_text: List of text documents to evaluate
            
        Returns:
            List of BLEU scores for each candidate text
        """
        logger.debug("Calculating Self-BLEU scores...")
        sentence_num = len(candidates_text)
        with Pool(self.num_processes) as pool:
            # Process in parallel with progress bar
            results = [
                pool.apply_async(
                    self._calc_bleu_single, 
                    args=(cid, candidates_text[:cid] + candidates_text[cid+1:], candidates_text[cid])
                )
                for cid in range(sentence_num)
            ]
            
            # Collect results with progress bar
            sorted_result = [0.0] * sentence_num
            for res in tqdm(results, desc="Calculating BLEU: "):
                cid, value = res.get()
                sorted_result[cid] = value
        
        return sorted_result

    def select_chunk(self, candidates: Dict[str, Dict[str, Any]]) -> Tuple:
        """Select optimal chunks based on specified metric and token constraints.
        
        Args:
            candidates: Dictionary of candidate chunks with their metadata
            
        Returns:
            Tuple of:
            - scores: Array of metric scores for each candidate
            - costs: Array of token counts for each candidate
            - selected_items: List of indices of selected candidates
            - max_value: Maximum achieved value from knapsack solution
        """
        logger.info(f"Starting chunk selection with {len(candidates)} candidates")
        
        texts = [value["content"] for _, value in sorted(candidates.items())]
        
        if self.token_constraint == 1:
            scores, costs, selected_items, max_value = [], [], list(range(len(texts))), 0
        elif self.token_constraint == 0:
            scores, costs, selected_items, max_value = [], [], [], 0
        else:
            # Calculate scores based on selected metric
            if self.select_metric == "BLEU":
                logger.debug("Using BLEU metric for selection")
                scores = np.array(self._calculate_selfbleu(candidates_text=texts))
            elif self.select_metric == "COSINE":
                logger.debug("Using Cosine Similarity metric for selection")
                matrix = self.embedder.get_embedding(texts=texts)
                matrix = torch.from_numpy(matrix)
                scores = self._calc_cosine_similarity(matrix)
                scores = scores.numpy()
            else:
                raise ValueError(f"Unsupported selection metric: {self.select_metric}")
            
            costs = np.array([value["tokens"] for key, value in sorted(candidates.items())])
        
            # Solve knapsack problem
            logger.debug("Solving knapsack problem...")
            max_value, selected_items = self._knapsack_problem(
                number=len(texts),
                weight=ceil(sum(costs) * self.token_constraint),
                w=costs,
                v=scores
            )
        
        results = (scores, costs, selected_items, max_value)
        self._save_cache(results)
        
        logger.info(f"Selection completed. Selected {len(selected_items)} chunks with max value {max_value:.2f}")
        return results
    
    def extract_by_llm(self, texts: List[Tuple[str, str]]):
        from utils.prompt import EXTRACTION_PROMPT
        import json
        
        if self.multiprocess:
            results, res_metadata = self.llm.multiprocess_batch_infer(
                messages_list=list(map
                    (
                        lambda x: [TextChatMessage(role="user", content=EXTRACTION_PROMPT.format(passage=x[1]), source=x[0])], 
                        texts
                    )
                ),
                format="json"
            )
        else:
            results, res_metadata = self.llm.sequential_batch_infer(
                messages_list=list(map
                    (
                        lambda x: [TextChatMessage(role="user", content=EXTRACTION_PROMPT.format(passage=x[1]), source=x[0])], 
                        texts
                    )
                ),
                format="json"
            )
        
        new_results = []
        for rid, result in results:
            try:
                result = json.loads(result)
                new_results.append({"chunk_id": rid, "knowledge_units": result["knowledge_units"]})
            except:
                logger.info(f"Fail to parse {rid} source...")
            
        #results = list(map(lambda x: {"chunk_id": x[0], "knowledge_units": x[1].knowledge_units}, results))
        return new_results, res_metadata
    
    def extract_by_nlp_tool(self, texts: List[Tuple[str, str]]):
        from nltk.tokenize import sent_tokenize
        results = []
        for id, text in texts:
            sentences = sent_tokenize(text)
            sentences = [sen for sen in sentences if sen != ""]
            results.append({"chunk_id": id, "knowledge_units": sentences})
        
        return results
    
    def knowledge_units_extraction(self, candidates: Dict[str, Dict[str, Any]]):
        from utils.llm import KnowledgeUnits
        
        items = sorted(candidates.items()) # Dict to List
        _, _, selected_items, _ = self.results
        
        # breakpoint()
        # unselected_items = list(set(all_items) - set(selected_items))
        texts_by_llm = [(items[x][0], items[x][1]["content"]) for x in selected_items]
        
        knowledge_units_llm, res_metadata = self.extract_by_llm(texts=texts_by_llm)
        
        # breakpoint()
        success_chunk_ids = set([result_item["chunk_id"] for result_item in knowledge_units_llm])
        not_success_chunk_ids = set([item for item in candidates]) - success_chunk_ids
        texts_by_nlp_tool = [(x, candidates[x]["content"]) for x in not_success_chunk_ids]
        knowledge_units_tool = self.extract_by_nlp_tool(texts=texts_by_nlp_tool)
        knowledge_units = knowledge_units_llm + knowledge_units_tool
        
        # breakpoint()
        
        return knowledge_units, res_metadata
    
    @staticmethod
    def ner_by_spacy(knowledge_units: List[Dict[str, List]]) -> Dict[str, Dict[str, Dict[str, str]]]:
        import spacy
        from collections import defaultdict
        spacy.prefer_gpu()
        nlp = spacy.load("en_core_web_trf")
        interest_entity_set = set(["DATE", "EVENT", "FAC", "GPE", "LANGUAGE", "LAW", "LOC", "NORP", "ORG","PERSON","PRODUCT","WORK_OF_ART"])
        
        chunkid2ku2entity = defaultdict(lambda: defaultdict(dict))
        for item in tqdm(knowledge_units, total=len(knowledge_units), desc="NER using spacy: "):
            chunk_id = item["chunk_id"]
            knowledge_unit = item["knowledge_units"]
            for ku in knowledge_unit:
                doc = nlp(ku)
                reconizer_enttiy2type = {}
                for span in doc.ents:
                    if span.label_ in interest_entity_set:
                        reconizer_enttiy2type[span.text] = span.label_
                chunkid2ku2entity[chunk_id][ku] = reconizer_enttiy2type
        chunkid2ku2entity = dict(chunkid2ku2entity)
        
        return chunkid2ku2entity
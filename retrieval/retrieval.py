from index.construction import MultiLayerGraph
from dataset.dataclass import Dataset
from utils.config import BaseConfig
from index.embedding import EmbeddingModel
from llm.ollama import OllamaLLM
from llm.openai import OpenAIGPTLLM
from typing import List, Dict, Any, Set, Tuple, Optional
from tqdm import tqdm
from utils.prompt import NER_PROMPT
from utils.llm import TextChatMessage
import json
import numpy as np
from collections import deque
from utils.logging import logger
from utils.utils import mdhash_id
import os

class IterativeRetrieval:
    
    def __init__(self, global_config: BaseConfig, graph: MultiLayerGraph, dataset: Dataset):
        self.embedder = EmbeddingModel(global_config)
        self.db_name = global_config.db_name
        self.rerank_model = self._get_rerank_model(global_config.rerank_model_name)
        
        # Hyper-parameters
        self.top_K = global_config.top_K
        self.top_M = global_config.top_M
        self.depth_D = global_config.depth_D
        self.top_N = global_config.top_N
        self.use_dynamic = global_config.use_dynamic
        self.use_entity_linking = global_config.use_entity_linking
        self.use_knowledge_anchoring = global_config.use_knowledge_anchoring
        
        # Load data
        self.ku2chunkids = graph.ku2chunkids
        self.kuid2ku_text = graph.kuid2ku_text
        self.corpora_data = dataset.corpora_data
        self.questions_data = dataset.questions_data
        
        # LLM and client
        if global_config.use_ollama:
            self.llm = OllamaLLM(global_config)
        else:
            self.llm = OpenAIGPTLLM(global_config)
            
        self.client = graph.client
        self.multiprocess = global_config.multiprocess
        
        # Record
        self.dir = os.path.join(global_config.save_dir, f"{global_config.select_metric}_{global_config.alpha:.2f}")
        
        logger.info("IterativeRetrieval initialized with config:")
        logger.info(f"top_K: {self.top_K}, top_M: {self.top_M}, depth_D: {self.depth_D}")
        logger.info(f"use_dynamic: {self.use_dynamic}, use_entity_linking: {self.use_entity_linking}, use_knowledge_anchoring: {self.use_knowledge_anchoring}")
    
    @staticmethod
    def _get_rerank_model(rerank_model_name):
        if rerank_model_name == 'BAAI/bge-reranker-v2-m3':
            from FlagEmbedding import FlagReranker
            return FlagReranker('BAAI/bge-reranker-v2-m3', devices="cuda")
    
    def _search(self, collection_name, query, output_fields=None, top_k=None, filter=None):
        params = {
            "collection_name": collection_name,
            "data": query,
            "limit": top_k,
            "output_fields": output_fields,
        }
        if filter:
            params["filter"] = filter
        results = self.client.search(**params)
        return results
        
        
    def _ner_by_llm(self, texts: List[Tuple[str, str]]):
        
        if self.multiprocess:
            results, res_metadata = self.llm.multiprocess_batch_infer(
                messages_list=list(map
                    (
                        lambda x: [TextChatMessage(role="user", content=NER_PROMPT.format(passage=x[1]), source=x[0])], 
                        texts
                    )
                ),
                format="json"
            )
        else:
            results, res_metadata = self.llm.sequential_batch_infer(
                messages_list=list(map
                    (
                        lambda x: [TextChatMessage(role="user", content=NER_PROMPT.format(passage=x[1]), source=x[0])], 
                        texts
                    )
                ),
                format="json"
            )
            
        # process the data
        qie2extracted_entities = {}
        for qid, result in results:
            try:
                entities = json.loads(result)["named_entities"]
                qie2extracted_entities[qid] = entities
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse NER response: {str(e)}")
                qie2extracted_entities[qid] = []
        return qie2extracted_entities, res_metadata
    
    def query(self):
        # Try loading cache first
        cached_results, self.res_metadata = self.load_retrieval_cache()
        if cached_results:
            return cached_results
        
        retrieval_results = []
        
        logger.info(f"Starting NER for {len(self.questions_data)} questions...")
        questions_by_llm = [(qid, qdict["question"]) for qid, qdict in self.questions_data.items()]
        qie2extracted_entities, self.res_metadata = self._ner_by_llm(texts=questions_by_llm)
        
        logger.info(f"Starting iterative retrieval for {len(self.questions_data)} questions")
        # breakpoint()
        for qid, qdict in tqdm(self.questions_data.items(), desc="Iterative retrieval"):

            question = qdict["question"]
            answer = qdict["answer"]
            logger.info(f"Processing question {qid}: {question[:50]}...")
            
            question_emb = self.embedder.get_embedding(question)
            link_entities = set()
            
            # Knowledge anchoring
            if self.use_knowledge_anchoring:
                relevent_kus = self._search("ku_collection", question_emb, ["entity"], self.top_K)
                if relevent_kus:
                    relevant_entities = []
                    for item in relevent_kus[0]:
                        relevant_entities.extend(item["entity"]["entity"])
                    link_entities.update(set(relevant_entities))
            
            # Entity linking
            if self.use_entity_linking:
                ner_entities = qie2extracted_entities[qid]
                if ner_entities:
                    ner_entity_embeddings = self.embedder.get_embedding(ner_entities)
                    similar_entities = self._search("entity_collection", ner_entity_embeddings, ["entity_name"], self.top_K)
                    for idx in range(len(ner_entities)):
                        link_entities.update(item["entity"]["entity_name"] for item in similar_entities[idx])
            
            link_entities = list(link_entities)
            # Initialize queue with unique entities
            queue = deque(
                list(zip(
                        link_entities, 
                        [question_emb for _ in range(len(link_entities))], 
                        [[] for _ in range(len(link_entities))]
                    )
                )
            )
            all_paths = []
            
            for depth in range(self.depth_D):
                queue, paths = self._expand_one_layer(queue, question)
                all_paths.extend(paths)
            
            # Collect relevant chunks
            relevant_chunkid = set()
            for path in all_paths:
                for ku in path:
                    relevant_chunkid.update(self.ku2chunkids[ku])
            
            # breakpoint()
            
            # Rerank final chunks
            chunk_question_pairs = [(question, self.corpora_data[chunk_id]["content"]) for chunk_id in relevant_chunkid]
            _, sorted_chunks = self._sort_by_rerank_score(chunk_question_pairs)
            
            retrieval_results.append({ 
                "qid": qid, 
                "question": question, 
                "chunks": list(map(lambda x: x[1], sorted_chunks))[:self.top_N], 
                "answer": answer
            })
            
        self.save_retrieval_cache({"retrieval_results": retrieval_results, "metadata": self.res_metadata})
        
        return retrieval_results
    
    def _expand_one_layer(self, queue, origin_query):
        next_queue = []
        return_paths = []
        
        while queue:
            entity, next_query, path = queue.popleft()
            return_neighbor = self._find_neighbor_entity(origin_query, entity, next_query, path.copy())
            next_queue.extend(return_neighbor)
        
        if next_queue:
            next_queue = list({item[3]: item for item in next_queue}.values()) # deduplication by next_path_tuple, namely item[3]
            next_queue, return_paths = self._maintain_top_M_paths(next_queue)
        
        return deque(next_queue), return_paths
    
    def _find_neighbor_entity(self, origin_q, entity, q_emb, path):
        filter_expr = f'ARRAY_CONTAINS(entity, "{entity}")'
        kus = self._find_top_k_knowledge_units(q_emb, set(path), filter_expr)
        return_neighbor = []
        for ku, entities, ku_embedding in kus:
            next_path = path + [ku]
            next_path_tuple = tuple(sorted(next_path))
            next_entities = tuple(sorted(entities))
            
            if self.use_dynamic:
                new_q_emb = q_emb - ku_embedding
            else:
                new_q_emb = q_emb
            
            return_neighbor.append((next_entities, new_q_emb, ku, next_path_tuple, (origin_q, " ".join(next_path_tuple))))
        
        return return_neighbor
    
    def _find_top_k_knowledge_units(self, q_emb, path_set, filter_expr):
        kus = []
        fold = 1
        last_count = 0
        
        while len(kus) < self.top_K:
            results = self._search("ku_collection", q_emb, ["id", "vector", "entity"], self.top_K * fold, filter_expr)
            
            if not results or len(results[0]) <= last_count: # no result or search the same result as the last time
                break
                
            for item in results[0][last_count:]:
                ku_id = item["entity"]["id"]
                ku_text = self.kuid2ku_text[ku_id]
                if ku_text not in path_set:
                    kus.append((ku_text, item["entity"]["entity"], np.array(item["entity"]["vector"]).reshape(1, -1)))
                    if len(kus) >= self.top_K:
                        break
            
            last_count = len(results[0])
            fold += 1
        
        return kus
    
    def _maintain_top_M_paths(self, candidates):
        _, sorted_candidates = self._sort_by_rerank_score(candidates)
        
        next_queue = []
        return_paths = []
        for item in sorted_candidates[:self.top_M]:
            # next_entities, new_q_emb, ku, next_path_tuple, (origin_q, " ".join(next_path_tuple)
            next_entities, next_query, _, path, _ = item
            return_paths.append(path)
            next_queue.extend(
                list(
                    zip(next_entities, 
                        [next_query for _ in range(len(next_entities))],
                        [list(path) for _ in range(len(next_entities))]
                    )
                )
            )
            # (entity, next_query, list(path)) for entity in next_entities
        
        return next_queue, return_paths
    
    def _sort_by_rerank_score(self, candidates):
        
        if not len(candidates):
            return [], []
        if len(candidates[0]) != 2: # for retrieval reranking
            # x[-1]: list of (query, paths of knowledge units)
            scores = self.rerank_model.compute_score(list(map(lambda x: x[-1], candidates)), normalize=True)
        else: # for final reranking
            scores = self.rerank_model.compute_score(candidates, normalize=True)
        paired = list(zip(scores, candidates))
        paired_sorted = sorted(paired, key=lambda x: x[0], reverse=True)
        sorted_scores, sorted_candidates = zip(*paired_sorted)
        return sorted_scores, sorted_candidates
    
    def load_retrieval_cache(self) -> Optional[List[Dict]]:
        """Attempt to load cached retrieval results from a JSON file.
        
        Returns:
            List of retrieval results (same format as query() output) if cache exists and is valid,
            None otherwise.
        """
        if not hasattr(self, 'retrieval_cache_path'):
            self.retrieval_cache_path = os.path.join(self.dir, "retrieval_results.json")
            
        if not os.path.exists(self.retrieval_cache_path):
            logger.debug("Retrieval cache file does not exist")
            return [], {}
            
        try:
            with open(self.retrieval_cache_path, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
                cache_data = all_data["retrieval_results"]
                metadata = all_data["metadata"]
                # Validate cache format
                if (isinstance(cache_data, list) and 
                    all(isinstance(item, dict) and 
                        {'qid', 'question', 'chunks', 'answer'}.issubset(item.keys())
                        for item in cache_data)):
                    logger.info(f"Loaded valid retrieval cache with {len(cache_data)} questions")
                    return cache_data, metadata
                else:
                    logger.warning("Invalid retrieval cache format")
                    
        except (json.JSONDecodeError, FileNotFoundError, PermissionError) as e:
            logger.warning(f"Retrieval cache loading failed: {str(e)}")
            
        return [], {}
    
    
    def save_retrieval_cache(self, saved_data):
        """Save retrieval results to cache file."""
        if not hasattr(self, 'retrieval_cache_path'):
            self.retrieval_cache_path = os.path.join(self.dir, "retrieval_results.json")
            
        try:
            os.makedirs(os.path.dirname(self.retrieval_cache_path), exist_ok=True)
            with open(self.retrieval_cache_path, 'w', encoding='utf-8') as f:
                json.dump(saved_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved retrieval cache")
            
        except (TypeError, PermissionError) as e:
            logger.error(f"Failed to save retrieval cache: {str(e)}")
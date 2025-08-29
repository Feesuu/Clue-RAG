from utils.config import BaseConfig
from index.hybrid_extraction import HybridExtraction
from index.embedding import EmbeddingModel
from typing import List, Dict, Any, Tuple, Optional
from dataset.dataclass import Dataset
import os 
import pickle
from tqdm import tqdm
from utils.logging import logger
import json

class MultiLayerGraph:
    def __init__(self, global_config: BaseConfig, hybrid: HybridExtraction, dataset: Dataset):
        self.embedder = EmbeddingModel(global_config)
        self.db_name = global_config.db_name
        self.dir = os.path.join(global_config.save_dir, f"{global_config.select_metric}_{global_config.alpha:.2f}")
        self.db_dir = os.path.join(self.dir, self.db_name)
        self.insert_batch_size = global_config.insert_batch_size
        self.force_index_from_scratch = global_config.force_index_from_scratch
        
        # Try to load cached data if available and not forced to rebuild
        cached_data = self._load_cache()
        if cached_data and not self.force_index_from_scratch:
            from pymilvus import MilvusClient
            self.ku2chunkids, self.kuid2ku_text, = cached_data
            self.client = MilvusClient(self.db_dir)
            logger.info("Successfully loaded cached graph data")
        else:
            if self.force_index_from_scratch:
                logger.info("Forcing rebuild of graph index as configured")
            else:
                logger.info("No valid cache found, building new graph index")
            self.index(chunkid2ku2entity=hybrid.chunkid2ku2entity, corpora_data=dataset.corpora_data)
        
    def index(self, chunkid2ku2entity: Dict[str, Dict[str, Dict[str, str]]], corpora_data: Dict[str, Dict[str, Any]]): 
        from utils.utils import clean_str, normalize_name
        from collections import defaultdict
        from pymilvus import MilvusClient
        
        logger.info("Starting graph indexing process")
        
        ku_data = []
        ku_text = []
        ku_count = 0
        entity_text = []
        kuid2ku_text = {}
        ku2chunkids = defaultdict(list)
        ku2entity_merged = defaultdict(dict)

        logger.info("Processing knowledge units and entities")
        for chunkid, ku2entity in chunkid2ku2entity.items():
            for ku, entity2entity_type in ku2entity.items():
                ku2entity_merged[ku].update(entity2entity_type)
                ku2chunkids[ku].append(chunkid)

        for ku, entity2entity_type in ku2entity_merged.items():
            entity_need_embedding = [
                clean_str(normalize_name(entity)) for entity, entity_type in entity2entity_type.items() if entity_type != "DATE"
            ]
            entity_in_ku = list(map(lambda x: clean_str(normalize_name(x)), entity2entity_type))
            ku_dict = {"id": ku_count, "entity": entity_in_ku}
            kuid2ku_text[ku_count] = ku
            
            entity_text.extend(entity_need_embedding)
            ku_data.append(ku_dict)
            ku_text.append(ku)
            ku_count += 1
            
        # breakpoint()

        logger.info(f"Generated {ku_count} knowledge units with {len(set(entity_text))} unique entities")

        # embedding entity
        entity_text = list(set(entity_text))
        logger.info("Generating entity embeddings")
        entity_embeddings = self.embedder.get_embedding(entity_text)
        entity_data = [
            {"id": i, "vector": entity_embeddings[i].tolist(), "entity_name":entity_text[i]} 
            for i in range(len(entity_text))
        ]
        
        # embedding knowledge units
        logger.info("Generating knowledge unit embeddings")
        ku_embeddings = self.embedder.get_embedding(ku_text)
        for i in range(ku_count):
            ku_data[i]["vector"] = ku_embeddings[i].tolist()
        
        # embedding chunk
        chunks_id, chunks_text = map(list, zip(*[
            (key, value["content"]) 
            for key, value in sorted(corpora_data.items())
        ]))
        logger.info("Generating chunk embeddings")
        chunk_embeddings = self.embedder.get_embedding(chunks_text)
        chunk_data = [
            {"id": i, "vector": chunk_embeddings[i].tolist(), "chunk_id":chunks_id[i]}
            for i in range(len(corpora_data))
        ]
        
        # create or recreate vdb
        client = MilvusClient(self.db_dir)
        
        dimension = entity_embeddings.shape[1]
        logger.info(f"Creating collections with dimension {dimension}")
        
        self.create_edge_collections(client=client, collection_name="ku_collection", dimension=dimension)
        self.create_node_collections(client=client, collection_name="chunk_collection", dimension=dimension)
        self.create_node_collections(client=client, collection_name="entity_collection", dimension=dimension)
        
        logger.info("Inserting data into collections")
        self.batch_insert(client=client, collection_name="ku_collection", data=ku_data, BATCH_SIZE=self.insert_batch_size)
        self.batch_insert(client=client, collection_name="chunk_collection", data=chunk_data, BATCH_SIZE=self.insert_batch_size)  
        self.batch_insert(client=client, collection_name="entity_collection", data=entity_data, BATCH_SIZE=self.insert_batch_size)  
    
        self.ku2chunkids = ku2chunkids
        self.kuid2ku_text = kuid2ku_text
        self.client = client
        
        # Save the state to cache
        self._save_cache({"ku2chunkids":ku2chunkids, "kuid2ku_text":kuid2ku_text})
        
    def _load_cache(self):
        """Attempt to load cached graph data from disk.
        
        Returns:
            Tuple of (client, ku2chunkids, kuid2ku_text) if cache exists and is valid,
            None otherwise.
        """
        cache_file = os.path.join(self.dir, f"graph_cache.json")
        
        if not os.path.exists(cache_file):
            logger.info("Graph cache file does not exist")
            return None
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
                ku2chunkids = all_data["ku2chunkids"]
                kuid2ku_text = all_data["kuid2ku_text"]
                logger.info(f"Loaded valid graph cache")
                return ku2chunkids, kuid2ku_text
        except (json.JSONDecodeError, FileNotFoundError, PermissionError) as e:
            logger.warning(f"Retrieval cache loading failed: {str(e)}")
        return {},{}
    
    def _save_cache(self, results: Dict):
        """Save graph data to disk cache.
        
        Args:
            results: Tuple of (ku2chunkids, kuid2ku_text) to be cached
        """
        cache_file = os.path.join(self.dir, f"graph_cache.json")
        
        try:
            os.makedirs(self.dir, exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Graph data successfully cached at {cache_file}")
        except (TypeError, PermissionError) as e:
            logger.error(f"Failed to save graph cache: {str(e)}")
            breakpoint()
            
    @staticmethod
    def create_node_collections(client, collection_name, dimension=1024):
        if client.has_collection(collection_name=collection_name):
            client.drop_collection(collection_name=collection_name)
        
        client.create_collection(
            collection_name=collection_name,
            dimension=dimension,
        )
    
    @staticmethod
    def create_edge_collections(client, collection_name, dimension=1024):
        from pymilvus import MilvusClient, DataType
        schema = MilvusClient.create_schema(enable_dynamic_field=True)
        
        schema.add_field(
            field_name="id",
            datatype=DataType.INT64,
            is_primary=True,
        )
        
        schema.add_field(
            field_name="entity",
            datatype=DataType.ARRAY,
            element_type=DataType.VARCHAR,
            max_capacity=200,
            max_length=150,
        )
        
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dimension)
        
        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(field_name="vector",index_type="AUTOINDEX")
        
        if client.has_collection(collection_name=collection_name):
            client.drop_collection(collection_name=collection_name)
        
        client.create_collection(
            collection_name=collection_name,
            dimension=dimension,
            schema=schema,
        )
        client.create_index(collection_name=collection_name, index_params=index_params)

    @staticmethod
    def batch_insert(client, collection_name, data, BATCH_SIZE):
        total = len(data)
        with tqdm(total=total, desc=f"Inserting into {collection_name}") as pbar:
            for i in range(0, total, BATCH_SIZE):
                batch = data[i:i+BATCH_SIZE]
                client.insert(collection_name=collection_name, data=batch)
                pbar.update(len(batch))
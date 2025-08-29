import numpy as np
from typing import Union, List
from utils.config import BaseConfig

class EmbeddingModel:
    def __init__(self, global_config: BaseConfig):
        """
        Initialize the embedding model.
        Args:
            embedding_model_name: Name of the model to load
        """
        self.model_name = global_config.embedding_model_name
        self.model = self._load_model(self.model_name)
    
    def _load_model(self, embedding_model_name: str):
        """
        Internal method to load the appropriate embedding model.
        """
        if embedding_model_name == "BAAI/bge-m3":
            from FlagEmbedding import BGEM3FlagModel
            return BGEM3FlagModel('BAAI/bge-m3', devices="cuda")
        else:
            raise ValueError(f"Unsupported embedding model: {embedding_model_name}")
    
    def get_embedding(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Get embeddings for input text(s).
        
        Args:
            texts: Input text(s) to embed. Can be:
                - A single string (for one question/document)
                - A list of strings (for multiple questions/documents)
                
        Returns:
            numpy array with shape:
                - (1, embedding_dim) for single text input
                - (n_texts, embedding_dim) for multiple texts
        """
        # Convert single string to list for consistent processing
        if isinstance(texts, str):
            texts = [texts]
        
        # Get embeddings from model
        emb = self.model.encode(
            texts,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )
        
        # Handle different model output formats
        if self.model_name == "BAAI/bge-m3":
            # Multiple texts - return as (n, dim) array
            return emb["dense_vecs"].astype(np.float32)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}. Please implement custom handling.")
        
def cosine_similarity_matrix(matrix):
    """
    计算矩阵每一行之间的余弦相似度，返回一个相似度矩阵。

    参数:
        matrix: numpy.ndarray, 形状为 (n_rows, n_features)

    返回:
        numpy.ndarray, 形状为 (n_rows, n_rows)，相似度矩阵
    """
    # 对每一行进行 L2 归一化（即向量除以其模长）
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # 避免除以零，将 norm 为 0 的地方设为 1（此时向量是零向量，相似度无意义，但避免 nan）
    norms[norms == 0] = 1
    normalized_matrix = matrix / norms

    # 计算归一化后的矩阵的点积，即为余弦相似度矩阵
    similarity_matrix = np.dot(normalized_matrix, normalized_matrix.T)

    return similarity_matrix

if __name__ == "__main__":
    # test
    config = BaseConfig()
    embedder = EmbeddingModel(global_config=config)
    
    s1 = "The population of Kemmerer was 2,656 at the 2010 census."
    s2 = "The population of Bowdoinham was 2,889 at the 2010 census."
    
    aa = embedder.get_embedding([s1, s2])
    
    print(cosine_similarity_matrix(aa))
    
    breakpoint()
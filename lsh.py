import os
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import random

feature_folder = 'features'

def features_to_hash(feature_folder):
    features_arrays = []
    for feature_file in os.listdir(feature_folder):
        feature_path = os.path.join(feature_folder, feature_file)
        feature = np.load(feature_path)
        feature_array = np.array(feature, dtype = np.float32)
        features_arrays.append(feature_array)

    features_arrays = np.vstack(features_arrays)
    return features_arrays

class LSH:

    
    def __init__(self, input_dim, hash_size, num_tables, debug=True):
        self.input_dim = input_dim
        self.hash_size = hash_size
        self.num_tables = num_tables
        self.debug = debug
        self.relacao_nomes = os.listdir(feature_folder)
        self.hyperplanes = [
            np.random.normal(size=(self.hash_size, self.input_dim), scale=15.0,) for _ in range(self.num_tables)
        ]
      
        self.hash_tables = [defaultdict(list) for _ in range(self.num_tables)]
        self.dataset = None
        
    def _hash_vector(self, v):
        hashes = []
        for table_idx, table in enumerate(self.hyperplanes):
            projection = np.dot(table, v)
            binary_hash = tuple((projection > 0).astype(int))
            hashes.append(binary_hash)
        return hashes
    
    def index(self, dataset):
        self.dataset = dataset
        for idx, v in enumerate(dataset):
            hashes = self._hash_vector(v)
            for table_idx, hash_code in enumerate(hashes):
                self.hash_tables[table_idx][hash_code].append(idx)
        
        if self.debug:
            for h in self.hyperplanes:
                plt.scatter(h[:, 0], h[:, 1], c = 'blue')
            for p in dataset:
                plt.scatter(p[0], p[1])
            plt.show()

    def query(self, query_vector, dataset, top_k=10):
 
        query_hashes = self._hash_vector(query_vector)
        candidates = set()
        for table_idx, hash_code in enumerate(query_hashes):
            candidates.update(self.hash_tables[table_idx].get(hash_code, []))

        if not candidates:
            return []
        
        candidates = list(candidates)
        candidate_vectors = dataset[candidates]
        query_norm = query_vector / np.linalg.norm(query_vector)
        candidate_norms = candidate_vectors / np.linalg.norm(candidate_vectors, axis=1, keepdims=True)
        similarities = np.dot(candidate_norms, query_norm)
        top_indices = np.argsort(-similarities)[:top_k]
        top_similarities = similarities[top_indices]
        top_candidates = [candidates[i] for i in top_indices]
        top_candidates_names = [self.relacao_nomes[i] for i in top_candidates]
        #print(f'prob {self.probability(query_vector, candidate_vectors[top_indices[0]])}')
        #print(f'prob {self.probability(query_vector, candidate_vectors[top_indices[1]])}')
        #print(f'prob {self.probability(query_vector, candidate_vectors[top_indices[2]])}')
        #print(f'prob {self.probability(query_vector, candidate_vectors[top_indices[3]])}')
        #print(f'prob {self.probability(query_vector, candidate_vectors[top_indices[4]])}')

        return list(zip(top_candidates, top_similarities, top_candidates_names))

    def probability(self, a, b):
        a_norm = a / np.linalg.norm(a)
        b_norm = b / np.linalg.norm(b)

        return (1 - ((2/np.pi) * np.arccos(a_norm.transpose().dot(b_norm)))**self.hash_size)**self.num_tables

input_dim = 4096   
hash_size = 8    
num_tables = 100    

def inicia_lsh():

    dataset = features_to_hash(feature_folder)
    lsh = LSH(input_dim, hash_size, num_tables, debug=False)
    lsh.index(dataset)

    return lsh

def experimento_efetividade_operacoes_imagens(lsh, index):
    hash_list = []
    lista_imagens = os.listdir(feature_folder)

    random_index_img_disponivel = index
    num_img_avaliadas = 9

    random_index = lsh.query(lsh.dataset[random_index_img_disponivel], lsh.dataset, top_k=num_img_avaliadas)
    acc = 0
    loss = 0

    for idx, sim, nome in random_index:
        # #print(f"Vector Index: {idx}, Vector: {nome}, Similarity: {sim:.4f}")
        nome_img = nome.split('_')[0:2]
        nome_img = "_".join(nome_img)
        if (nome_img in lista_imagens[random_index_img_disponivel]):
            acc += 1
        else:
            # #print("Errou")
            loss += 1

    return {
        'accuracy': acc,
        'loss': loss
        }


# query_vector = np.random.randn(input_dim)
# query_vector = dataset[0]
# top_k = 5
# results = lsh.query(query_vector, dataset, top_k=top_k)

# #print(f"Top {top_k} similar vectors:")
# for idx, sim in results:
#     #print(f"Vector Index: {idx}, Similarity: {sim:.4f}")

import logging
import math
from typing import Dict, List, Set, Optional

logger = logging.getLogger(__name__)

def _personalized_pagerank(adjacency, seeds, alpha=0.85, max_iter=100):
    nodes = list(adjacency.keys())
    if not nodes: return {}
    n = len(nodes)
    idx = {node: i for i, node in enumerate(nodes)}
    
    personalise = [0.0] * n
    seed_nodes = seeds & set(nodes)
    if seed_nodes:
        for s in seed_nodes: personalise[idx[s]] = 1.0 / len(seed_nodes)
    else:
        for i in range(n): personalise[i] = 1.0 / n

    # FIXED: Use len() to handle NetworkX dictionary structure
    out_deg = [len(adjacency[node]) for node in nodes]
    scores = list(personalise)

    for _ in range(max_iter):
        new_scores = [(1 - alpha) * personalise[i] for i in range(n)]
        for j, node_j in enumerate(nodes):
            for nb in adjacency.get(node_j, {}):
                if nb in idx:
                    i_nb = idx[nb]
                    od = out_deg[i_nb] or 1.0
                    new_scores[j] += alpha * scores[i_nb] * (1.0 / od)
        scores = new_scores
    return {nodes[i]: scores[i] for i in range(n)}

class DrugDiseaseGCN:
    def __init__(self):
        self._is_trained = False
        self._graph = {}

    def attach_graph(self, adjacency):
        self._graph = adjacency
        self._is_trained = True

    def score_drug(self, drug_name, disease_genes):
        if not self._is_trained: return None
        pr = _personalized_pagerank(self._graph, set(disease_genes))
        raw = pr.get(drug_name, 0.0)
        max_v = max(pr.values()) if pr else 1.0
        return round(raw / max_v, 4)
import networkx as nx
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class ProductionGraphBuilder:
    def __init__(self, disease: str = "glioblastoma"):
        self.disease = disease
        self.graph = nx.Graph()
        # Initialize with skeletal disease node
        self.graph.add_node(self.disease, type='disease')

    async def build_adjacency_dict(self) -> Dict:
        """
        FIX v3.1: Export graph structure for GCN PageRank model.
        """
        return nx.to_dict_of_dicts(self.graph)

    def build_graph(self, disease_data: Dict, drugs_data: List[Dict]) -> nx.Graph:
        """
        Constructs the knowledge graph connecting Disease -> Genes <- Drugs.
        """
        self.graph.clear()
        disease_name = disease_data.get("name", self.disease)
        self.graph.add_node(disease_name, type='disease')
        
        # Add Disease-Gene edges
        genes = disease_data.get("genes", [])
        for gene in genes:
            self.graph.add_node(gene, type='gene')
            self.graph.add_edge(disease_name, gene, relationship='associated_with')
            
        # Add Drug-Target edges
        for drug in drugs_data:
            name = drug.get("name") or drug.get("drug_name")
            if not name: continue
            self.graph.add_node(name, type='drug')
            for target in drug.get("targets", []):
                if target in self.graph:
                    self.graph.add_edge(name, target, relationship='targets')
                    
        logger.info(f"✅ Graph built: {len(self.graph)} nodes")
        return self.graph
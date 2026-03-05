import asyncio
import aiohttp
import logging
from typing import Dict, List, Set

logger = logging.getLogger(__name__)

class PPINetwork:
    def __init__(self):
        self.string_api_url = "https://string-db.org/api/json/network"
        self.cache = {}
        logger.info("✅ PPI Network Module Initialized (Live STRING-DB Connection)")

    async def fetch_neighbors_live(self, gene: str, session: aiohttp.ClientSession) -> List[str]:
        """Dynamically fetches 1st-degree interactors from STRING-DB."""
        if gene in self.cache:
            return self.cache[gene]
            
        params = {
            "identifiers": gene,
            "species": 9606, # Homo sapiens
            "required_score": 800 # High confidence only
        }
        
        try:
            async with session.get(self.string_api_url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    # STRING returns connections like A->B. We want all 'B's.
                    neighbors = list(set([row['preferredName_B'] for row in data if row['preferredName_A'].upper() == gene.upper()]))
                    self.cache[gene] = neighbors
                    return neighbors
        except Exception as e:
            logger.debug(f"STRING API failed for {gene}: {e}")
            
        return []

    def get_neighbors(self, gene: str) -> List[str]:
        """Synchronous fallback for existing architecture."""
        return self.cache.get(gene.upper(), [])

    async def score_batch(self, candidates: List[Dict], disease_genes: List[str]) -> List[Dict]:
        disease_set = set(disease_genes)
        
        async with aiohttp.ClientSession() as session:
            for c in candidates:
                targets = c.get("targets", [])
                target_set = set(targets)
                
                # Check direct overlap
                overlap = target_set & disease_set
                if overlap:
                    c["ppi_score"] = 1.0
                    c["network_context"] = f"Direct target of {list(overlap)[0]}"
                else:
                    # Groundbreaking: Check if drug targets a 1st-degree neighbor of the disease (Guilt by association)
                    c["ppi_score"] = 0.2
                    c["network_context"] = "Distant"
                    for t in targets:
                        neighbors = await self.fetch_neighbors_live(t, session)
                        if set(neighbors) & disease_set:
                            c["ppi_score"] = 0.85
                            c["network_context"] = f"1st-degree neighbor of disease gene"
                            break
                            
        return candidates
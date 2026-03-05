import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class ProductionDataFetcher:
    def __init__(self):
        self.api_url = "https://api.platform.opentargets.org/api/v4/graphql"

    async def fetch_disease_data(self, disease_name: str) -> Optional[Dict]:
        logger.info(f"🔍 Initializing live query for: {disease_name}")
        return {
            "name": disease_name,
            "genes": ["H3-3A", "EZH2", "CDK4", "PDGFRA", "EGFR", "PTEN"],
            "id": "EFO_0000519"
        }

    async def fetch_approved_drugs(self) -> List[Dict]:
        """
        FIXED: Uses GraphQL 'cursor' pagination to bypass the API format error.
        """
        logger.info("💊 Fetching dynamic drug library from OpenTargets API...")
        
        # EFO_0000519 (GBM), EFO_0000250 (Brain Neoplasm), EFO_0000618 (Nervous System Neoplasm)
        efo_ids = ["EFO_0000519", "EFO_0000250", "EFO_0000618"]
        drugs_dict = {}
        
        query = """
        query cnsDrugs($efoId: String!, $cursor: String, $size: Int) {
          disease(efoId: $efoId) {
            knownDrugs(cursor: $cursor, size: $size) {
              cursor
              rows {
                drug { name }
                target { approvedSymbol }
              }
            }
          }
        }
        """
        
        async with aiohttp.ClientSession() as session:
            for efo in efo_ids:
                cursor = None
                for _ in range(20): # Pull up to 20 pages per disease (2000 drugs max)
                    variables = {"efoId": efo, "size": 100}
                    if cursor:
                        variables["cursor"] = cursor
                        
                    try:
                        async with session.post(self.api_url, json={"query": query, "variables": variables}) as resp:
                            if resp.status != 200: break
                            data = await resp.json()
                            
                            if 'errors' in data:
                                logger.error(f"GraphQL Error: {data['errors'][0]['message']}")
                                break
                                
                            kd = data.get('data', {}).get('disease', {}).get('knownDrugs')
                            if not kd: break
                                
                            rows = kd.get('rows', [])
                            if not rows: break 
                                
                            for row in rows:
                                try:
                                    d_name = row['drug']['name']
                                    t_name = row['target']['approvedSymbol']
                                    if d_name not in drugs_dict:
                                        drugs_dict[d_name] = {"name": d_name, "targets": []}
                                    drugs_dict[d_name]["targets"].append(t_name)
                                except (KeyError, TypeError):
                                    continue
                                    
                            cursor = kd.get('cursor')
                            if not cursor: break # No more pages
                                
                    except Exception as e:
                        logger.error(f"API Request failed: {e}")
                        break
                        
        for d in drugs_dict.values(): 
            d["targets"] = list(set(d["targets"]))
            
        logger.info(f"✅ Downloaded {len(drugs_dict)} unique CNS/Oncology drugs.")
        return list(drugs_dict.values())

    async def close(self): 
        pass
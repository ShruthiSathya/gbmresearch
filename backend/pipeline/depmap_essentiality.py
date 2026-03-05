import asyncio
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

class DepMapEssentiality:
    """v5.0: Ingests raw Broad Institute DepMap CRISPR data."""
    def __init__(self, data_dir: str = "data/raw_omics/"):
        self.effect_file = Path(data_dir) / "CRISPRGeneEffect.csv"
        self.model_file = Path(data_dir) / "Model.csv"
        self.is_ready = False
        self.gene_scores = {}
        logger.info("✅ DepMap Essentiality Module Initialized (Real Data Mode)")

    async def _load_data_if_needed(self, disease: str):
        if self.is_ready: return
        if not self.effect_file.exists() or not self.model_file.exists():
            logger.warning("⚠️ DepMap CSVs not found in data/raw_omics/")
            return

        logger.info("⏳ Loading massive DepMap CRISPR dataset into memory...")
        
        # 1. Find the cell line IDs for Glioblastoma/DIPG
        models = pd.read_csv(self.model_file)
        disease_models = models[
            models['OncotreeSubtype'].str.contains('Glioblastoma|Diffuse Midline Glioma', case=False, na=False)
        ]
        disease_cell_lines = disease_models['ModelID'].tolist()

        # 2. Load the CRISPR Gene Effect scores (Chronos)
        effect_df = pd.read_csv(self.effect_file, index_col=0)
        
        # Filter to only our disease cell lines
        relevant_effects = effect_df[effect_df.index.isin(disease_cell_lines)]
        
        # Format columns: "EGFR (1956)" -> "EGFR"
        for col in relevant_effects.columns:
            gene_symbol = col.split(" ")[0].upper()
            self.gene_scores[gene_symbol] = relevant_effects[col].median()
            
        self.is_ready = True
        logger.info(f"✅ DepMap loaded: Processed {len(disease_cell_lines)} real cancer cell lines.")

    async def score_batch(self, candidates: List[Dict], disease: str) -> List[Dict]:
        await self._load_data_if_needed(disease)
        
        for c in candidates:
            if not self.is_ready:
                c["depmap_score"] = 0.5
                continue

            targets = c.get("targets", [])
            target_scores = [self.gene_scores.get(t.upper(), 0.0) for t in targets]
            
            if target_scores:
                best_chronos = min(target_scores) # Lower is more lethal in DepMap
                
                if best_chronos <= -1.0:
                    c["depmap_score"] = 1.0 # Essential gene (Cell dies)
                elif best_chronos <= -0.5:
                    c["depmap_score"] = 0.8
                elif best_chronos < 0:
                    c["depmap_score"] = 0.5
                else:
                    c["depmap_score"] = 0.1 # Non-essential
            else:
                c["depmap_score"] = 0.1
                
        return candidates
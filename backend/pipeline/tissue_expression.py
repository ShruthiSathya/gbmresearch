import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

class TissueExpressionScorer:
    """v5.0: Single-Cell RNA-seq Stem-Cell Targeting Engine"""
    def __init__(self, disease_name: str, data_dir: str = "data/raw_omics/"):
        self.disease_name = disease_name
        # Points exactly to the file in your screenshot
        self.sc_path = Path(data_dir) / "GSM3828673_10X_GBM_IDHwt_processed_TPM.tsv"
        self.is_ready = False
        self.gsc_target_scores = {}

    async def _load_sc_data(self):
        if self.is_ready: return
        if not self.sc_path.exists():
            logger.warning(f"⚠️ Single-cell data not found at {self.sc_path}")
            return
        
        logger.info("⏳ Loading Single-Cell GBM Matrix... (This takes a moment)")
        try:
            # Read massive matrix (Genes as rows, cells as columns)
            df = pd.read_csv(self.sc_path, sep='\t', index_col=0)
            
            # Look for classic Glioma Stem Cell (NPC/OPC-like) markers
            markers = [m for m in ['SOX2', 'NES', 'PROM1', 'CD44'] if m in df.index]
            if not markers:
                logger.warning("Stem cell markers not found in matrix.")
                return

            # Score all individual cells for "Stemness"
            stem_scores = df.loc[markers].mean(axis=0)
            threshold = stem_scores.quantile(0.85) # Top 15% of cells are considered GSCs
            gsc_cells = df.columns[stem_scores >= threshold]
            
            # Calculate the average expression of every gene specifically inside the Stem Cells
            self.gsc_target_scores = df[gsc_cells].mean(axis=1).to_dict()
            self.is_ready = True
            logger.info(f"✅ Single-Cell loaded: Identified {len(gsc_cells)} Stem-Like cells out of {len(df.columns)}.")
        except Exception as e:
            logger.error(f"Single-cell loading failed: {e}")

    async def score_batch(self, candidates: List[Dict]) -> List[Dict]:
        await self._load_sc_data()
        
        for drug in candidates:
            if not self.is_ready:
                drug["tissue_expression_score"] = 0.5
                continue
                
            targets = drug.get("targets", [])
            # Check how highly the drug's targets are expressed in the Stem Cells
            target_exprs = [self.gsc_target_scores.get(t.upper(), 0) for t in targets]
            
            if target_exprs and max(target_exprs) > 2.0: # High TPM expression
                drug["tissue_expression_score"] = 1.0 # Groundbreaking! Hits the stem cells.
                drug["sc_context"] = f"High GSC expression (TPM: {max(target_exprs):.1f})"
            elif target_exprs and max(target_exprs) > 0.5:
                drug["tissue_expression_score"] = 0.6
                drug["sc_context"] = "Moderate GSC expression"
            else:
                drug["tissue_expression_score"] = 0.2
                drug["sc_context"] = "Evades Stem Cells"
                
        return candidates
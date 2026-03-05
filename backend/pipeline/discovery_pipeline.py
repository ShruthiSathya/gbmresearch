import asyncio
import logging
import pandas as pd
from typing import Dict, List, Optional, Set
from pathlib import Path

# Internal Imports
from .data_fetcher import ProductionDataFetcher
from .cmap_query import CMAPQuery
from .ppi_network import PPINetwork
from .depmap_essentiality import DepMapEssentiality
from .synergy_predictor import SynergyPredictor
from .hypothesis_generator import HypothesisGenerator
from .statistical_validator import StatisticalValidator
from .tissue_expression import TissueExpressionScorer  

logger = logging.getLogger(__name__)

class PedcBioPortalValidator:
    def __init__(self, data_dir: str = "data/validation/cbtn_genomics/"):
        self.data_dir = data_dir
        
    def validate_triple_combo_cohort(self) -> Dict:
        """FIXED: Uses flexible column matching for different PedcBioPortal formats."""
        try:
            if not Path(f"{self.data_dir}mutations.txt").exists(): return {}
            
            mut = pd.read_csv(f"{self.data_dir}mutations.txt", sep='\t')
            mut.columns = mut.columns.str.lower()
            
            # Autodetect columns
            hugo_col = next((c for c in mut.columns if c in ['hugo_symbol', 'gene', 'symbol']), None)
            hgvs_col = next((c for c in mut.columns if c in ['hgvsp_short', 'protein_change', 'amino_acid_change', 'mutation']), None)
            sample_col = next((c for c in mut.columns if c in ['tumor_sample_barcode', 'sample_id', 'sample']), None)
            
            if not hugo_col or not hgvs_col or not sample_col:
                logger.warning(f"Genomic columns missing. Found: {mut.columns.tolist()}")
                return {}

            h3k27m_samples = set(mut[
                (mut[hugo_col].str.upper().isin(['H3-3A', 'H3F3A'])) & 
                (mut[hgvs_col].str.contains('K28M|K27M', na=False, case=False))
            ][sample_col])

            cna = pd.read_csv(f"{self.data_dir}cna.txt", sep='\t', index_col=0)
            cdkn2a_idx = next((idx for idx in cna.index if str(idx).upper() == 'CDKN2A'), None)
            cdkn2a_del_samples = set(cna.columns[(cna.loc[cdkn2a_idx] == -2)]) if cdkn2a_idx else set()
            
            rna = pd.read_csv(f"{self.data_dir}rna_zscores.txt", sep='\t', index_col=0)
            overlap_rna = list(h3k27m_samples.intersection(rna.columns))
            high_exp = rna[overlap_rna].mean(axis=1) if overlap_rna else pd.Series(dtype=float)
            upregulated_genes = set(high_exp[high_exp > 2.0].index) if not high_exp.empty else set()

            overlap = h3k27m_samples.intersection(cdkn2a_del_samples)
            return {
                "h3k27m_count": len(h3k27m_samples),
                "cdkn2a_del_count": len(cdkn2a_del_samples),
                "overlap_count": len(overlap),
                "prevalence": (len(overlap) / len(h3k27m_samples)) if h3k27m_samples else 0,
                "upregulated_genes": upregulated_genes,
                "total_samples": len(cna.columns)
            }
        except Exception as e:
            logger.warning(f"Genomic validation error: {e}")
            return {}

class ProductionPipeline:
    def __init__(self):
        self._data_fetcher = ProductionDataFetcher()
        self._cmap = CMAPQuery()
        self._ppi = PPINetwork()
        self._depmap = DepMapEssentiality()
        self._tissue = TissueExpressionScorer("dipg") 
        self._synergy = SynergyPredictor()
        self._hyp_gen = HypothesisGenerator()
        self._genomic_validator = PedcBioPortalValidator()
        self._stat_validator = StatisticalValidator()

    async def initialize(self, disease: str):
        return True

    async def run(self, disease_name: str, top_k: int = 15) -> Dict:
        disease_data = await self._data_fetcher.fetch_disease_data(disease_name)
        candidates = await self._data_fetcher.fetch_approved_drugs()
        
        if not candidates:
            logger.warning("⚠️ API returned no drugs. Using fallback safety library.")
            candidates = [
                {"name": "ONC201", "targets": ["DRD2", "CLPB"]},
                {"name": "Panobinostat", "targets": ["HDAC1", "HDAC2"]},
                {"name": "Abemaciclib", "targets": ["CDK4", "CDK6"]}
            ]

        genomic_stats = self._genomic_validator.validate_triple_combo_cohort()
        p_val = self._stat_validator.calculate_cooccurrence_p_value(genomic_stats)
        upregulated = genomic_stats.get("upregulated_genes", set())

        # Escape Route Analysis
        for drug in candidates:
            targets = drug.get("targets", [])
            escape_hits = []
            for t in targets:
                neighbors = self._ppi.get_neighbors(t)
                escape_hits.extend([n for n in neighbors if n in upregulated])
            drug["resistance_nodes"] = list(set(escape_hits))
            drug["escape_bypass_score"] = 1.0 if not escape_hits else 0.4

        # Run Massive Omics Streams
        candidates = await self._tissue.score_batch(candidates) 
        candidates = await self._depmap.score_batch(candidates, disease_name) 
        candidates = await self._ppi.score_batch(candidates, disease_data["genes"]) 
        
        # Combine the scores
        for c in candidates:
            t_score = c.get("tissue_expression_score", 0.1)
            d_score = c.get("depmap_score", 0.1)
            p_score = c.get("ppi_score", 0.1)
            e_score = c.get("escape_bypass_score", 0.4)
            
            c["score"] = (t_score * 0.40) + (d_score * 0.30) + (e_score * 0.20) + (p_score * 0.10)

        sorted_candidates = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)

        hypotheses = self._hyp_gen.generate(
            candidates=sorted_candidates[:top_k], 
            cmap_results=[], synergy_combos=[], differential_cmap=[], 
            genomic_stats=genomic_stats, p_value=p_val
        )

        return {"hypotheses": hypotheses, "stats": {"p_value": p_val}}
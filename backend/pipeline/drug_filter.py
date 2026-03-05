import asyncio
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

STREAM_WEIGHTS = {
    "cmap_score": 0.25, "depmap_score": 0.25, "ppi_score": 0.20,
    "bbb_score": 0.15, "tme_score": 0.10, "gcn_score": 0.05,
    "tissue_expression_score": 0.05
}

class ProductionPipeline:
    def __init__(self):
        self._initialized = False
        self._cmap = None
        self._depmap = None
        self._ppi = None
        self._bbb = None
        self._tme = None
        self._tissue = None
        self._synergy = None
        self._hyp_gen = None
        self._calibrator = None
        self._data_fetcher = None
        self._gcn = None
        self._graph_builder = None

    async def initialize(self, disease: str) -> None:
        if self._initialized: return
        
        from .cmap_query import CMAPQueryEngine
        from .depmap_essentiality import DepMapEssentialityEngine
        from .ppi_network import PPINetworkScorer
        from .bbb_filter import BBBFilter
        from .tme_scorer import TMEScorer
        from .tissue_expression import TissueExpressionScorer
        from .gcn_model import DrugDiseaseGCN
        from .graph_builder import ProductionGraphBuilder
        from .synergy_predictor import SynergyPredictor
        from .hypothesis_generator import HypothesisGenerator
        from .trial_outcome_calibrator import TrialOutcomeCalibrator
        from .data_fetcher import DataFetcher

        self._cmap = CMAPQueryEngine()
        self._depmap = DepMapEssentialityEngine()
        self._ppi = PPINetworkScorer()
        self._bbb = BBBFilter(penalise_low=True, hard_exclude_mw=800.0)
        self._tme = TMEScorer(disease=disease)
        self._tissue = TissueExpressionScorer(disease_name=disease)
        self._synergy = SynergyPredictor(disease=disease)
        self._hyp_gen = HypothesisGenerator(disease=disease)
        self._data_fetcher = DataFetcher()
        self._gcn = DrugDiseaseGCN()
        self._graph_builder = ProductionGraphBuilder(disease=disease)
        self._calibrator = TrialOutcomeCalibrator(disease=disease)

        adj = await self._graph_builder.build_adjacency_dict()
        self._gcn.attach_graph(adj)
        self._initialized = True

    async def run(self, disease_name: str, candidates: Optional[List[Dict]] = None, top_k: int = 15) -> Dict:
        await self.initialize(disease_name)
        disease_data = await self._data_fetcher.fetch_disease_data(disease_name)
        disease_genes = disease_data.get("genes", [])
        
        if candidates is None:
            candidates = await self._data_fetcher.fetch_approved_drugs()

        # Execute Evidence Streams
        cmap_results = await self._cmap.query_reversers(disease=disease_name)
        differential = await self._cmap.query_differential_reversers() if "dipg" in disease_name.lower() else []
        
        candidates = await self._depmap.score_batch(candidates, disease_name=disease_name)
        candidates = await self._ppi.score_batch(candidates, disease_genes=disease_genes)
        candidates, _ = self._bbb.filter_and_rank(candidates)
        candidates = self._tme.score_batch(candidates)
        candidates = await self._tissue.score_batch(candidates)

        # Build Graph and Update GCN
        self._graph_builder.build_graph(disease_data, candidates)
        real_adj = await self._graph_builder.build_adjacency_dict()
        self._gcn.attach_graph(real_adj)

        gcn_active = self._gcn._is_trained
        for c in candidates:
            c["gcn_score"] = self._gcn.score_drug(c.get("name"), disease_genes) if gcn_active else None

        # Composite Integration
        for c in candidates:
            s, w = 0.0, 0.0
            for stream, weight in STREAM_WEIGHTS.items():
                val = c.get(stream)
                if val is not None:
                    s += val * weight
                    w += weight
            c["composite_score"] = s / w if w > 0 else 0.0

        candidates.sort(key=lambda x: x.get("composite_score", 0), reverse=True)
        top_candidates = candidates[:top_k]

        # FIXED: Removed 'await' because predict_top_combinations is synchronous
        synergy_combos = self._synergy.predict_top_combinations(top_candidates[:30])

        hypotheses = self._hyp_gen.generate(
            candidates=top_candidates,
            cmap_results=cmap_results,
            synergy_combos=synergy_combos,
            differential_cmap=differential
        )

        return {
            "hypotheses": hypotheses,
            "top_candidates": self._calibrator.predict_batch(top_candidates),
            "pipeline_stats": {"gcn_active": gcn_active}
        }

    async def close(self):
        if self._data_fetcher: await self._data_fetcher.close()
        if self._cmap: await self._cmap.close()
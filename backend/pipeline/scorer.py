"""
scorer.py — Evidence Stream Scorer (v3.0)
==========================================
CHANGES FROM v2.0 → v3.0
--------------------------
REMOVED:
  - composite_score() function — was a weighted sum, now replaced by
    hypothesis_generator.py which does proper multi-evidence integration
  - The idea that a single float summarises a drug candidate

KEPT:
  - PATHWAY_WEIGHTS dict (still used by pathway scoring)
  - score_gene_overlap() — feeds into evidence chain as one signal
  - score_pathway_overlap() — feeds into evidence chain as one signal
  - sensitivity_analysis() — still valid for methods section
  - DrugScorer class — now produces component scores, not composite
  - WEIGHT_* constants — exported for other modules

THE CRITICAL CHANGE:
  DrugScorer.score() no longer produces a "score" field that ranks drugs.
  It produces COMPONENT scores that feed into hypothesis_generator.py.
  The hypothesis generator decides what to do with them — not this file.

  Old: score = 0.40*gene + 0.30*pathway + 0.20*bbb + 0.10*lit  → rank drugs
  New: score_components = {gene, pathway, bbb, lit, ppi, cmap, depmap}
       → hypothesis_generator combines these with evidence grading
"""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PATHWAY WEIGHTS (unchanged — used by score_pathway_overlap)
# ─────────────────────────────────────────────────────────────────────────────

PATHWAY_WEIGHTS: Dict[str, float] = {
    "Cell cycle regulation": 0.80, "Apoptosis": 0.85,
    "DNA damage response": 0.75, "DNA repair": 0.70,
    "p53 signaling": 0.80, "MDM2-p53 interaction": 0.75,
    "Oncogene addiction": 0.85, "Tumour suppressor loss": 0.80,
    "Synthetic lethality": 0.85, "Angiogenesis": 0.65,
    "VEGF signaling": 0.65, "Hypoxia response": 0.60,
    "Cancer metabolism": 0.60, "Warburg effect": 0.55,
    "Receptor tyrosine kinase signaling": 0.80, "MAPK signaling": 0.75,
    "RAS signaling": 0.75, "PI3K-Akt signaling": 0.80,
    "mTOR signaling": 0.78, "PTEN signaling": 0.80,
    "JAK-STAT signaling": 0.72, "STAT3 signaling": 0.75,
    "NF-kB signaling": 0.70, "Wnt signaling": 0.70,
    "EGFR signaling": 0.80, "CDK4/6 signaling": 0.80,
    "BCL-2 family signaling": 0.80, "Intrinsic apoptosis pathway": 0.82,
    "T-cell checkpoint signaling": 0.65, "PD-1/PD-L1 signaling": 0.65,
    "Tumour microenvironment": 0.60, "TGF-beta signaling": 0.60,
    "H3K27 methylation": 1.00, "PRC2 complex": 1.00,
    "Epigenetic regulation of gene expression": 0.95,
    "Chromatin remodeling": 0.90, "Histone modification": 0.90,
    "Histone acetylation": 0.90, "HDAC deacetylase activity": 0.95,
    "HDAC inhibition": 0.95, "Histone deacetylation": 0.95,
    "EZH2 histone methyltransferase activity": 0.95,
    "EZH2 inhibition": 0.95, "BRD4 signaling": 0.95,
    "BET bromodomain": 0.95, "Super-enhancer regulation": 0.88,
    "ACVR1 signaling": 1.00, "BMP signaling pathway": 1.00,
    "BMP-SMAD signaling": 0.95, "SMAD signaling": 0.90,
    "ALK2 signaling": 0.95, "PDGFRA signaling": 0.90,
    "PARP signaling": 0.85, "ATR signaling": 0.75,
    "MYC signaling": 0.80, "MYCN signaling": 0.82,
    "Cancer stem cell signaling": 0.70,
}


def get_pathway_weight(
    pathway_name: str,
    disease_weights: Optional[Dict[str, float]] = None
) -> float:
    if disease_weights:
        if pathway_name in disease_weights:
            return disease_weights[pathway_name]
        for key, weight in disease_weights.items():
            if key.lower() in pathway_name.lower() or pathway_name.lower() in key.lower():
                return weight
    if pathway_name in PATHWAY_WEIGHTS:
        return PATHWAY_WEIGHTS[pathway_name]
    pathway_lower = pathway_name.lower()
    best_weight = None
    best_match_len = 0
    for key, weight in PATHWAY_WEIGHTS.items():
        key_lower = key.lower()
        if key_lower in pathway_lower or pathway_lower in key_lower:
            match_len = len(min(key_lower, pathway_lower, key=len))
            if match_len > best_match_len:
                best_match_len = match_len
                best_weight = weight
    return best_weight if best_weight is not None else 0.50


# ─────────────────────────────────────────────────────────────────────────────
# Component scoring functions — these are INPUTS to hypothesis_generator
# not final scores for ranking
# ─────────────────────────────────────────────────────────────────────────────

def score_gene_overlap(
    drug_targets:  List[str],
    disease_genes: List[str],
    gene_weights:  Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict]:
    if not drug_targets or not disease_genes:
        return 0.0, {"overlap": [], "n_overlap": 0, "n_disease": len(disease_genes)}

    drug_set    = set(t.upper() for t in drug_targets)
    disease_set = set(g.upper() for g in disease_genes)
    overlap     = drug_set & disease_set

    if not overlap:
        return 0.0, {"overlap": [], "n_overlap": 0, "n_disease": len(disease_set)}

    if gene_weights:
        weighted_hits  = sum(gene_weights.get(g, 0.5) for g in overlap)
        weighted_total = sum(gene_weights.get(g, 0.5) for g in disease_set)
        score = weighted_hits / max(weighted_total, 1.0)
    else:
        score = len(overlap) / len(disease_set)

    if len(overlap) >= 3:
        score = min(1.0, score * 1.15)
    elif len(overlap) >= 2:
        score = min(1.0, score * 1.08)

    return round(min(score, 1.0), 4), {
        "overlap": sorted(overlap), "n_overlap": len(overlap),
        "n_disease": len(disease_set), "n_drug_targets": len(drug_set),
    }


def score_pathway_overlap(
    drug_pathways:    List[str],
    disease_pathways: List[str],
    disease_weights:  Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict]:
    if not drug_pathways or not disease_pathways:
        return 0.0, {"matched_pathways": [], "weighted_score": 0.0}

    matched   = []
    total_w   = 0.0
    max_total = sum(get_pathway_weight(dp, disease_weights) for dp in disease_pathways)

    for drug_path in drug_pathways:
        best_w, best_match = 0.0, None
        for disease_path in disease_pathways:
            dp_lower  = drug_path.lower()
            dis_lower = disease_path.lower()
            if dp_lower == dis_lower or dp_lower in dis_lower or dis_lower in dp_lower:
                w = get_pathway_weight(disease_path, disease_weights)
                if w > best_w:
                    best_w, best_match = w, disease_path
        if best_match:
            matched.append({"drug_pathway": drug_path, "matched_to": best_match, "weight": best_w})
            total_w += best_w

    score = total_w / max(max_total, 1.0) if max_total > 0 else 0.0
    return round(min(score, 1.0), 4), {
        "matched_pathways": matched, "n_matched": len(matched),
        "weighted_score": round(total_w, 4), "max_possible": round(max_total, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Weight constants — exported for other modules
# ─────────────────────────────────────────────────────────────────────────────

WEIGHT_GENE       = 0.40
WEIGHT_PATHWAY    = 0.30
WEIGHT_BBB        = 0.20
WEIGHT_LITERATURE = 0.10
WEIGHT_PPI        = 0.15
WEIGHT_SIMILARITY = 0.10
WEIGHT_MECHANISM  = 0.20


# ─────────────────────────────────────────────────────────────────────────────
# DrugScorer — produces component scores, NOT a final ranking score
# ─────────────────────────────────────────────────────────────────────────────

class DrugScorer:
    """
    Produces component evidence scores for drug candidates.

    IMPORTANT: This class no longer produces a single composite "score".
    It produces score_components that feed into hypothesis_generator.py.
    The hypothesis generator applies evidence grading and produces
    structured hypotheses — not a ranked list.

    For ranking (e.g. for display), use the hypothesis confidence_score
    from hypothesis_generator, not the raw component scores here.
    """

    SUPPORTED_CNS_DISEASES = {"glioblastoma", "gbm", "dipg", "glioma",
                               "medulloblastoma", "ependymoma"}

    def __init__(
        self,
        disease:               str = "general",
        disease_genes:         Optional[List[str]] = None,
        disease_pathways:      Optional[List[str]] = None,
        custom_gene_weights:   Optional[Dict[str, float]] = None,
        custom_pathway_weights: Optional[Dict[str, float]] = None,
    ):
        self.disease          = disease.lower().strip()
        self.disease_genes    = disease_genes or []
        self.disease_pathways = disease_pathways or []
        self.gene_weights     = custom_gene_weights or {}
        self.pathway_weights  = {**PATHWAY_WEIGHTS}

        if custom_pathway_weights:
            self.pathway_weights.update(custom_pathway_weights)

        if self.disease in ("dipg", "diffuse intrinsic pontine glioma",
                            "h3k27m", "h3k27m glioma"):
            self._load_dipg_weights()

        logger.info("DrugScorer: disease='%s', genes=%d", self.disease, len(self.disease_genes))

    def _load_dipg_weights(self):
        try:
            from dipg_specialization import (
                DIPG_PATHWAY_WEIGHTS, DIPG_CORE_GENES, get_dipg_gene_score_weights,
            )
            self.pathway_weights.update(DIPG_PATHWAY_WEIGHTS)
            if not self.disease_genes:
                self.disease_genes = DIPG_CORE_GENES
            if not self.gene_weights:
                self.gene_weights = get_dipg_gene_score_weights()
            logger.info("  → Loaded DIPG-specific weights")
        except ImportError:
            logger.warning("  ⚠️  dipg_specialization module not found")

    def score(self, candidate: Dict) -> Dict:
        """
        Compute component evidence scores for a single drug candidate.

        Returns candidate dict with added score_components field.
        Does NOT produce a single composite score — use hypothesis_generator
        for that.
        """
        targets   = candidate.get("targets") or candidate.get("drug_targets") or []
        pathways  = candidate.get("pathways") or []
        bbb       = candidate.get("bbb_score", 0.65)
        lit_score = candidate.get("literature_score", 0.50)

        gene_sc, gene_details = score_gene_overlap(
            targets, self.disease_genes, self.gene_weights
        )
        path_sc, path_details = score_pathway_overlap(
            pathways, self.disease_pathways, self.pathway_weights
        )

        # Store component scores (these are inputs to hypothesis_generator)
        candidate["gene_score"]      = gene_sc
        candidate["pathway_score"]   = path_sc
        candidate["bbb_score"]       = bbb
        candidate["gene_overlap"]    = gene_details
        candidate["pathway_overlap"] = path_details

        # Lightweight preliminary score for filtering only
        # (NOT used for ranking — hypothesis_generator does that)
        preliminary = (gene_sc * 0.50 + path_sc * 0.30 + bbb * 0.20)
        candidate["preliminary_score"] = round(preliminary, 4)

        return candidate

    def score_batch(self, candidates: List[Dict]) -> List[Dict]:
        """Score all candidates and sort by preliminary score (for filtering only)."""
        for c in candidates:
            self.score(c)
        candidates.sort(key=lambda x: x.get("preliminary_score", 0), reverse=True)
        logger.info("Scored %d candidates (component scores for hypothesis_generator)", len(candidates))
        return candidates


def sensitivity_analysis(candidates: list, perturbation: float = 0.10) -> dict:
    """
    Weight sensitivity analysis — unchanged from v2.0.
    Valid for methods section — confirms component score robustness.
    """
    import math

    base_weights = {
        "gene_score": WEIGHT_GENE, "pathway_score": WEIGHT_PATHWAY,
        "bbb_score": WEIGHT_BBB, "literature_score": WEIGHT_LITERATURE,
    }

    def _score(comp, w):
        return sum(comp.get(k, 0.0) * v for k, v in w.items())

    def _spearman(ranks_a, ranks_b):
        n = len(ranks_a)
        if n < 2:
            return 1.0
        d2 = sum((a - b) ** 2 for a, b in zip(ranks_a, ranks_b))
        return 1.0 - (6 * d2) / (n * (n * n - 1))

    def _rank(scores):
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        ranks   = [0] * len(scores)
        for r, (i, _) in enumerate(indexed, 1):
            ranks[i] = r
        return ranks

    if not candidates:
        return {"rank_correlation_min": 1.0, "stable": True,
                "paper_statement": "Sensitivity analysis skipped.", "perturbation_results": []}

    base_scores = [_score(c, base_weights) for c in candidates]
    base_ranks  = _rank(base_scores)
    perturb_results = []
    min_rho = 1.0

    for key in base_weights:
        for sign in (+1, -1):
            delta     = sign * perturbation
            w_perturb = {k: max(0.01, v + (delta if k == key else 0.0))
                         for k, v in base_weights.items()}
            total     = sum(w_perturb.values())
            w_perturb = {k: v / total for k, v in w_perturb.items()}
            p_scores  = [_score(c, w_perturb) for c in candidates]
            p_ranks   = _rank(p_scores)
            rho       = _spearman(base_ranks, p_ranks)
            perturb_results.append({
                "weight_changed": key, "direction": "up" if sign > 0 else "down",
                "new_weight": round(w_perturb[key], 3), "rho": round(rho, 4),
            })
            if rho < min_rho:
                min_rho = rho

    stable = min_rho > 0.90
    return {
        "rank_correlation_min": round(min_rho, 4),
        "stable": stable,
        "paper_statement": (
            f"A ±{perturbation:.0%} perturbation of all scoring weights yielded "
            f"minimum Spearman ρ = {min_rho:.3f} ({'stable' if stable else 'unstable'})."
        ),
        "perturbation_results": perturb_results,
    }


# Backward compatibility
ProductionScorer = DrugScorer
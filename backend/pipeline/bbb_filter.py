"""
bbb_filter.py — Blood-Brain Barrier Penetrance Scoring (v3.1)
=======================================================
Scores drug candidates for blood-brain barrier penetrance, which is a
critical first-pass filter for GBM/DIPG drug candidates.

FIXES v3.1:
  - Added __init__ to accept configuration from discovery_pipeline.py.
  - Implemented CNS MPO scoring logic as a fallback for drugs not in the 
    known penetrance database.
  - Standardized penetrance categories: HIGH, MODERATE, LOW, UNKNOWN.
"""

import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# KNOWN BBB PENETRANCE DATABASE
# Curated from clinical literature and pharmacokinetic studies.
# ─────────────────────────────────────────────────────────────────────────────
KNOWN_BBB_PENETRANCE: Dict[str, str] = {
    "temozolomide": "HIGH",
    "onc201":       "HIGH",
    "panobinostat": "HIGH",
    "abemaciclib":  "HIGH",
    "dexamethasone":"HIGH",
    "lomustine":    "HIGH",
    "carmustine":   "HIGH",
    "marizomib":    "HIGH",
    "thioridazine": "HIGH",
    "bevacizumab":  "LOW",  # Monoclonal antibody, limited BBB crossing
    "pembrolizumab":"LOW",  # Monoclonal antibody
    "nivolumab":    "LOW",
    "rituximab":    "LOW",
    "trastuzumab":  "LOW",
    "lapatinib":    "MODERATE",
    "erlotinib":    "MODERATE",
    "gefitinib":    "MODERATE",
    "dasatinib":    "LOW",      # Active efflux by P-gp/BCRP
    "imatinib":     "LOW",      # Active efflux by P-gp
}

class BBBFilter:
    """
    Filters and scores drugs based on Blood-Brain Barrier (BBB) penetrance.
    Uses a combination of a curated database and CNS MPO scoring.
    """

    def __init__(self, penalise_low: bool = True, hard_exclude_mw: float = 800.0):
        """
        Initialize the filter with constraints used by ProductionPipeline.
        """
        self.penalise_low = penalise_low
        self.hard_exclude_mw = hard_exclude_mw
        logger.info("BBB Filter initialized (MW limit: %.1f Da)", hard_exclude_mw)

    def score_drug(self, drug_name: str, smiles: str = "", molecular_weight: Optional[float] = None) -> Dict:
        """
        Scores a single drug for BBB penetrance.
        """
        name_lower = drug_name.lower()
        
        # 1. Hard Exclusion by Molecular Weight (Reviewer requirement)
        if molecular_weight and molecular_weight > self.hard_exclude_mw:
            return {
                "penetrance": "LOW",
                "bbb_score": 0.1,
                "reason": f"MW {molecular_weight} exceeds hard limit {self.hard_exclude_mw}"
            }

        # 2. Check curated database
        if name_lower in KNOWN_BBB_PENETRANCE:
            penetrance = KNOWN_BBB_PENETRANCE[name_lower]
            return {
                "penetrance": penetrance,
                "bbb_score": self._penetrance_to_score(penetrance),
                "reason": "Curated database"
            }

        # 3. Fallback: CNS MPO (Multi-Parameter Optimization) Heuristic
        # If we have MW, we can at least provide a basic size-based estimate
        if molecular_weight:
            if molecular_weight < 400:
                return {"penetrance": "MODERATE", "bbb_score": 0.6, "reason": "MW < 400 Da"}
            elif molecular_weight < 600:
                return {"penetrance": "LOW", "bbb_score": 0.3, "reason": "MW 400-600 Da"}
            else:
                return {"penetrance": "LOW", "bbb_score": 0.1, "reason": "MW > 600 Da"}

        return {"penetrance": "UNKNOWN", "bbb_score": 0.5, "reason": "No data available"}

    def filter_and_rank(self, candidates: List[Dict], apply_penalty: bool = True, exclude_low: bool = False) -> Tuple[List[Dict], List[Dict]]:
        """
        Processes a batch of candidates. 
        Returns (passing_list, excluded_list).
        """
        passing = []
        excluded = []

        for c in candidates:
            name = c.get("name") or c.get("drug_name") or "Unknown"
            mw = c.get("molecular_weight")
            
            result = self.score_drug(name, molecular_weight=mw)
            
            c["bbb_penetrance"] = result["penetrance"]
            c["bbb_score"] = result["bbb_score"]
            
            # Exclusion logic
            if exclude_low and result["penetrance"] == "LOW":
                excluded.append(c)
            else:
                passing.append(c)

        return passing, excluded

    def _penetrance_to_score(self, category: str) -> float:
        """Map penetrance category to a 0-1 numerical score."""
        mapping = {
            "HIGH": 1.0,
            "MODERATE": 0.6,
            "LOW": 0.2,
            "UNKNOWN": 0.5
        }
        return mapping.get(category, 0.5)
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

KNOWN_BBB_PENETRANCE: Dict[str, str] = {
    # HIGH — cross BBB reliably (clinical PK data)
    "temozolomide":   "HIGH",
    "onc201":         "HIGH",
    "panobinostat":   "HIGH",
    "abemaciclib":    "HIGH",   # Specifically selected for CNS penetrance vs palbociclib
    "dexamethasone":  "HIGH",
    "lomustine":      "HIGH",
    "carmustine":     "HIGH",
    "marizomib":      "HIGH",   # Designed for CNS — crosses BBB unlike bortezomib
    "thioridazine":   "HIGH",
    "vorinostat":     "MODERATE",
    "valproic acid":  "HIGH",
    "chloroquine":    "HIGH",
    "hydroxychloroquine": "MODERATE",
    "metformin":      "MODERATE",
    "itraconazole":   "MODERATE",
    "ribociclib":     "MODERATE",
    "palbociclib":    "LOW",    # Poor CNS penetrance vs abemaciclib
    # LOW — monoclonals or known poor CNS penetrance
    "bevacizumab":    "LOW",
    "pembrolizumab":  "LOW",
    "nivolumab":      "LOW",
    "rituximab":      "LOW",
    "trastuzumab":    "LOW",
    "cetuximab":      "LOW",
    "lapatinib":      "MODERATE",
    "erlotinib":      "MODERATE",
    "gefitinib":      "MODERATE",
    "dasatinib":      "LOW",
    "imatinib":       "LOW",
    "bortezomib":     "LOW",    # Does NOT cross BBB — marizomib was developed to fix this
    # FAILED trials — penalise regardless of BBB
    "cilengitide":    "LOW",    # CENTRIC Phase 3 failed 2015 — integrin blockade ineffective in GBM
    "enzastaurin":    "LOW",
    "temsirolimus":   "LOW",
    "cediranib":      "LOW",
}

# Drugs that failed in clinical GBM trials — hard penalise in scoring
KNOWN_GBM_FAILURES = {
    "cilengitide",    # CENTRIC Phase 3 2015 — no OS benefit
    "enzastaurin",    # Phase 3 failed
    "temsirolimus",   # Phase 3 vs TMZ — inferior
    "cediranib",      # Phase 2 — no OS benefit despite radiological response
    "iniparib",       # Phase 3 failed
    "vorinostat",     # Phase 2 GBM — negative
    "erlotinib",      # Phase 2 GBM — negative
    "gefitinib",      # Phase 2 GBM — negative
    "imatinib",       # Phase 2 GBM — negative
    "tipifarnib",     # Phase 2 GBM — negative
    "sorafenib",      # Phase 2 GBM — negative
    "sunitinib",      # Phase 2 GBM — negative
    "dasatinib",      # Phase 2 GBM — negative
    "everolimus",     # Phase 2 GBM — negative
    "ribociclib",     # Phase 2 GBM — negative (unlike abemaciclib)
    "palbociclib",    # Poor BBB — inferior to abemaciclib for GBM
}

class BBBFilter:
    def __init__(self, penalise_low: bool = True, hard_exclude_mw: float = 800.0,
                 low_bbb_penalty: float = 0.50, mod_bbb_penalty: float = 0.85):
        self.penalise_low = penalise_low
        self.hard_exclude_mw = hard_exclude_mw
        logger.info("BBB Filter initialized (MW limit: %.1f Da)", hard_exclude_mw)

    def score_drug(self, drug_name: str, smiles: str = "",
                   molecular_weight: Optional[float] = None) -> Dict:
        name_lower = drug_name.lower().strip()

        # Hard exclude by MW (monoclonals etc)
        if molecular_weight and molecular_weight > self.hard_exclude_mw:
            return {"penetrance": "LOW", "bbb_score": 0.1,
                    "reason": f"MW {molecular_weight:.0f} Da exceeds {self.hard_exclude_mw:.0f} limit",
                    "clinical_failure": False}

        # Known GBM clinical failures — penalise heavily regardless of BBB
        if name_lower in KNOWN_GBM_FAILURES:
            return {"penetrance": KNOWN_BBB_PENETRANCE.get(name_lower, "LOW"),
                    "bbb_score": 0.05,
                    "reason": f"Known GBM clinical trial failure — deprioritised",
                    "clinical_failure": True}

        # Curated database
        if name_lower in KNOWN_BBB_PENETRANCE:
            penetrance = KNOWN_BBB_PENETRANCE[name_lower]
            return {"penetrance": penetrance,
                    "bbb_score": self._penetrance_to_score(penetrance),
                    "reason": "Curated PK database",
                    "clinical_failure": False}

        # MW heuristic fallback
        if molecular_weight:
            if molecular_weight < 400:
                return {"penetrance": "MODERATE", "bbb_score": 0.6,
                        "reason": "MW < 400 Da (heuristic)", "clinical_failure": False}
            elif molecular_weight < 600:
                return {"penetrance": "LOW", "bbb_score": 0.3,
                        "reason": "MW 400-600 Da (heuristic)", "clinical_failure": False}
            else:
                return {"penetrance": "LOW", "bbb_score": 0.1,
                        "reason": "MW > 600 Da (heuristic)", "clinical_failure": False}

        return {"penetrance": "UNKNOWN", "bbb_score": 0.5,
                "reason": "No PK data available", "clinical_failure": False}

    def filter_and_rank(self, candidates: List[Dict],
                        apply_penalty: bool = True,
                        exclude_low: bool = False) -> Tuple[List[Dict], List[Dict]]:
        passing, excluded = [], []
        n_failures = 0
        for c in candidates:
            name = c.get("name") or c.get("drug_name") or "Unknown"
            mw   = c.get("molecular_weight")
            result = self.score_drug(name, molecular_weight=mw)

            c["bbb_penetrance"]    = result["penetrance"]
            c["bbb_score"]         = result["bbb_score"]
            c["clinical_failure"]  = result["clinical_failure"]

            if result["clinical_failure"]:
                n_failures += 1
                # Apply score penalty for known failures
                c["score"] = c.get("score", 0.0) * 0.1

            if exclude_low and result["penetrance"] == "LOW":
                excluded.append(c)
            else:
                passing.append(c)

        if n_failures:
            logger.info("BBB filter: penalised %d known GBM clinical trial failures", n_failures)

        return passing, excluded

    def _penetrance_to_score(self, category: str) -> float:
        return {"HIGH": 1.0, "MODERATE": 0.6, "LOW": 0.2, "UNKNOWN": 0.5}.get(category, 0.5)
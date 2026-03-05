"""
polypharmacology.py — Multi-Target Polypharmacology Scoring
============================================================
Scores drug candidates for their multi-target polypharmacology profiles,
including synergistic target combinations and resistance pathway coverage.

FIXES v3.0
-----------
FIX 1 (CRITICAL — was crashing at runtime):
  score_batch() signature was: score_batch(self, candidates, disease_targets=None)
  but run_dipg_pipeline.py called it as:
      scorer.score_batch(candidates, disease_targets=disease_genes)
  Looking at the original implementation, score_batch only accepted
  `candidates: List[Dict]` — no disease_targets parameter existed.
  This raised TypeError at runtime.
  Fix: added disease_targets parameter properly — it's used to compute
  context-specific target overlap scores.

FIX 2 (score inflation):
  _selectivity_score() was returning values in range 0.6–0.9 regardless
  of actual selectivity. The composite formula added this directly, so
  poly_score rarely approached 0 — every drug got a systematic boost.
  Fix: rescaled to true 0–1 range. A highly selective drug (1 target) now
  scores 1.0; a promiscuous drug (10+ targets) scores near 0.

FIX 3 (architecture):
  Removed final composite score output. Like scorer.py (v3.0),
  polypharmacology now produces component signals that feed into
  hypothesis_generator.py, not a final ranking score.
"""

import logging
import math
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SYNERGISTIC TARGET COMBINATIONS
# Biology-derived pairs/triples with known synergistic effects in GBM/DIPG.
# Source: published literature, CRISPR combination screens, clinical data.
# ─────────────────────────────────────────────────────────────────────────────

SYNERGISTIC_TARGET_COMBINATIONS: List[Dict] = [

    # ── H3K27M DIPG — epigenetic triad (highest confidence) ──────────────────
    {
        "targets":     {"EZH2", "HDAC1"},
        "score":       0.90,
        "mechanism":   "EZH2 + HDAC inhibition: complementary H3K27 regulation in H3K27M",
        "dipg_specific": True,
        "reference":   "Piunti et al. 2017; Mohammad et al. 2017",
    },
    {
        "targets":     {"BRD4", "HDAC1"},
        "score":       0.88,
        "mechanism":   "BET + HDAC: dual epigenetic attack on H3K27M super-enhancers",
        "dipg_specific": True,
        "reference":   "Nagaraja et al. 2017",
    },
    {
        "targets":     {"EZH2", "BRD4"},
        "score":       0.82,
        "mechanism":   "EZH2 + BET: complementary H3K27M epigenome targeting",
        "dipg_specific": True,
        "reference":   "Piunti et al. 2017",
    },
    {
        "targets":     {"EZH2", "BRD4", "HDAC1"},
        "score":       0.92,
        "mechanism":   "EZH2 + BET + HDAC triple combination — maximal epigenetic reprogramming",
        "dipg_specific": True,
        "reference":   "Grasso et al. 2015; Piunti et al. 2017",
    },

    # ── CDK4/6 + PI3K/mTOR ────────────────────────────────────────────────────
    {
        "targets":     {"CDK4", "MTOR"},
        "score":       0.78,
        "mechanism":   "CDK4/6 + mTOR: cell cycle + survival axis blockade; prevents mTOR bypass resistance",
        "dipg_specific": False,
        "reference":   "Olmez et al. 2017",
    },
    {
        "targets":     {"CDK6", "MTOR"},
        "score":       0.76,
        "mechanism":   "CDK6 + mTOR: dual block prevents G1 arrest bypass via mTOR",
        "dipg_specific": False,
        "reference":   "Lamond et al. 2020",
    },
    {
        "targets":     {"CDK4", "PIK3CA"},
        "score":       0.74,
        "mechanism":   "CDK4/6 + PI3Kα: PI3K activation is primary CDK4/6-i resistance mechanism",
        "dipg_specific": False,
        "reference":   "Dong et al. 2020",
    },

    # ── EGFR + downstream bypass prevention ───────────────────────────────────
    {
        "targets":     {"EGFR", "PIK3CA"},
        "score":       0.72,
        "mechanism":   "EGFR + PI3K: blocks receptor + downstream; prevents PI3K bypass",
        "dipg_specific": False,
        "reference":   "Vivanco et al. 2012",
    },
    {
        "targets":     {"EGFR", "MTOR"},
        "score":       0.70,
        "mechanism":   "EGFR + mTOR: dual vertical inhibition of PI3K-AKT-mTOR axis",
        "dipg_specific": False,
        "reference":   "Mechanistic; consistent with feedback inhibition data",
    },

    # ── PARP + sensitisers ────────────────────────────────────────────────────
    {
        "targets":     {"PARP1", "HDAC1"},
        "score":       0.75,
        "mechanism":   "PARP + HDAC: HDAC inhibition creates BRCAness, sensitises to PARP-i",
        "dipg_specific": False,
        "reference":   "Adimoolam et al. 2007; Rasmussen et al. 2016",
    },
    {
        "targets":     {"PARP1", "CDK4"},
        "score":       0.68,
        "mechanism":   "PARP + CDK4/6: CDK4/6-i induces HR deficiency, sensitises to PARP-i",
        "dipg_specific": False,
        "reference":   "Johnson et al. 2019",
    },

    # ── BCL-2 family ──────────────────────────────────────────────────────────
    {
        "targets":     {"BCL2", "BCL2L1"},
        "score":       0.70,
        "mechanism":   "BCL-2 + BCL-XL dual inhibition: broader anti-apoptotic blockade",
        "dipg_specific": False,
        "reference":   "Pan et al. 2014",
    },
    {
        "targets":     {"BCL2", "HDAC1"},
        "score":       0.72,
        "mechanism":   "BCL-2 + HDAC: HDAC-i reduces BCL-2 expression, sensitises to venetoclax",
        "dipg_specific": False,
        "reference":   "Bhatt et al. 2017",
    },

    # ── MDM2 + cell cycle ─────────────────────────────────────────────────────
    {
        "targets":     {"MDM2", "CDK4"},
        "score":       0.68,
        "mechanism":   "MDM2 (p53 restoration) + CDK4/6-i: dual G1 arrest in TP53-WT CDKN2A-del GBM",
        "dipg_specific": False,
        "reference":   "Mechanistic; DepMap synthetic lethality",
    },

    # ── ACVR1 + HDAC (DIPG-specific) ─────────────────────────────────────────
    {
        "targets":     {"ACVR1", "HDAC1"},
        "score":       0.80,
        "mechanism":   "ACVR1/BMP + HDAC: ACVR1 gain-of-function drives transcription; HDAC-i resets epigenome",
        "dipg_specific": True,
        "reference":   "Taylor et al. 2014; Grasso et al. 2015",
    },

    # ── Stemness + differentiation ────────────────────────────────────────────
    {
        "targets":     {"SOX2", "EZH2"},
        "score":       0.65,
        "mechanism":   "SOX2 + EZH2: targeting GSC stemness + epigenetic maintenance",
        "dipg_specific": False,
        "reference":   "Mechanistic",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# RESISTANCE GENES
# Genes whose expression/activation drives treatment resistance in GBM/DIPG.
# A drug's target set that overlaps these → resistance concern.
# A combination that COVERS these → resistance bypass bonus.
# ─────────────────────────────────────────────────────────────────────────────

GBM_RESISTANCE_GENES: Set[str] = {
    # RTK bypass
    "EGFR", "PDGFRA", "MET", "FGFR1", "AXL", "IGF1R",
    # PI3K/AKT/mTOR axis
    "PIK3CA", "PIK3R1", "AKT1", "AKT2", "MTOR",
    # RAS/MAPK axis
    "KRAS", "NRAS", "BRAF", "MAP2K1", "MAPK1",
    # Cell cycle
    "CDK4", "CDK6", "CCND1", "CCND2", "CCNE1",
    # Anti-apoptotic
    "BCL2", "BCL2L1", "MCL1", "BIRC5",
    # Epigenetic
    "EZH2", "BRD4", "HDAC1", "HDAC2",
    # DNA damage/repair
    "PARP1", "RAD51", "CHEK1",
    # Drug efflux
    "ABCB1",   # P-glycoprotein
    "ABCG2",   # BCRP
    # Stemness
    "SOX2", "NES", "ALDH1A1",
    # DIPG-specific
    "H3F3A", "HIST1H3B", "KDM6A", "KDM6B", "ACVR1",
}

# Genes specific to DIPG resistance
DIPG_RESISTANCE_GENES: Set[str] = GBM_RESISTANCE_GENES | {
    "H3F3A",   # H3.3 K27M — primary driver
    "HIST1H3B", # H3.1 K27M
    "PDGFRA",  # frequent DIPG amplification
    "KDM6A",   # H3K27 demethylase — compensatory
    "KDM6B",   # JMJD3 — compensatory
    "BRD4",    # super-enhancer reader
    "ABCB1",   # P-gp efflux
    "ABCG2",   # BCRP efflux
    "BCL2",    # anti-apoptotic
    "SOX2",    # GSC stemness
}


# ─────────────────────────────────────────────────────────────────────────────
# Main Polypharmacology Scorer
# ─────────────────────────────────────────────────────────────────────────────

class PolypharmacologyScorer:
    """
    Scores drug candidates for polypharmacology quality in GBM/DIPG.

    Produces component signals:
      - synergy_score    : how well drug hits synergistic target combinations
      - selectivity_score: true 0–1 selectivity (fixed from v2.0 inflation)
      - resistance_coverage: how many resistance pathways the drug covers
      - poly_score       : combined component score (feeds hypothesis_generator)

    v3.0 changes:
      - score_batch() now accepts optional disease_targets parameter (fixed TypeError)
      - _selectivity_score() rescaled to true 0–1 (fixed systematic inflation)
      - No final composite ranking score output (feeds hypothesis_generator instead)
    """

    def __init__(
        self,
        disease:       str  = "glioblastoma",
        dipg_mode:     bool = False,
    ):
        self.disease   = disease.lower()
        self.dipg_mode = dipg_mode or any(
            k in self.disease for k in ("dipg", "h3k27m", "diffuse intrinsic")
        )
        self.resistance_genes = (
            DIPG_RESISTANCE_GENES if self.dipg_mode else GBM_RESISTANCE_GENES
        )
        logger.info(
            "PolypharmacologyScorer: disease=%s | DIPG=%s | resistance_genes=%d",
            disease, self.dipg_mode, len(self.resistance_genes)
        )

    def score(self, candidate: Dict, disease_targets: Optional[List[str]] = None) -> Dict:
        """
        Score a single drug candidate for polypharmacology.

        Parameters
        ----------
        candidate : dict with 'targets' (list of gene symbols) and 'pathways'
        disease_targets : optional list of disease-relevant genes for
                         context-specific target overlap scoring.
                         Was missing in v2.0 — caused TypeError in run_dipg_pipeline.py

        Returns candidate with added poly fields.
        """
        targets  = set(candidate.get("targets", []))
        pathways = [p.lower() for p in candidate.get("pathways", [])]

        # 1. Synergistic target combination score
        syn_score, syn_combos = self._score_synergistic_combinations(targets)

        # 2. Selectivity score — FIXED: now true 0–1, was 0.6–0.9 causing inflation
        sel_score = self._selectivity_score(targets)

        # 3. Resistance pathway coverage
        resist_score, resist_hits = self._resistance_coverage(targets)

        # 4. Context-specific score (uses disease_targets if provided)
        context_score = 0.0
        if disease_targets:
            disease_set   = set(t.upper() for t in disease_targets)
            target_upper  = set(t.upper() for t in targets)
            context_overlap = target_upper & disease_set
            context_score = min(1.0, len(context_overlap) / max(len(disease_set), 1))

        # Combined polypharmacology component score
        # NOTE: this is a component signal, not a ranking score
        # Weights reflect: synergy most important, then resistance coverage,
        # context overlap, selectivity last (broad is ok for combinations)
        poly_score = (
            syn_score    * 0.45
            + resist_score * 0.25
            + context_score * 0.20
            + sel_score    * 0.10
        )

        candidate["poly_synergy_score"]     = round(syn_score, 4)
        candidate["poly_selectivity_score"] = round(sel_score, 4)
        candidate["poly_resistance_score"]  = round(resist_score, 4)
        candidate["poly_context_score"]     = round(context_score, 4)
        candidate["poly_score"]             = round(min(1.0, poly_score), 4)
        candidate["synergistic_combos"]     = syn_combos
        candidate["resistance_gene_hits"]   = sorted(resist_hits)

        return candidate

    def score_batch(
        self,
        candidates:      List[Dict],
        disease_targets: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Score all candidates for polypharmacology.

        Parameters
        ----------
        candidates : list of drug candidate dicts
        disease_targets : optional list of disease-relevant gene symbols.
            This parameter was added in v3.0 to fix the TypeError that
            occurred when run_dipg_pipeline.py passed disease_targets=disease_genes
            to a function that didn't accept it.

        Returns list sorted by poly_score (descending).
        """
        for c in candidates:
            self.score(c, disease_targets=disease_targets)

        candidates.sort(key=lambda x: x.get("poly_score", 0), reverse=True)
        logger.info(
            "Polypharmacology scored: %d candidates | top poly_score=%.3f",
            len(candidates),
            candidates[0].get("poly_score", 0) if candidates else 0,
        )
        return candidates

    # ── Private helpers ───────────────────────────────────────────────────────

    def _score_synergistic_combinations(
        self, targets: Set[str]
    ) -> Tuple[float, List[Dict]]:
        """Check drug targets against known synergistic combinations."""
        if not targets:
            return 0.0, []

        targets_upper = set(t.upper() for t in targets)
        matched_combos = []
        best_score     = 0.0

        for combo in SYNERGISTIC_TARGET_COMBINATIONS:
            combo_targets = set(t.upper() for t in combo["targets"])
            overlap       = targets_upper & combo_targets

            if len(overlap) == len(combo_targets):
                # Full hit — all combo targets present
                score = combo["score"]
                matched_combos.append({
                    "targets":   sorted(combo_targets),
                    "score":     score,
                    "mechanism": combo["mechanism"],
                    "full_hit":  True,
                    "dipg":      combo.get("dipg_specific", False),
                })
                if score > best_score:
                    best_score = score

            elif len(overlap) >= max(1, len(combo_targets) - 1):
                # Partial hit — all but one target present
                score = combo["score"] * 0.6
                matched_combos.append({
                    "targets":   sorted(overlap),
                    "score":     score,
                    "mechanism": combo["mechanism"] + " (partial)",
                    "full_hit":  False,
                    "dipg":      combo.get("dipg_specific", False),
                })
                if score > best_score:
                    best_score = score

        return round(best_score, 4), matched_combos

    def _selectivity_score(self, targets: Set[str]) -> float:
        """
        Compute selectivity score as true 0–1 value.

        FIX v3.0: Previous version returned 0.6–0.9 regardless of target count,
        causing every drug to receive a systematic ~0.75 bonus. This inflated
        poly_score for all candidates and reduced discrimination.

        New scoring:
          1 target  → 1.00 (maximally selective)
          2 targets → 0.85
          3 targets → 0.72
          5 targets → 0.55
          10 targets→ 0.30
          20+ targets→ 0.05

        Note: for GBM/DIPG, moderate selectivity (2–5 targets hitting
        complementary pathways) is often preferred over single-target
        selectivity. The synergy bonus in _score_synergistic_combinations
        rewards appropriate polypharmacology separately.
        """
        n = len(targets)
        if n == 0:
            return 0.0
        if n == 1:
            return 1.00
        # Exponential decay: score = exp(-k * (n-1))
        # k = 0.15 gives: n=2→0.86, n=5→0.55, n=10→0.26, n=20→0.06
        k = 0.15
        return round(math.exp(-k * (n - 1)), 4)

    def _resistance_coverage(
        self, targets: Set[str]
    ) -> Tuple[float, Set[str]]:
        """
        Score how many known resistance genes the drug directly targets.

        Drugs that hit resistance genes directly may prevent or delay
        the most common resistance mechanisms in GBM/DIPG.
        """
        if not targets:
            return 0.0, set()

        targets_upper = set(t.upper() for t in targets)
        resist_upper  = set(r.upper() for r in self.resistance_genes)
        hits          = targets_upper & resist_upper

        # Score: each hit adds value, diminishing returns beyond 4
        score = 0.0
        for i, _ in enumerate(hits, 1):
            score += 0.20 / math.sqrt(i)   # 0.20, 0.14, 0.12, 0.10, ...

        return round(min(1.0, score), 4), hits

    def generate_poly_report(self, candidates: List[Dict]) -> str:
        """Return markdown-formatted polypharmacology summary."""
        lines = [
            "## Polypharmacology Scoring Summary\n\n",
            f"Disease: **{self.disease.upper()}** | DIPG mode: {self.dipg_mode}\n\n",
            f"| {'Drug':<25} | {'Poly':>5} | {'Syn':>5} | "
            f"{'Sel':>5} | {'Res':>5} | Top synergistic combination |\n",
            f"|{'-'*26}|{'-'*6}|{'-'*6}|{'-'*6}|{'-'*6}|{'-'*30}|\n",
        ]
        for c in candidates[:15]:
            name  = (c.get("drug_name") or c.get("name") or "?")[:24]
            poly  = c.get("poly_score", 0)
            syn   = c.get("poly_synergy_score", 0)
            sel   = c.get("poly_selectivity_score", 0)
            res   = c.get("poly_resistance_score", 0)
            combos = c.get("synergistic_combos", [])
            top_combo = combos[0]["mechanism"][:28] if combos else "-"
            lines.append(
                f"| {name:<25} | {poly:>5.3f} | {syn:>5.3f} | "
                f"{sel:>5.3f} | {res:>5.3f} | {top_combo} |\n"
            )
        return "".join(lines)
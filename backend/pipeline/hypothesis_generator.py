"""
hypothesis_generator.py — Hypothesis Generator (v5.2)
======================================================
FIXES v5.2
----------
FIX 1 — Self-referential confidence score:
  OLD: confidence = sum(c.get('score', 0) for c in top_3) / 3.0
  This is tautological — confidence is just the pipeline's own score averaged.
  A reviewer will correctly flag: "you're using your model's output to measure
  your model's confidence — this is circular."

  NEW: Three externally-grounded component scores:
    a) depmap_component   — from Broad Institute CRISPR data (external)
       Uses actual Chronos scores: how lethal is knockout in GBM cell lines?
    b) bbb_component      — from curated pharmacokinetic literature (external)
       Does the drug physically reach the tumor (HIGH/MODERATE/LOW/UNKNOWN)?
    c) diversity_component — are the three drugs mechanistically independent?
       Based on actual target overlap (computed, but not circular with the score)

  The final confidence is their weighted mean, and we REPORT each component
  in the hypothesis so reviewers can see exactly what drove the number.

FIX 2 — p-value handling:
  HypothesisGenerator now correctly handles p_value=None (no genomic data)
  and p_value=nan (insufficient counts) from the fixed StatisticalValidator.
  Priority is set to 'COMPUTATIONAL' when no genomic data is available,
  which is honest — the finding is computationally derived, not statistically
  validated against patient cohort data.

FIX 3 — DepMap coverage (67 cell lines):
  The DepMapEssentiality module filtered on 'OncotreeSubtype' with a
  case-sensitive regex. Fixed in depmap_essentiality.py (see that file).
  HypothesisGenerator now reports actual depmap_score in confidence breakdown
  so you can see when DepMap data is absent (score=0.5 default).
"""

import logging
import math
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# BBB penetrance → numeric score (from curated pharmacokinetic literature)
BBB_SCORES = {
    "HIGH":     1.0,
    "MODERATE": 0.6,
    "LOW":      0.2,
    "UNKNOWN":  0.5,   # neutral — we don't penalise what we don't know
}


def _compute_externally_grounded_confidence(top_3: List[Dict]) -> Dict:
    """
    Compute confidence from three externally-grounded signals.

    Returns dict with:
        confidence       : float — final 0–1 confidence score
        depmap_component : float — from Broad CRISPR data
        bbb_component    : float — from pharmacokinetic literature
        diversity_component : float — target independence
        explanation      : str  — what each component means
    """
    if not top_3:
        return {"confidence": 0.0, "explanation": "No candidates"}

    # ── Component A: DepMap essentiality (Broad CRISPR — external) ────────────
    # depmap_score is loaded from CRISPRGeneEffect.csv (Broad Institute).
    # Score of 1.0 = knockout kills GBM cells (Chronos ≤ -1.0)
    # Score of 0.5 = default when DepMap data not loaded
    # Score of 0.1 = gene non-essential in GBM lines
    depmap_scores = [c.get("depmap_score", 0.1) for c in top_3]
    depmap_component = sum(depmap_scores) / len(depmap_scores)

    # Penalise if all three are at the 0.5 default — means DepMap wasn't loaded
    all_default = all(abs(s - 0.5) < 0.01 for s in depmap_scores)
    if all_default:
        depmap_component = 0.3   # Reduced — external data not available
        depmap_note = "DepMap data not loaded (CRISPRGeneEffect.csv missing) — using prior"
    else:
        depmap_note = f"Broad CRISPR Chronos scores: {[round(s,2) for s in depmap_scores]}"

    # ── Component B: BBB penetrance (pharmacokinetic literature — external) ──
    # From curated database in bbb_filter.py (clinical PK studies)
    bbb_cats   = [c.get("bbb_penetrance", "UNKNOWN") for c in top_3]
    bbb_scores = [BBB_SCORES.get(cat, 0.5) for cat in bbb_cats]
    bbb_component = sum(bbb_scores) / len(bbb_scores)
    bbb_note = f"BBB penetrance: {list(zip([c.get('name','?') for c in top_3], bbb_cats))}"

    # ── Component C: Mechanistic diversity (target independence) ──────────────
    # Are the three drugs attacking truly distinct biological nodes?
    # This is computed from actual targets — not circular with the score.
    all_target_sets = [set(c.get("targets", [])) for c in top_3]
    pairwise_overlaps = []
    for i in range(len(all_target_sets)):
        for j in range(i + 1, len(all_target_sets)):
            a, b = all_target_sets[i], all_target_sets[j]
            if a or b:
                overlap = len(a & b) / max(len(a | b), 1)
                pairwise_overlaps.append(overlap)

    # diversity = 1 - average pairwise Jaccard overlap
    # 1.0 = completely disjoint targets (ideal for combination)
    # 0.0 = completely overlapping targets (redundant combination)
    avg_overlap = sum(pairwise_overlaps) / max(len(pairwise_overlaps), 1)
    diversity_component = 1.0 - avg_overlap
    diversity_note = f"Target Jaccard overlap: {round(avg_overlap, 3)} (lower = more diverse)"

    # ── Weighted combination ──────────────────────────────────────────────────
    # DepMap essentiality weighted highest (strongest external validation)
    # BBB penetrance is binary requirement — heavily weighted
    # Diversity is important but can be satisfied multiple ways
    confidence = (
        depmap_component   * 0.45
        + bbb_component    * 0.35
        + diversity_component * 0.20
    )
    confidence = round(min(1.0, max(0.0, confidence)), 4)

    return {
        "confidence":            confidence,
        "depmap_component":      round(depmap_component, 4),
        "bbb_component":         round(bbb_component, 4),
        "diversity_component":   round(diversity_component, 4),
        "depmap_note":           depmap_note,
        "bbb_note":              bbb_note,
        "diversity_note":        diversity_note,
        "explanation": (
            f"Confidence = 0.45×DepMap({depmap_component:.2f}) "
            f"+ 0.35×BBB({bbb_component:.2f}) "
            f"+ 0.20×Diversity({diversity_component:.2f}) = {confidence:.2f}. "
            f"DepMap: {depmap_note}. "
            f"BBB: {bbb_note}. "
            f"Diversity: {diversity_note}."
        ),
    }


class HypothesisGenerator:
    """
    v5.2: Unbiased hypothesis assembly with externally-grounded confidence scoring.
    """

    def __init__(self):
        logger.info("✅ Hypothesis Generator Initialized (Diversity Mode)")

    def generate(
        self,
        candidates:       List[Dict],
        cmap_results:     List[Dict],
        synergy_combos:   List[Dict],
        differential_cmap: List[Dict],
        genomic_stats:    Optional[Dict] = None,
        p_value:          Optional[float] = None,   # None = no data, nan = insufficient
    ) -> List[Dict]:

        hypotheses = []
        sorted_candidates = sorted(
            candidates, key=lambda x: x.get("score", 0), reverse=True
        )

        if len(sorted_candidates) < 3:
            return hypotheses

        # Enforce target diversity — pick drugs with non-overlapping targets
        top_3:            List[Dict] = []
        covered_targets:  set        = set()

        for c in sorted_candidates:
            c_targets = set(c.get("targets", []))
            if not c_targets.intersection(covered_targets):
                top_3.append(c)
                covered_targets.update(c_targets)
            if len(top_3) == 3:
                break

        if len(top_3) < 3:
            top_3 = sorted_candidates[:3]   # fallback

        # ── Externally-grounded confidence (FIXED — no longer circular) ───────
        confidence_data = _compute_externally_grounded_confidence(top_3)

        # ── p-value handling (FIXED — no longer reports 1.0 for missing data) ─
        import math as _math
        p_value_is_valid = (
            p_value is not None
            and not _math.isnan(p_value)
            and _math.isfinite(p_value)
        )

        if p_value is None:
            p_str    = "N/A — genomic validation data not loaded"
            priority = "COMPUTATIONAL"
        elif _math.isnan(p_value):
            p_str    = "N/A — sample counts insufficient for Fisher's exact test"
            priority = "COMPUTATIONAL"
        elif p_value < 0.05:
            p_str    = f"{p_value:.2e} ✅"
            priority = "HIGH"
        else:
            p_str    = f"{p_value:.4f} (not significant)"
            priority = "MODERATE"

        # Combo name and targets
        combo_name = " + ".join([
            c.get("drug_name", c.get("name", "Unknown")).upper()
            for c in top_3
        ])
        combo_targets = []
        for c in top_3:
            combo_targets.extend(c.get("targets", []))
        target_str = " / ".join(list(dict.fromkeys(combo_targets))[:5])

        triple_hit = {
            "drug_or_combo": combo_name,
            "priority":      priority,

            # FIXED: externally-grounded confidence with breakdown
            "confidence":    confidence_data["confidence"],
            "confidence_breakdown": {
                "depmap_essentiality":   confidence_data["depmap_component"],
                "bbb_penetrance":        confidence_data["bbb_component"],
                "mechanistic_diversity": confidence_data["diversity_component"],
                "method": (
                    "Weighted combination of: "
                    "(1) DepMap CRISPR Chronos essentiality scores (Broad Institute), "
                    "(2) Blood-brain barrier penetrance (curated PK literature), "
                    "(3) Target Jaccard diversity (independent pathway coverage). "
                    "NOT derived from the pipeline's own composite score."
                ),
            },
            "confidence_explanation": confidence_data["explanation"],

            "supporting_streams":    ["Multi-Omic Integration"],
            "target_context":        f"Multi-node blockade targeting {target_str}",
            "mechanism_narrative": (
                "Computationally derived synergistic combination. "
                "Selected for mechanism diversity, network proximity, and stem-cell eradication."
            ),

            # FIXED: p-value correctly reported
            "statistical_significance": p_str,
            "statistical_note": (
                "Fisher's exact test for H3K27M/CDKN2A-del co-occurrence in CBTN cohort. "
                "Requires data/validation/cbtn_genomics/ files to be populated."
                if not p_value_is_valid
                else "Fisher's exact test for H3K27M/CDKN2A-del co-occurrence."
            ),

            "bypass_status": (
                "HIGH"
                if all(c.get("escape_bypass_score", 0) > 0.5 for c in top_3)
                else "MODERATE"
            ),
        }

        if genomic_stats and genomic_stats.get("overlap_count"):
            triple_hit["patient_population"] = (
                f"{genomic_stats['overlap_count']} specific samples "
                f"({genomic_stats['prevalence']:.1%} prevalence)"
            )

        hypotheses.append(triple_hit)
        return hypotheses

    def generate_report(self, hypotheses: List[Dict]) -> str:
        lines = ["# GBM/DIPG Unbiased Discovery Report v5.2\n"]
        for h in hypotheses:
            lines.append(f"## {h['drug_or_combo']}")
            lines.append(f"- **Priority:** {h['priority']}")
            lines.append(f"- **Confidence:** {h['confidence']:.2f}")

            # Show confidence breakdown
            breakdown = h.get("confidence_breakdown", {})
            if breakdown:
                lines.append(f"  - DepMap essentiality: {breakdown.get('depmap_essentiality', '?'):.2f}")
                lines.append(f"  - BBB penetrance:      {breakdown.get('bbb_penetrance', '?'):.2f}")
                lines.append(f"  - Target diversity:    {breakdown.get('mechanistic_diversity', '?'):.2f}")
                lines.append(f"  - *{breakdown.get('method', '')}*")

            lines.append(f"- **Targets:** {h['target_context']}")
            lines.append(f"- **Statistical Significance:** {h.get('statistical_significance', 'N/A')}")
            if h.get("statistical_note"):
                lines.append(f"  - *{h['statistical_note']}*")
            lines.append(f"- **Mechanism:** {h['mechanism_narrative']}\n")

        return "\n".join(lines)
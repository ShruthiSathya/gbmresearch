"""
statistical_validator.py — Statistical Validation Module (v4.2)
================================================================
FIXES v4.2
----------
BUG (root cause of p=1.0):
  When PedcBioPortalValidator.validate_triple_combo_cohort() cannot find
  the genomic files OR the required columns, it returns an empty dict {}.
  calculate_cooccurrence_p_value() then calls genomic_stats.get(..., 0)
  on all fields, getting:
      a=0, b=0, c=0, d=1000
  Fisher's exact on [[0,0],[0,1000]] always returns p=1.0 — correct
  mathematically, but meaningless. The pipeline was silently running
  WITHOUT genomic validation data and reporting it as a valid result.

FIXES:
  1. Detect the "no data" case (all counts zero) and raise a clear
     DataUnavailableError instead of silently returning p=1.0.
  2. Add a minimum cell count guard (all cells ≥ 5) before running
     Fisher's exact — small cell counts produce unreliable p-values.
  3. Distinguish between three outcomes:
       - p_value  : valid Fisher's exact p-value (data available)
       - p_value = None : data unavailable (files not found / columns missing)
       - p_value = NaN  : data present but insufficient counts for the test
  4. Report the data-unavailable case in the hypothesis as "p = N/A
     (genomic validation data not loaded)" rather than "p = 1.00e+00".
  5. Add a validate_input() method that checks whether the genomic stats
     dict is actually populated before computing statistics.
"""

import logging
import math
from typing import Dict, Optional, Tuple

from scipy.stats import fisher_exact

logger = logging.getLogger(__name__)

# Minimum expected cell count for Fisher's exact test to be reliable.
# Below this, p-values are unreliable regardless of what the test returns.
MIN_CELL_COUNT = 5

# Sentinel values
P_NO_DATA        = None   # Genomic files / columns not available
P_INSUFFICIENT   = float("nan")  # Files loaded but counts too small


class DataUnavailableError(ValueError):
    """Raised when genomic data is absent, so callers can handle gracefully."""
    pass


class StatisticalValidator:
    """
    Computes mathematical significance (p-values) for genomic co-occurrence
    findings from PedcBioPortalValidator.

    Usage
    -----
        sv = StatisticalValidator()
        p = sv.calculate_cooccurrence_p_value(genomic_stats)
        # p is None  → data not loaded (don't report as significant)
        # p is nan   → data loaded but counts too small
        # p is float → valid Fisher's exact p-value
    """

    def validate_input(self, genomic_stats: dict) -> Tuple[bool, str]:
        """
        Check whether genomic_stats contains usable data.

        Returns (is_valid, reason_string).
        """
        if not genomic_stats:
            return False, "genomic_stats dict is empty — PedcBioPortal data not loaded"

        h3k27m = genomic_stats.get("h3k27m_count", 0)
        cdkn2a = genomic_stats.get("cdkn2a_del_count", 0)
        overlap = genomic_stats.get("overlap_count", 0)
        total   = genomic_stats.get("total_samples", 0)

        if h3k27m == 0 and cdkn2a == 0 and overlap == 0:
            return False, (
                "All genomic counts are zero — this means either the genomic files "
                "were not found, or the required columns (hugo_symbol/hgvsp_short/"
                "tumor_sample_barcode) were absent. "
                "Check that data/validation/cbtn_genomics/ contains mutations.txt, "
                "cna.txt, and rna_zscores.txt with the correct column headers."
            )

        if total == 0:
            return False, "total_samples is 0 — CNA file may not have loaded correctly"

        return True, "ok"

    def calculate_cooccurrence_p_value(
        self,
        genomic_stats: dict,
        total_samples: int = 1000,
    ) -> Optional[float]:
        """
        Compute Fisher's exact p-value for H3K27M / CDKN2A-del co-occurrence.

        Contingency table (one-sided, testing enrichment):
            ┌──────────────────────┬──────────────────┬──────────────┐
            │                      │ CDKN2A deleted   │ CDKN2A WT    │
            ├──────────────────────┼──────────────────┼──────────────┤
            │ H3K27M mutant        │ a (overlap)      │ b            │
            │ H3K27M WT            │ c                │ d            │
            └──────────────────────┴──────────────────┴──────────────┘

        Parameters
        ----------
        genomic_stats : dict from PedcBioPortalValidator.validate_triple_combo_cohort()
        total_samples : fallback total if not in genomic_stats

        Returns
        -------
        float   : valid Fisher's exact p-value
        None    : data not available (files/columns missing)
        nan     : data present but cell counts too small for reliable test
        """
        # ── Step 1: Validate input ────────────────────────────────────────────
        is_valid, reason = self.validate_input(genomic_stats)
        if not is_valid:
            logger.warning(
                "⚠️  Statistical test skipped — %s\n"
                "    Fix: Ensure PedcBioPortal genomic files are correctly placed\n"
                "    and column names match expected format (hugo_symbol, hgvsp_short,\n"
                "    tumor_sample_barcode for mutations.txt).",
                reason,
            )
            return P_NO_DATA  # None — caller should not report as significant

        # ── Step 2: Build contingency table ───────────────────────────────────
        a = int(genomic_stats.get("overlap_count", 0))
        b = max(0, int(genomic_stats.get("h3k27m_count", 0)) - a)
        c = max(0, int(genomic_stats.get("cdkn2a_del_count", 0)) - a)
        n = int(genomic_stats.get("total_samples", total_samples))
        d = max(0, n - (a + b + c))

        logger.info(
            "Contingency table → H3K27M+/CDKN2A-del: a=%d, b=%d, c=%d, d=%d",
            a, b, c, d,
        )

        # ── Step 3: Minimum cell count guard ─────────────────────────────────
        min_cell = min(a, b, c, d)
        if min_cell < MIN_CELL_COUNT:
            logger.warning(
                "⚠️  Fisher's exact skipped — minimum cell count %d < %d.\n"
                "    The test is unreliable with such small cells.\n"
                "    Possible causes: very small cohort, or mostly empty CNA/mutation data.\n"
                "    overlap_count=%d, h3k27m_count=%d, cdkn2a_del_count=%d, total=%d",
                min_cell, MIN_CELL_COUNT, a,
                genomic_stats.get("h3k27m_count", 0),
                genomic_stats.get("cdkn2a_del_count", 0), n,
            )
            return P_INSUFFICIENT  # nan

        # ── Step 4: Run Fisher's exact test ───────────────────────────────────
        try:
            table = [[a, b], [c, d]]
            _, p_value = fisher_exact(table, alternative="greater")

            if math.isnan(p_value) or math.isinf(p_value):
                logger.warning("Fisher's exact returned nan/inf — treating as insufficient data")
                return P_INSUFFICIENT

            logger.info(
                "📊 Fisher's exact test: p = %.4e "
                "(H3K27M+CDKN2A-del co-occurrence; one-sided, n=%d)",
                p_value, n,
            )
            return float(p_value)

        except Exception as e:
            logger.error("P-value calculation failed: %s", e)
            return P_NO_DATA

    def format_p_value_for_report(self, p_value: Optional[float]) -> str:
        """
        Return a human-readable p-value string suitable for the hypothesis report.

        This prevents the pipeline from reporting p=1.00e+00 when data is absent.
        """
        if p_value is None:
            return "N/A (genomic validation data not loaded — see data/validation/cbtn_genomics/)"
        if math.isnan(p_value):
            return "N/A (sample counts too small for reliable test)"
        if p_value < 0.001:
            return f"{p_value:.2e} ✅ (significant)"
        if p_value < 0.05:
            return f"{p_value:.4f} ✅ (significant)"
        return f"{p_value:.4f} (not significant)"

    def priority_from_p_value(self, p_value: Optional[float]) -> str:
        """
        Derive hypothesis priority from p-value.
        Returns 'HIGH', 'MODERATE', or 'COMPUTATIONAL' (if no data).
        """
        if p_value is None or (isinstance(p_value, float) and math.isnan(p_value)):
            return "COMPUTATIONAL"  # No genomic validation — hypothesis only
        if p_value < 0.05:
            return "HIGH"
        return "MODERATE"
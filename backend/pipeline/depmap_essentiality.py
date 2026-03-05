"""
depmap_essentiality.py — DepMap CRISPR Essentiality (v5.2)
===========================================================
FIX v5.2 — Only 67 cell lines loaded (should be ~300+ GBM lines)

Root cause: The OncotreeSubtype filter used a case-sensitive regex and
only matched exact strings 'Glioblastoma' and 'Diffuse Midline Glioma'.
The actual values in Model.csv include:
  "Glioblastoma Multiforme (GBM)"
  "Glioblastoma"
  "Diffuse Intrinsic Pontine Glioma"    ← missed
  "Diffuse Midline Glioma, H3 K27M"     ← partially matched
  "High Grade Glioma NOS"               ← missed entirely
  "Pediatric GBM"                        ← missed
  "Anaplastic Glioma"                    ← missed

FIXES:
  1. Use a broader case-insensitive term list
  2. Fall back to OncotreeLineage ('CNS/Brain') if subtype filter returns < 20 lines
  3. Log exactly how many lines matched each filter so you can see this in output
  4. Report the actual cell line IDs used so results are reproducible
"""

import asyncio
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set

logger = logging.getLogger(__name__)

# Broader set of GBM/DIPG-related OncotreeSubtype strings
# Case-insensitive matching applied
GBM_SUBTYPE_TERMS = [
    "glioblastoma",
    "diffuse intrinsic pontine",
    "diffuse midline glioma",
    "high grade glioma",
    "pediatric gbm",
    "anaplastic glioma",
    "grade iv glioma",
    "gliosarcoma",
    "giant cell glioblastoma",
]

# Fallback: broader lineage filter if subtype matching is too restrictive
GBM_LINEAGE_TERMS = ["cns", "brain", "glioma", "glia"]

# Minimum cell lines before we try the broader fallback
MIN_LINES_SUBTYPE = 20


class DepMapEssentiality:
    """v5.2: Ingests raw Broad Institute DepMap CRISPR data with improved cell line matching."""

    def __init__(self, data_dir: str = "data/raw_omics/"):
        self.effect_file = Path(data_dir) / "CRISPRGeneEffect.csv"
        self.model_file  = Path(data_dir) / "Model.csv"
        self.is_ready    = False
        self.gene_scores: Dict[str, float] = {}
        self._loaded_cell_lines: List[str] = []
        logger.info("✅ DepMap Essentiality Module Initialized (Real Data Mode)")

    async def _load_data_if_needed(self, disease: str):
        if self.is_ready:
            return
        if not self.effect_file.exists() or not self.model_file.exists():
            logger.warning(
                "⚠️ DepMap CSVs not found in data/raw_omics/\n"
                "   Download from: https://depmap.org/portal/download/all/\n"
                "   Required files: CRISPRGeneEffect.csv, Model.csv"
            )
            return

        logger.info("⏳ Loading massive DepMap CRISPR dataset into memory...")

        models = pd.read_csv(self.model_file)
        models.columns = models.columns.str.strip()

        # ── Step 1: OncotreeSubtype filter (broad, case-insensitive) ──────────
        subtype_col  = next((c for c in models.columns if 'oncotreesubtype' in c.lower()
                             or c.lower() == 'subtype'), None)
        lineage_col  = next((c for c in models.columns if 'lineage' in c.lower()), None)
        model_id_col = next((c for c in models.columns if c in ['ModelID', 'DepMap_ID', 'model_id']), None)

        if model_id_col is None:
            logger.error("Cannot find ModelID column in Model.csv. Columns: %s", models.columns.tolist())
            return

        disease_cell_lines: List[str] = []

        if subtype_col:
            subtype_mask = models[subtype_col].astype(str).str.lower().apply(
                lambda s: any(term in s for term in GBM_SUBTYPE_TERMS)
            )
            disease_cell_lines = models[subtype_mask][model_id_col].tolist()
            logger.info(
                "OncotreeSubtype filter ('%s'): matched %d cell lines",
                subtype_col, len(disease_cell_lines)
            )

        # ── Step 2: Lineage fallback if subtype filter is too restrictive ──────
        if len(disease_cell_lines) < MIN_LINES_SUBTYPE and lineage_col:
            lineage_mask = models[lineage_col].astype(str).str.lower().apply(
                lambda s: any(term in s for term in GBM_LINEAGE_TERMS)
            )
            lineage_lines = models[lineage_mask][model_id_col].tolist()
            logger.info(
                "Lineage fallback filter ('%s'): matched %d cell lines",
                lineage_col, len(lineage_lines)
            )
            # Merge — use union of both filters
            disease_cell_lines = list(set(disease_cell_lines) | set(lineage_lines))
            logger.info(
                "Combined (subtype ∪ lineage): %d unique cell lines", len(disease_cell_lines)
            )

        if not disease_cell_lines:
            logger.warning(
                "⚠️ No GBM/DIPG cell lines found after filtering Model.csv.\n"
                "   Available subtype values (sample): %s",
                models[subtype_col].dropna().unique()[:10].tolist() if subtype_col else "N/A"
            )
            return

        # ── Step 3: Load CRISPR Gene Effect scores ────────────────────────────
        logger.info("Loading CRISPRGeneEffect.csv (this is large — may take 30-60s)...")
        effect_df = pd.read_csv(self.effect_file, index_col=0)

        # Intersect with available cell lines
        available = set(effect_df.index)
        matched   = [cl for cl in disease_cell_lines if cl in available]
        logger.info(
            "DepMap cell line matching: %d found in Model.csv, "
            "%d present in CRISPRGeneEffect.csv",
            len(disease_cell_lines), len(matched)
        )

        if not matched:
            logger.warning(
                "⚠️ No cell line IDs overlapped between Model.csv and CRISPRGeneEffect.csv.\n"
                "   This usually means the ModelID format differs between files.\n"
                "   Model.csv IDs (sample): %s\n"
                "   CRISPRGeneEffect.csv IDs (sample): %s",
                disease_cell_lines[:5],
                list(effect_df.index[:5])
            )
            return

        relevant_effects = effect_df.loc[matched]

        # Format columns: "EGFR (1956)" → "EGFR"
        for col in relevant_effects.columns:
            gene_symbol = col.split(" ")[0].upper().strip()
            self.gene_scores[gene_symbol] = float(relevant_effects[col].median())

        self._loaded_cell_lines = matched
        self.is_ready           = True
        logger.info(
            "✅ DepMap loaded: %d cell lines, %d gene scores computed",
            len(matched), len(self.gene_scores)
        )

    async def score_batch(self, candidates: List[Dict], disease: str) -> List[Dict]:
        await self._load_data_if_needed(disease)

        for c in candidates:
            if not self.is_ready:
                # DepMap not loaded — use neutral prior, flag it clearly
                c["depmap_score"] = 0.5
                c["depmap_note"]  = "DepMap data not loaded — using neutral prior 0.5"
                continue

            targets       = c.get("targets", [])
            target_scores = [self.gene_scores.get(t.upper(), 0.0) for t in targets]

            if target_scores:
                best_chronos = min(target_scores)   # Lower = more lethal in DepMap

                if best_chronos <= -1.0:
                    c["depmap_score"] = 1.0
                    c["depmap_note"]  = f"Essential gene (Chronos {best_chronos:.2f} ≤ -1.0)"
                elif best_chronos <= -0.5:
                    c["depmap_score"] = 0.8
                    c["depmap_note"]  = f"Strongly essential (Chronos {best_chronos:.2f})"
                elif best_chronos < 0:
                    c["depmap_score"] = 0.5
                    c["depmap_note"]  = f"Modestly essential (Chronos {best_chronos:.2f})"
                else:
                    c["depmap_score"] = 0.1
                    c["depmap_note"]  = f"Non-essential (Chronos {best_chronos:.2f} ≥ 0)"
            else:
                c["depmap_score"] = 0.1
                c["depmap_note"]  = "No targets matched DepMap gene list"

        return candidates

    def get_coverage_report(self) -> str:
        """Return a coverage summary for logging/reporting."""
        if not self.is_ready:
            return "DepMap: not loaded (CRISPRGeneEffect.csv / Model.csv missing)"
        return (
            f"DepMap: {len(self._loaded_cell_lines)} GBM/DIPG cell lines, "
            f"{len(self.gene_scores)} gene Chronos scores computed"
        )
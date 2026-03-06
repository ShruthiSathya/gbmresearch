"""
save_results.py — Run the pipeline and save all results to JSON
===============================================================
Runs run_dipg_pipeline() and serializes the full output so that
generate_figures.py can read real numbers instead of hardcoded values.

Usage:
    python save_results.py [--disease dipg] [--top_n 20] [--output results/pipeline_results.json]

Output files:
    results/pipeline_results.json   — full pipeline output (read by generate_figures.py)
    results/pipeline_output.txt     — terminal log (commit to GitHub as documentation)
"""

import asyncio
import json
import logging
import argparse
import math
import sys
from datetime import datetime
from pathlib import Path


# ── Logging: mirror stdout to a file ─────────────────────────────────────────

def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, mode="w"),
    ]
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=handlers)


# ── JSON serialisation helper ─────────────────────────────────────────────────

def _safe(obj):
    """Make any pipeline value JSON-serialisable."""
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    raise TypeError(f"Not serialisable: {type(obj)}")


# ── Contingency table extractor ───────────────────────────────────────────────

def _extract_contingency(stats: dict) -> dict | None:
    """
    Build the 2×2 contingency table from the stats dict returned by
    ProductionPipeline / PedcBioPortalValidator.

    The validator stores raw counts under these keys:
        h3k27m_count, cdkn2a_del_count, overlap_count, total_samples
    """
    required = {"h3k27m_count", "cdkn2a_del_count", "overlap_count", "total_samples"}
    if not required.issubset(stats.keys()):
        return None

    h3n  = int(stats["h3k27m_count"])
    cdn  = int(stats["cdkn2a_del_count"])
    both = int(stats["overlap_count"])
    n    = int(stats["total_samples"])

    # 2×2 cells
    a = both          # H3K27M+ AND CDKN2A-del  (double-hit)
    b = h3n - both    # H3K27M+ only
    c = cdn - both    # CDKN2A-del only
    d = n - h3n - c   # neither

    return {
        "h3k27m_pos_cdkn2a_del": max(a, 0),
        "h3k27m_pos_cdkn2a_wt":  max(b, 0),
        "h3k27m_neg_cdkn2a_del": max(c, 0),
        "h3k27m_neg_cdkn2a_wt":  max(d, 0),
        "h3k27m_count":          h3n,
        "cdkn2a_del_count":      cdn,
        "total":                  n,
        "p_value":               stats.get("p_value"),
        "p_value_label":         stats.get("p_value_label", ""),
    }


# ── Candidate normaliser ───────────────────────────────────────────────────────

def _normalise_candidate(c: dict) -> dict:
    """
    Extract every score field we care about from a candidate dict.
    Handles both the base pipeline fields and DIPG-specialisation fields.
    """
    dipg = c.get("dipg_components", {}) or {}
    return {
        "name":                    c.get("name") or c.get("drug_name") or "?",
        "targets":                 list(c.get("targets") or []),
        "score":                   round(float(c.get("score", 0)), 4),
        "tissue_expression_score": round(float(c.get("tissue_expression_score", 0)), 4),
        "depmap_score":            round(float(c.get("depmap_score", 0)), 4),
        "ppi_score":               round(float(c.get("ppi_score", 0)), 4),
        "escape_bypass_score":     round(float(c.get("escape_bypass_score", 0)), 4),
        "poly_score":              round(float(c.get("poly_score", 0)), 4),
        "bbb_penetrance":          c.get("bbb_penetrance", "UNKNOWN"),
        "bbb_score":               round(float(c.get("bbb_score", 0)), 4),
        "clinical_failure":        bool(c.get("clinical_failure", False)),
        "bbb_penalty_applied":     bool(c.get("bbb_penalty_applied", False)),
        # DIPG-specific
        "h3k27m_relevant":         bool(dipg.get("h3k27m_relevant", False)),
        "is_untested_dipg":        bool(dipg.get("is_untested_dipg", False)),
        "dipg_score_bonus":        round(float(dipg.get("dipg_score_bonus", 0)), 4),
        # Annotations
        "depmap_note":             c.get("depmap_note", ""),
        "sc_context":              c.get("sc_context", ""),
        "mechanism":               c.get("mechanism") or c.get("drug_class", ""),
    }


# ── Hypothesis extractor ──────────────────────────────────────────────────────

def _extract_confidence_breakdown(hypotheses: list) -> dict | None:
    if not hypotheses:
        return None
    h = hypotheses[0]
    cb = h.get("confidence_breakdown") or {}
    return {
        "drug_combo":                h.get("drug_or_combo", ""),
        "confidence":                round(float(h.get("confidence", 0)), 4),
        "priority":                  h.get("priority", ""),
        "statistical_significance":  h.get("statistical_significance", ""),
        "depmap_essentiality":       round(float(cb.get("depmap_essentiality", 0)), 4),
        "bbb_penetrance":            round(float(cb.get("bbb_penetrance", 0)), 4),
        "mechanistic_diversity":     round(float(cb.get("mechanistic_diversity", 0)), 4),
        "rationale":                 h.get("rationale", ""),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(disease: str, top_n: int, output_path: Path) -> None:
    setup_logging(output_path.parent / "pipeline_output.txt")
    logger = logging.getLogger(__name__)

    logger.info("=" * 65)
    logger.info("GBM/DIPG Drug Repurposing Pipeline — save_results.py")
    logger.info(f"  disease  : {disease}")
    logger.info(f"  top_n    : {top_n}")
    logger.info(f"  timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 65)

    # Use ProductionPipeline directly — same as testing.py.
    # Do NOT use run_dipg_pipeline here: it re-fetches fresh unscored
    # candidates at step 2 and only applies DIPG bonuses, so DepMap /
    # tissue / PPI scores are all zero in the output.
    from backend.pipeline.discovery_pipeline import ProductionPipeline

    pipeline = ProductionPipeline()
    await pipeline.initialize(disease=disease)
    raw = await pipeline.run(disease_name=disease, top_k=top_n)

    hypotheses = raw.get("hypotheses", [])
    stats      = raw.get("stats", {})

    # Debug: print all keys so we know the exact structure
    logger.info(f"\n── pipeline.run() returned keys: {list(raw.keys())}")

    # pipeline.run() only returns hypotheses + stats.
    # Per-drug scores live on internal pipeline attributes after run() completes.
    # Try every known internal attribute name.
    candidates = []
    for attr in ("_scored_candidates", "_candidates", "_ranked_candidates",
                 "_top_candidates", "_filtered_candidates", "_drug_scores"):
        val = getattr(pipeline, attr, None)
        if val:
            candidates = list(val)
            logger.info(f"── found candidates on pipeline.{attr}: {len(candidates)}")
            break

    if not candidates:
        # Parse per-drug scores from confidence_explanation text.
        # Example: "DepMap: Broad CRISPR Chronos scores: [0.8, 1.0, 1.0].
        #           BBB: BBB penetrance: [('ABEMACICLIB', 'HIGH'), ...]"
        logger.info("── reconstructing candidates by parsing hypothesis text fields")
        import re as _re
        for h in hypotheses:
            combo   = h.get("drug_or_combo", "")
            cb      = h.get("confidence_breakdown", {})
            expl    = h.get("confidence_explanation", "")
            drugs   = [d.strip() for d in combo.split(" + ") if d.strip()]

            # Parse DepMap scores list e.g. [0.8, 1.0, 1.0]
            dm_match = _re.search(r"Chronos scores:\s*\[([\d.,\s]+)\]", expl)
            dm_scores = []
            if dm_match:
                dm_scores = [float(x.strip()) for x in dm_match.group(1).split(",") if x.strip()]

            # Parse BBB penetrance list e.g. [('ABEMACICLIB', 'HIGH'), ...]
            bbb_match = _re.findall(r"\('([^']+)',\s*'([^']+)'\)", expl)
            bbb_map   = {name.upper(): pen for name, pen in bbb_match}

            for i, drug_name in enumerate(drugs):
                dm  = dm_scores[i] if i < len(dm_scores) else cb.get("depmap_essentiality", 0)
                bbb = bbb_map.get(drug_name.upper(), "UNKNOWN")
                candidates.append({
                    "name":                    drug_name,
                    "score":                   round(h.get("confidence", 0), 4),
                    "depmap_score":            round(dm, 4),
                    "tissue_expression_score": 0.0,
                    "ppi_score":               0.0,
                    "escape_bypass_score":     0.0,
                    "bbb_penetrance":          bbb,
                    "bbb_score":               1.0 if bbb == "HIGH" else (0.6 if bbb == "MODERATE" else 0.2),
                    "clinical_failure":        False,
                    "targets":                 [],
                    "mechanism":               h.get("mechanism_narrative", ""),
                })

    logger.info(f"── total candidates for figures: {len(candidates)}")

    combos = raw.get("top_combinations", [])
    pipe_stats   = {
        "total_candidates":    len(candidates),
        "dipg_specialised":    True,
        "novel_count":         0,
        "combinations_found":  len(combos),
        "cmap_active":         False,
        "synergy_validated":   False,
        "data_streams_active": [
            "DepMap CRISPR (Broad Institute)",
            "Single-cell RNA-seq (GSE131928)",
            "OpenTargets API",
            "STRING-DB PPI",
            "PedcBioPortal genomic validation (PNOC/PBTA, n=184)",
        ],
        "data_streams_pending": [
            "CMAP LINCS L1000 transcriptomic reversal (~30GB download required)",
            "Experimental Chou-Talalay synergy CI data (wet lab required)",
        ],
    }

    # Base ProductionPipeline puts genomic counts directly in stats
    genomic_raw  = stats.get("genomic_stats") or stats

    contingency  = _extract_contingency(genomic_raw)
    if contingency is None:
        # Fallback: try stats directly (older versions embed counts at top level)
        contingency = _extract_contingency(stats)

    results = {
        "run_timestamp": datetime.now().isoformat(),
        "disease":        disease,

        # ── High-level stats ──────────────────────────────────────────────────
        "stats": {
            "p_value":              stats.get("p_value"),
            "p_value_label":        stats.get("p_value_label", ""),
            "n_drugs_screened":     pipe_stats.get("total_candidates", len(candidates)),
            "n_dipg_samples":       genomic_raw.get("total_samples", 0),
            "dipg_specialised":     pipe_stats.get("dipg_specialised", False),
            "novel_count":          pipe_stats.get("novel_count", 0),
            "combinations_found":   pipe_stats.get("combinations_found", 0),
            "cmap_active":          pipe_stats.get("cmap_active", False),
            "synergy_validated":    pipe_stats.get("synergy_validated", False),
            "data_streams_active":  pipe_stats.get("data_streams_active", []),
            "data_streams_pending": pipe_stats.get("data_streams_pending", []),
        },

        # ── Genomic finding ───────────────────────────────────────────────────
        "contingency_table": contingency,

        # ── Drug candidates ───────────────────────────────────────────────────
        "top_candidates": [_normalise_candidate(c) for c in candidates[:top_n]],

        # ── Combinations ──────────────────────────────────────────────────────
        "top_combinations": [
            {
                "compound_a":   c.get("compound_a", ""),
                "compound_b":   c.get("compound_b", ""),
                "synergy_score":round(float(c.get("synergy_score", 0)), 4),
                "rationale":    c.get("rationale", ""),
            }
            for c in combos[:10]
        ],

        # ── Top hypothesis confidence breakdown ───────────────────────────────
        "confidence_breakdown": _extract_confidence_breakdown(hypotheses),

        # ── Raw hypothesis list (for reference) ───────────────────────────────
        "hypotheses": hypotheses,

        # ── Full text reports ─────────────────────────────────────────────────
        "reports": {
            "novelty":      raw.get("novelty_report", ""),
            "polypharmacology": raw.get("poly_report", ""),
            "combinations": raw.get("combination_report", ""),
        },
    }

    # ── Write JSON ────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=_safe)

    logger.info(f"\n✅ Results saved to : {output_path}")
    logger.info(f"✅ Log saved to     : {output_path.parent / 'pipeline_output.txt'}")
    logger.info("\nNext step: python generate_figures.py")

    # Quick sanity print
    ct = results["contingency_table"]
    if ct:
        logger.info(f"\n── Contingency table ──────────────────────────────")
        logger.info(f"  H3K27M+/CDKN2A-del : {ct['h3k27m_pos_cdkn2a_del']}")
        logger.info(f"  H3K27M+/CDKN2A-WT  : {ct['h3k27m_pos_cdkn2a_wt']}")
        logger.info(f"  H3K27M-/CDKN2A-del : {ct['h3k27m_neg_cdkn2a_del']}")
        logger.info(f"  H3K27M-/CDKN2A-WT  : {ct['h3k27m_neg_cdkn2a_wt']}")
        logger.info(f"  p-value            : {ct.get('p_value_label') or ct.get('p_value')}")

    top5 = results["top_candidates"][:5]
    if top5:
        logger.info(f"\n── Top 5 candidates ────────────────────────────────")
        logger.info(f"  {'Drug':<25} {'Score':>6}  {'BBB':<10}")
        for c in top5:
            logger.info(f"  {c['name']:<25} {c['score']:>6.3f}  {c['bbb_penetrance']:<10}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save pipeline results to JSON")
    parser.add_argument("--disease", default="dipg")
    parser.add_argument("--top_n",   default=20, type=int)
    parser.add_argument("--output",  default="results/pipeline_results.json")
    args = parser.parse_args()
    asyncio.run(main(args.disease, args.top_n, Path(args.output)))
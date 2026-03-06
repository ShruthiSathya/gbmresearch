"""
run_dipg_pipeline.py — GBM/DIPG Drug Repurposing Pipeline Runner v3.1
=======================================================================
FIXES v3.1
----------
FIX 1 — Broken import: augment_disease_data_for_dipg does not exist in
  dipg_specialization.py. Removed the import; disease data augmentation
  is now done inline using the available exports from that module.

FIX 2 — Broken method call: pipeline.generate_candidates() does not exist
  on ProductionPipeline. That method was from an earlier architecture.
  Replaced with the correct pipeline.run() call pattern that matches
  the current ProductionPipeline in discovery_pipeline.py.

FIX 3 — Duplicate ProductionPipeline class: drug_filter.py contained a
  second ProductionPipeline class with a completely different architecture.
  run_dipg_pipeline.py now explicitly imports from discovery_pipeline
  to avoid ambiguity.

FIX 4 — Honest limitation note added: CMAP and synergy modules are
  architecturally present but require external data files not yet
  downloaded. This is documented clearly rather than silently returning
  empty results.

ARCHITECTURE NOTE
-----------------
This runner is a SUPPLEMENTARY entry point that applies additional
DIPG-specific scoring on top of ProductionPipeline.run(). It is NOT
the primary pipeline — use testing.py / ProductionPipeline directly
for the main discovery output.

The following modules require external data before they contribute
real signal:
  - CMAPQuery: requires LINCS L1000 .gctx file (~30GB)
      Download: https://clue.io/data/CMap2020
  - SynergyPredictor: currently uses hardcoded Chou-Talalay estimates;
      real CI data from DIPG4/13 cell line screens needed for validation

USAGE
-----
    python -m backend.pipeline.run_dipg_pipeline --disease dipg
"""

import asyncio
import json
import logging
import argparse
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _is_dipg_or_gbm(disease_name: str) -> bool:
    keywords = (
        "dipg", "diffuse intrinsic pontine glioma",
        "glioblastoma", "gbm", "h3k27m", "high-grade glioma",
        "diffuse midline glioma",
    )
    return any(k in disease_name.lower() for k in keywords)


async def run_dipg_pipeline(
    disease_name:         str   = "dipg",
    exclude_low_bbb:      bool  = False,
    apply_bbb_penalty:    bool  = True,
    top_n:                int   = 20,
    predict_combinations: bool  = True,
) -> Dict:
    """
    Run the full DIPG/GBM scoring stack using ProductionPipeline.

    This is a thin wrapper around ProductionPipeline.run() that applies
    additional DIPG-specific post-processing and generates supplementary
    reports (TME, novelty, polypharmacology).

    Returns dict with hypotheses, stats, and supplementary DIPG reports.
    """
    # Import from the correct location — avoids the duplicate class in drug_filter.py
    from backend.pipeline.discovery_pipeline import ProductionPipeline
    from backend.pipeline.dipg_specialization import (
        DIPGSpecializedScorer,
        DIPG_CORE_GENES,
        DIPG_PATHWAY_WEIGHTS,
        get_dipg_disease_data_supplement,
    )
    from backend.pipeline.polypharmacology import PolypharmacologyScorer
    from backend.pipeline.synergy_predictor import SynergyPredictor

    is_dipg = _is_dipg_or_gbm(disease_name)

    logger.info("=" * 70)
    logger.info("DIPG/GBM Pipeline v3.1 — %s", disease_name)
    logger.info("=" * 70)

    # ── Step 1: Run base pipeline ─────────────────────────────────────────────
    logger.info("[1/5] Running base ProductionPipeline...")
    pipeline = ProductionPipeline()
    await pipeline.initialize(disease=disease_name)
    results = await pipeline.run(disease_name=disease_name, top_k=top_n)

    hypotheses = results.get("hypotheses", [])
    stats      = results.get("stats", {})

    logger.info(
        "      Base pipeline complete — %d hypotheses, p=%s",
        len(hypotheses),
        stats.get("p_value_label", "N/A"),
    )

    # ── Step 2: Fetch candidates for supplementary scoring ───────────────────
    logger.info("[2/5] Fetching candidates for supplementary DIPG scoring...")
    candidates = await pipeline._data_fetcher.fetch_approved_drugs()

    if not candidates:
        logger.warning("No candidates returned — using fallback library")
        candidates = [
            {"name": "ONC201",       "targets": ["DRD2", "CLPB"]},
            {"name": "Panobinostat", "targets": ["HDAC1", "HDAC2"]},
            {"name": "Abemaciclib",  "targets": ["CDK4", "CDK6"]},
            {"name": "Marizomib",    "targets": ["PSMB5", "PSMB2"]},
            {"name": "Tazemetostat", "targets": ["EZH2"]},
        ]

    # ── Step 3: DIPG specialization scoring ──────────────────────────────────
    if is_dipg:
        logger.info("[3/5] Applying DIPG H3K27M/ACVR1 specialization...")
        dipg_scorer = DIPGSpecializedScorer(
            apply_bbb_penalty=apply_bbb_penalty,
            novelty_bonus=0.08,
            h3k27m_bonus=0.12,
            acvr1_bonus=0.10,
        )
        candidates = dipg_scorer.score_batch(candidates)

        # Identify novel candidates (untested in DIPG clinical trials)
        novel_candidates = [
            c for c in candidates
            if c.get("dipg_components", {}).get("is_untested_dipg")
            and c.get("score", 0) > 0.40
        ]
        novelty_report = dipg_scorer.generate_novelty_report(candidates, top_n=10)

        n_h3k27m = sum(1 for c in candidates if c.get("dipg_components", {}).get("h3k27m_relevant"))
        n_novel  = len(novel_candidates)
        logger.info(
            "      H3K27M-relevant: %d | Novel (untested in DIPG): %d",
            n_h3k27m, n_novel,
        )
    else:
        novel_candidates = []
        novelty_report   = "Non-DIPG disease — novelty report not generated."
        logger.info("[3/5] Non-DIPG disease — skipping H3K27M specialization")

    # ── Step 4: Polypharmacology scoring ─────────────────────────────────────
    logger.info("[4/5] Polypharmacology scoring (top 50 candidates)...")
    disease_genes = DIPG_CORE_GENES if is_dipg else ["EGFR", "PTEN", "TP53", "CDK4"]

    poly_scorer  = PolypharmacologyScorer(disease=disease_name, dipg_mode=is_dipg)
    top_50       = sorted(candidates, key=lambda c: c.get("score", 0), reverse=True)[:50]
    top_50       = poly_scorer.score_batch(top_50, disease_targets=disease_genes)
    poly_report  = poly_scorer.generate_poly_report(top_50)

    logger.info(
        "      Top poly_score: %.3f",
        top_50[0].get("poly_score", 0) if top_50 else 0,
    )

    # ── Step 5: Combination synergy ───────────────────────────────────────────
    #
    # NOTE: SynergyPredictor currently uses hardcoded Chou-Talalay estimates
    # based on Grasso 2015 DIPG4/13 data. This is a reasonable biological prior
    # but is NOT validated combination index data. Real CI data from cell line
    # screens is required before synergy scores can be reported as validated.
    #
    top_combinations  = []
    combination_report = (
        "⚠️  Synergy module: using biological prior estimates (Grasso 2015).\n"
        "    Experimental Chou-Talalay CI validation required before reporting.\n"
    )

    if predict_combinations:
        logger.info("[5/5] Drug combination synergy prediction (prior-based)...")
        syn_predictor    = SynergyPredictor()
        top_combinations = syn_predictor.predict_top_combinations(top_50[:30])
        if top_combinations:
            logger.info(
                "      Top combination: %s + %s (score: %.3f)",
                top_combinations[0].get("compound_a", "?"),
                top_combinations[0].get("compound_b", "?"),
                top_combinations[0].get("synergy_score", 0),
            )
            combination_report = (
                "⚠️  NOTE: Synergy scores are biological priors (Grasso 2015 DIPG4/13 logic),\n"
                "    not experimentally validated combination indices.\n\n"
                + "\n".join(
                    f"  {c.get('compound_a','?')} + {c.get('compound_b','?')}: "
                    f"score={c.get('synergy_score',0):.2f} | {c.get('rationale','')}"
                    for c in top_combinations[:5]
                )
            )
    else:
        logger.info("[5/5] Combination prediction skipped")

    # ── Final summary ─────────────────────────────────────────────────────────
    top_candidates = sorted(candidates, key=lambda c: c.get("score", 0), reverse=True)[:top_n]

    logger.info("\n" + "=" * 70)
    logger.info("SUPPLEMENTARY DIPG RESULTS — Top 10")
    logger.info("%-30s %6s %8s %8s", "Drug", "Score", "H3K27M", "Novel")
    logger.info("-" * 60)
    for c in top_candidates[:10]:
        name    = (c.get("name") or "?")[:29]
        score   = c.get("score", 0)
        h3k27m  = "Y" if c.get("dipg_components", {}).get("h3k27m_relevant") else "-"
        novel   = "Y" if c.get("dipg_components", {}).get("is_untested_dipg") else "-"
        logger.info("%-30s %6.3f %8s %8s", name, score, h3k27m, novel)

    return {
        # Core pipeline outputs (from ProductionPipeline)
        "hypotheses":          hypotheses,
        "stats":               stats,

        # Supplementary DIPG outputs
        "top_candidates":      top_candidates,
        "novel_candidates":    novel_candidates,
        "top_combinations":    top_combinations,

        # Reports
        "novelty_report":      novelty_report,
        "poly_report":         poly_report,
        "combination_report":  combination_report,

        "pipeline_stats": {
            "total_candidates":    len(candidates),
            "dipg_specialised":    is_dipg,
            "novel_count":         len(novel_candidates),
            "combinations_found":  len(top_combinations),
            "cmap_active":         False,   # Requires LINCS L1000 .gctx download
            "synergy_validated":   False,   # Requires experimental CI data
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
        },
    }


# ── CLI entry point ───────────────────────────────────────────────────────────

async def _cli_main(disease: str, output: Optional[str], top_n: int, combinations: bool):
    result = await run_dipg_pipeline(
        disease_name=disease,
        top_n=top_n,
        predict_combinations=combinations,
    )

    print("\n" + "=" * 70)
    print("HYPOTHESES")
    print("=" * 70)
    from backend.pipeline.discovery_pipeline import ProductionPipeline
    tmp = ProductionPipeline()
    print(tmp._hyp_gen.generate_report(result["hypotheses"]))

    print("\n" + "=" * 70)
    print("NOVELTY REPORT")
    print("=" * 70)
    print(result["novelty_report"])

    print("\n" + "=" * 70)
    print("COMBINATION REPORT")
    print("=" * 70)
    print(result["combination_report"])

    print("\n" + "=" * 70)
    print("PIPELINE STATUS")
    print("=" * 70)
    ps = result["pipeline_stats"]
    print(f"Active data streams ({len(ps['data_streams_active'])}):")
    for s in ps["data_streams_active"]:
        print(f"  ✅ {s}")
    print(f"\nPending data streams ({len(ps['data_streams_pending'])}):")
    for s in ps["data_streams_pending"]:
        print(f"  ⏳ {s}")

    if output:
        with open(output, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nResults saved to: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GBM/DIPG Drug Repurposing Pipeline v3.1")
    parser.add_argument("--disease",      default="dipg")
    parser.add_argument("--output",       default=None,  help="Save results to JSON file")
    parser.add_argument("--top_n",        default=20,    type=int)
    parser.add_argument("--combinations", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    asyncio.run(_cli_main(args.disease, args.output, args.top_n, args.combinations))
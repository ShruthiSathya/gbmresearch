"""
run_dipg_pipeline.py — GBM/DIPG Drug Repurposing Pipeline Runner  v3.0
=======================================================================
Integrates all pipeline components:
  1. BBB Filter              — Blood-brain barrier penetrance
  2. DIPG Specialization     — H3K27M/ACVR1 biology
  3. Scorer                  — Gene/pathway overlap scoring
  4. Polypharmacology        — Multi-target synergy + resistance
  5. TME Scorer        [NEW] — Tumour microenvironment scoring
  6. Synergy Predictor [NEW] — Drug combination prediction
  7. Cell Line Validator[NEW]— PDCL validation data
  8. In Silico Trial         — Virtual trial simulation

USAGE
-----
    python -m backend.pipeline.run_dipg_pipeline --disease glioblastoma
    python -m backend.pipeline.run_dipg_pipeline --disease dipg --combinations

INTEGRATION WITH ProductionPipeline
-------------------------------------
Add inside ProductionPipeline.analyze_disease() after polypharmacology step:

    if any(k in disease_name.lower() for k in
           ("dipg", "glioblastoma", "gbm", "h3k27m", "diffuse intrinsic pontine")):
        from .run_dipg_pipeline import run_dipg_pipeline
        dipg_result = await run_dipg_pipeline(
            pipeline=self, disease_data=disease_data, drugs_data=drugs_data,
            apply_bbb_penalty=True, top_n=top_n_for_trial,
            run_trial=True, predict_combinations=True,
        )
        return {
            "disease": disease_name, "disease_data": disease_data,
            **{k: dipg_result[k] for k in [
                "top_candidates","dipg_novel","excluded_bbb","trial_results",
                "trial_report","tme_summary","top_combinations",
                "combination_report","cellline_summary","pipeline_stats"
            ]}
        }
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
    pipeline,
    disease_data:         Dict,
    drugs_data:           List[Dict],
    exclude_low_bbb:      bool  = False,
    apply_bbb_penalty:    bool  = True,
    top_n:                int   = 20,
    run_trial:            bool  = True,
    predict_combinations: bool  = True,
    tme_weight:           float = 0.20,
    cellline_weight:      float = 0.15,
) -> Dict:
    """
    Run the full DIPG/GBM scoring stack.
    Returns dict with top_candidates, tme_summary, top_combinations,
    combination_report, cellline_summary, trial_report, pipeline_stats, etc.
    """
    from .bbb_filter import BBBFilter
    from .dipg_specialization import (
        DIPGSpecializedScorer, DIPG_CORE_GENES,
        DIPG_PATHWAY_WEIGHTS, augment_disease_data_for_dipg,
    )
    from .polypharmacology import PolypharmacologyScorer
    from .insilico_trial import InSilicoTrialSimulator
    from .tme_scorer import TMEScorer
    from .synergy_predictor import SynergyPredictor
    from .cellline_validator import CellLineValidator

    disease_name = disease_data.get("name", "unknown")
    is_dipg      = _is_dipg_or_gbm(disease_name)

    logger.info("=" * 70)
    logger.info("DIPG/GBM Pipeline v3.0 — %s", disease_name)
    logger.info("=" * 70)

    # [1/8] Augment disease data
    if is_dipg:
        logger.info("[1/8] Augmenting with DIPG/H3K27M gene sets...")
        disease_data = augment_disease_data_for_dipg(disease_data)
    else:
        logger.info("[1/8] Non-DIPG — skipping augmentation")

    # [2/8] Base scoring
    logger.info("[2/8] Base pipeline scoring...")
    candidates = await pipeline.generate_candidates(
        disease_data=disease_data, drugs_data=drugs_data,
        min_score=0.0, fetch_pubmed=False,
        use_tissue=False, use_polypharm=False,
    )
    logger.info("      %d candidates scored", len(candidates))

    # [3/8] BBB filter
    logger.info("[3/8] BBB penetrance filter...")
    bbb_filter = BBBFilter(
        penalise_low=apply_bbb_penalty, hard_exclude_mw=800.0,
        low_bbb_penalty=0.50, mod_bbb_penalty=0.85,
    )
    smiles_lookup = {d["name"]: d.get("smiles", "") for d in drugs_data}
    for c in candidates:
        c["smiles"] = smiles_lookup.get(c["name"], "")
    candidates, excluded_bbb = bbb_filter.filter_and_rank(
        candidates, apply_penalty=apply_bbb_penalty, exclude_low=exclude_low_bbb,
    )
    logger.info("      %d retained | %d excluded", len(candidates), len(excluded_bbb))

    # [4/8] DIPG specialization + polypharmacology
    if is_dipg:
        logger.info("[4/8] DIPG H3K27M/ACVR1 specialization...")
        dipg_scorer = DIPGSpecializedScorer(
            apply_bbb_penalty=apply_bbb_penalty,
            novelty_bonus=0.08, h3k27m_bonus=0.12, acvr1_bonus=0.10,
        )
        candidates = dipg_scorer.score_batch(candidates)
    else:
        logger.info("[4/8] Non-DIPG — skipping H3K27M")

    logger.info("      CNS polypharmacology (top 50)...")
    disease_genes = disease_data.get("genes", [])
    poly_scorer = PolypharmacologyScorer(disease_name=disease_name)
    top_50 = sorted(candidates, key=lambda c: c["score"], reverse=True)[:50]
    top_50 = poly_scorer.score_batch(top_50, disease_targets=disease_genes)
    for c in top_50:
        poly = c.get("polypharmacology_score", 0.0)
        if poly > 0:
            c["score"] = min(1.0, c["score"] + poly * 0.25)
    top_50_names = {c["name"] for c in top_50}
    rest = [c for c in candidates if c["name"] not in top_50_names]
    candidates = sorted(top_50 + rest, key=lambda c: c["score"], reverse=True)

    # [5/8] TME scoring
    logger.info("[5/8] Tumour microenvironment (TME) scoring...")
    tme_scorer = TMEScorer(disease=disease_name, tme_weight=tme_weight)
    candidates = tme_scorer.score_batch(candidates)
    tme_summary = tme_scorer.get_tme_summary(candidates)

    # [6/8] Cell line validation
    logger.info("[6/8] Patient-derived cell line validation...")
    cl_validator = CellLineValidator(
        disease=disease_name, validation_weight=cellline_weight, penalise_inactive=True,
    )
    candidates = cl_validator.validate_batch(candidates)
    cellline_summary = cl_validator.get_validated_summary(candidates)
    discordant = cl_validator.get_discordant_drugs(candidates)
    if discordant:
        logger.warning(
            "      %d discordant drugs (high score but inactive in PDCLs): %s",
            len(discordant),
            ", ".join(c.get("drug_name", c.get("name", "?")) for c in discordant[:5]),
        )

    # [7/8] Combination synergy prediction
    top_combinations = []
    combination_report = ""
    if predict_combinations:
        logger.info("[7/8] Drug combination synergy prediction...")
        syn_predictor = SynergyPredictor(
            disease=disease_name, min_synergy=0.50,
            max_candidates=min(len(candidates), 30),
        )
        top_combinations = syn_predictor.predict_top_combinations(candidates, top_k=20)
        combination_report = syn_predictor.generate_combination_report(top_combinations)
        logger.info("      %d combinations found", len(top_combinations))
        if top_combinations:
            top = top_combinations[0]
            logger.info("      Top: %s + %s (%.3f | %s)",
                        top["drug_a"], top["drug_b"],
                        top["synergy_score"], top["evidence_level"])
    else:
        logger.info("[7/8] Combination prediction skipped")

    # [8/8] In silico trial
    trial_results = []
    trial_report  = ""
    if run_trial:
        n_trial = min(top_n, 10)
        logger.info("[8/8] Virtual trials (top %d)...", n_trial)
        try:
            simulator = InSilicoTrialSimulator(disease=disease_name, n_patients=200)
            top_for_trial = candidates[:n_trial]
            for c in top_for_trial:
                c["disease_genes"] = disease_genes
            trial_results = await simulator.run_batch(top_for_trial)
            trial_report  = simulator.generate_trial_report(trial_results)
            trial_map = {r["drug_name"]: r for r in trial_results}
            for c in candidates:
                tr = trial_map.get(c.get("drug_name", c.get("name")))
                if tr:
                    c["trial_orr"]      = tr.get("orr", 0.0)
                    c["trial_p2_prob"]  = tr.get("phase2_success_probability", 0.0)
                    c["trial_priority"] = tr.get("priority", "")
        except Exception as e:
            logger.warning("Virtual trial failed (non-fatal): %s", e)
    else:
        logger.info("[8/8] Trials skipped")

    # Final ranking
    candidates.sort(key=lambda c: c.get("score", 0), reverse=True)
    top_candidates = candidates[:top_n]

    dipg_novel = [
        c for c in top_candidates
        if (
            c.get("dipg_components", {}).get("is_untested_dipg")
            or c.get("cellline_validation", {}).get("is_novel_opportunity")
        ) and c.get("score", 0) > 0.50
    ]

    # Log summary
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS — Top 10")
    logger.info("%-25s %6s %8s %8s %8s %8s",
                "Drug", "Score", "BBB", "TME", "PDCL", "Novel")
    logger.info("-" * 70)
    for c in top_candidates[:10]:
        name  = (c.get("drug_name") or c.get("name") or "?")[:24]
        score = c.get("score", 0)
        bbb   = c.get("bbb_penetrance", "?")[:7]
        tme   = c.get("tme_score", 0)
        pdcl  = c.get("cellline_validation", {}).get("dipg_activity", "?")[:7]
        novel = "Y" if c.get("cellline_validation", {}).get("is_novel_opportunity") else "-"
        logger.info("%-25s %6.3f %8s %8.3f %8s %8s", name, score, bbb, tme, pdcl, novel)

    return {
        "top_candidates":     top_candidates,
        "excluded_bbb":       excluded_bbb,
        "dipg_novel":         dipg_novel,
        "discordant":         discordant,
        "trial_report":       trial_report,
        "trial_results":      trial_results,
        "tme_summary":        tme_summary,
        "top_combinations":   top_combinations,
        "combination_report": combination_report,
        "cellline_summary":   cellline_summary,
        "pipeline_stats": {
            "total_scored":        len(candidates) + len(excluded_bbb),
            "bbb_excluded":        len(excluded_bbb),
            "after_bbb_filter":    len(candidates),
            "dipg_novel_count":    len(dipg_novel),
            "discordant_count":    len(discordant),
            "combinations_found":  len(top_combinations),
            "is_dipg_specialised": is_dipg,
            "tme_scoring":         True,
            "cellline_validation": True,
            "synergy_prediction":  predict_combinations,
        },
    }


# CLI
async def _cli_main(disease: str, output: Optional[str], top_n: int, combinations: bool):
    from backend.pipeline.production_pipeline import ProductionPipeline
    pipeline     = ProductionPipeline()
    disease_data = await pipeline.data_fetcher.fetch_disease_data(disease)
    if not disease_data:
        print(f"ERROR: Disease '{disease}' not found")
        return
    drugs_data = await pipeline.fetch_approved_drugs(limit=3000)
    result = await run_dipg_pipeline(
        pipeline=pipeline, disease_data=disease_data, drugs_data=drugs_data,
        top_n=top_n, predict_combinations=combinations,
    )
    print("\n" + result["tme_summary"])
    print("\n" + result["cellline_summary"])
    if combinations:
        print("\n" + result["combination_report"])
    if result["trial_report"]:
        print("\n" + result["trial_report"])
    if output:
        with open(output, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nResults saved to: {output}")
    await pipeline.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GBM/DIPG Drug Repurposing Pipeline v3.0")
    parser.add_argument("--disease",      default="glioblastoma")
    parser.add_argument("--output",       default=None)
    parser.add_argument("--top_n",        default=20, type=int)
    parser.add_argument("--combinations", action="store_true")
    parser.add_argument("--exclude_low_bbb", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%H:%M:%S")
    asyncio.run(_cli_main(args.disease, args.output, args.top_n, args.combinations))
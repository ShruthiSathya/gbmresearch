# GBM/DIPG Multi-Omic Drug Repurposing Pipeline

A computational drug repurposing framework for pediatric high-grade glioma (DIPG) and glioblastoma (GBM), integrating DepMap CRISPR essentiality, single-cell RNA-seq glioma stem cell profiling, protein-protein interaction networks, and patient genomic cohort data to prioritize drug combinations targeting H3K27M-mutant tumors.

**Author:** Shruthi Sathya Narayanan — Computer Science, UCLA  
**GitHub:** [github.com/ShruthiSathya/gbmresearch](https://github.com/ShruthiSathya/gbmresearch)

---

## Current Top Hypothesis

**Abemaciclib + Marizomib** (+ third agent from combinatorial screen)

| Metric | Value |
|--------|-------|
| Priority | HIGH |
| Confidence | 0.80 |
| DepMap essentiality (CDK4/6) | 0.93 (Chronos ≤ -1.0 in 52 GBM lines) |
| BBB penetrance | HIGH (both drugs — curated PK literature) |
| Target diversity | 1.00 (non-overlapping pathways) |
| Statistical significance | p = 1.16×10⁻⁴ ✅ |
| Drugs screened | 557 CNS/Oncology drugs |
| DIPG patient cohort | 184 samples (PNOC/PBTA, PedcBioPortal) |

### Statistical Finding

Fisher's exact test (two-sided) on H3K27M mutation vs. CDKN2A deletion co-occurrence in 184 DIPG/pediatric HGG samples (PNOC/PBTA cohort, PedcBioPortal):

| | CDKN2A deleted | CDKN2A WT |
|---|---|---|
| **H3K27M+** | 14 | 81 |
| **H3K27M−** | 36 | 53 |

p = 1.16×10⁻⁴ — **significant mutual exclusivity** (not co-occurrence). H3K27M mutation and CDKN2A deletion represent alternative oncogenic mechanisms in this cohort, consistent with Mackay et al. (2017) integrated molecular meta-analysis of 1,000 pediatric HGG samples. The pipeline uses this subgroup structure to stratify drug targeting: epigenetic drugs for H3K27M+ tumors (95/184, 52%), CDK4/6 inhibition for CDKN2A-deleted tumors (50/184, 27%), and combination therapy for the 14-sample double-hit subgroup.

---

## Data Sources

| Stream | Source | n |
|--------|--------|---|
| Drug library | OpenTargets API (EFO_0000519, EFO_0001422, EFO_0000618) | 557 drugs |
| CRISPR essentiality | DepMap CRISPRGeneEffect.csv (Broad Institute) | 52 GBM cell lines |
| Stem cell expression | GSE131928 (Neftel et al. 2019, single-cell RNA-seq) | 2,431 GSCs / 16,201 cells |
| PPI network | STRING-DB (confidence ≥ 800, Homo sapiens) | Live API |
| Genomic validation | PNOC/PBTA (PedcBioPortal, mutations + CNA + RNA z-scores) | 184 samples |

---

## Architecture

```
OpenTargets API (557 drugs)
        │
        ▼
┌───────────────────────────────────────────────┐
│              ProductionPipeline               │
│                                               │
│  DepMap CRISPR ──────────► depmap_score       │
│  Single-cell GSC ────────► tissue_expr_score  │
│  STRING-DB PPI ──────────► ppi_score          │
│  PNOC/PBTA genomic ──────► Fisher's p-value   │
│  BBB filter ─────────────► clinical failure   │
│                            penalty            │
│                                               │
│  Composite score → HypothesisGenerator        │
│  → Abemaciclib + Marizomib (p=1.16e-4)        │
└───────────────────────────────────────────────┘
```

### Module Overview

| Module | Function |
|--------|----------|
| `data_fetcher.py` | OpenTargets GraphQL pagination, null-safe API handling |
| `depmap_essentiality.py` | Broad Institute CRISPR Chronos score loading + GBM line filtering |
| `tissue_expression.py` | Single-cell GSC marker scoring (SOX2, NES, PROM1, CD44) |
| `ppi_network.py` | STRING-DB live PPI neighbor scoring |
| `statistical_validator.py` | Fisher's exact test with sentinel value handling (None/NaN/float) |
| `discovery_pipeline.py` | Main orchestrator — PedcBioPortalValidator + ProductionPipeline |
| `hypothesis_generator.py` | Externally-grounded confidence scoring (DepMap + BBB + diversity) |
| `bbb_filter.py` | Blood-brain barrier penetrance + GBM clinical failure penalty |
| `dipg_specialization.py` | H3K27M/ACVR1-specific scoring bonuses and novelty detection |
| `calibration.py` | Platt/isotonic score calibration with ECE and AUROC metrics |
| `trial_outcome_calibrator.py` | Calibrated P(Phase 2 success) from 29 real GBM trial outcomes |

---

## Confidence Score Methodology

The pipeline confidence score is **not** derived from the pipeline's own composite score. It is a weighted combination of three externally-grounded signals:

**Confidence = 0.45 × DepMap + 0.35 × BBB + 0.20 × Diversity**

- **DepMap component (0.45 weight):** Median Chronos CRISPR knockout score across 52 GBM cell lines (Broad Institute DepMap). Score 1.0 = Chronos ≤ −1.0 (essential gene); 0.1 = non-essential. External data, not derived from pipeline scoring.

- **BBB component (0.35 weight):** Blood-brain barrier penetrance from curated pharmacokinetic literature (`bbb_filter.py`). HIGH = 1.0, MODERATE = 0.6, LOW = 0.2. External source.

- **Diversity component (0.20 weight):** 1 − mean pairwise Jaccard overlap of drug target sets. 1.0 = completely non-overlapping targets (ideal for combination therapy). Computed, but independent of composite score.

This is a heuristic confidence measure. It is not a validated classifier and does not predict clinical trial success. For calibrated P(Phase 2 success), see `trial_outcome_calibrator.py`, which is trained on 29 published GBM Phase 2 trial outcomes.

---

## Limitations

**Active data streams (4):**
- DepMap CRISPR essentiality (Broad Institute)
- Single-cell RNA-seq GSC expression (GSE131928)
- OpenTargets drug-disease associations
- STRING-DB PPI network
- PedcBioPortal genomic cohort (PNOC/PBTA, n=184)

**Pending data streams (2):**
- **CMAP transcriptomic reversal:** architecturally implemented in `cmap_query.py` but requires LINCS L1000 `.gctx` file (~30GB). Download: https://clue.io/data/CMap2020. Until downloaded, CMAP contributes no signal.
- **Combination synergy:** `synergy_predictor.py` uses biological priors from Grasso et al. 2015 (DIPG4/13 Chou-Talalay logic). Experimental CI data from cell line screens required for validation.

**Wet lab validation required before clinical translation.** The top hypothesis (Abemaciclib + Marizomib) requires at minimum: IC50 curves in DIPG cell lines (DIPG4, DIPG13, SU-DIPG-VI), Chou-Talalay combination index assays, and ideally xenograft mouse model data.

---

## Installation

```bash
git clone https://github.com/ShruthiSathya/gbmresearch
cd gbmresearch
pip install -r requirements.txt
```

**Required data files** (not included — download separately):

```
data/raw_omics/
  CRISPRGeneEffect.csv     # DepMap — https://depmap.org/portal/download/all/
  Model.csv                # DepMap
  GSM3828673_10X_GBM_IDHwt_processed_TPM.tsv  # GEO GSE131928

data/validation/cbtn_genomics/
  mutations.txt            # PedcBioPortal PNOC/PBTA — Transposed Matrix format
  cna.txt
  rna_zscores.txt
```

**Run:**
```bash
python3 -m backend.pipeline.testing
```

---

## Key References

- Mackay A et al. (2017). Integrated Molecular Meta-Analysis of 1,000 Pediatric High-Grade and Diffuse Intrinsic Pontine Glioma. *Cancer Cell.* doi:10.1016/j.ccell.2017.08.017
- Neftel C et al. (2019). An Integrative Model of Cellular States, Plasticity, and Genetics for Glioblastoma. *Cell.* doi:10.1016/j.cell.2019.06.024
- Grasso CS et al. (2015). Functionally defined therapeutic targets in diffuse intrinsic pontine glioma. *Nature Medicine.* doi:10.1038/nm.3855
- Wu G et al. (2012). Somatic histone H3 alterations in pediatric diffuse intrinsic pontine gliomas. *Nature Genetics.* doi:10.1038/ng.1102
- Meyers RM et al. (2017). Computational correction of copy number effect improves specificity of CRISPR-Cas9 essentiality screens in cancer cells. *Nature Genetics.*
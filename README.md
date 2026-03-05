# 🧬 GBM/DIPG AI Discovery Engine v5.2

An unbiased, multi-omic drug repurposing pipeline for Glioblastoma (GBM) and Diffuse Intrinsic Pontine Glioma (DIPG). Instead of testing one drug at a time, this engine performs a massively parallel screen of the human pharmacopeia against the specific genetic vulnerabilities of high-grade gliomas — with a focus on eradicating the Glioma Stem Cell population that drives recurrence.

---

## 🎯 The Core Idea

Standard GBM drugs fail because they kill the bulk tumor but miss the **Glioma Stem Cells (GSCs)** — the 15% of cells that cause relapse. This pipeline finds drug combinations that specifically target those stem cells, enforcing **mechanistic diversity** so the tumor can't develop resistance by rewiring a single pathway.

The output is a **three-drug combination** where each drug attacks a completely independent biological node — making simultaneous bypass of all three statistically improbable.

---

## 🔬 How It Works

Four independent data streams are integrated in parallel:

| Data Source | What It Answers |
|---|---|
| **OpenTargets API** (~500 drugs) | What is the search space? Every CNS-active drug ever tested in humans. |
| **Broad DepMap CRISPR** (`CRISPRGeneEffect.csv`) | Which targets are genuinely essential? Drugs hitting genes whose knockout kills GBM cells get a major score boost. |
| **Single-Cell RNA-seq** (GSE131928, Neftel 2019) | Which drugs hit the stem cells? Only drugs targeting genes highly expressed in the GSC population are prioritized. |
| **STRING-DB PPI Network** | What are the escape routes? If a drug's target connects to a resistance gene, the pipeline detects it and ensures the combination blocks that bypass. |

Each candidate drug is scored across four components:
```
Final Score = (Tissue Expression × 0.40) + (DepMap Essentiality × 0.30) + (Escape Bypass × 0.20) + (PPI Network × 0.10)
```

The top candidates are then assembled into a triple combination that maximizes **target diversity** — the three drugs are required to have zero overlapping targets.

---

## 🏆 Current Top Result

**ABEMACICLIB + MARIZOMIB + [third drug]**

| Drug | Mechanism | Why |
|---|---|---|
| Abemaciclib | CDK4/6 inhibitor | Highest CNS penetrance of its class; CDK4 is CRISPR-essential in GBM lines |
| Marizomib | Proteasome inhibitor | Engineered specifically to cross the BBB, unlike bortezomib |

- **Confidence: 0.80**
- **DepMap essentiality: 0.93** (52 GBM cell lines, Broad Institute CRISPR data)
- **Target diversity: 1.00** (zero overlapping targets across the combination)

---

## 📂 Data Setup

Download these files and place them in the correct directories:
```
data/raw_omics/
  ├── CRISPRGeneEffect.csv        # depmap.org/portal/download/all/
  ├── Model.csv                   # same download
  └── GSM3828673_10X_GBM_IDHwt_processed_TPM.tsv  # NCBI GEO: GSE131928

data/validation/cbtn_genomics/
  ├── mutations.txt               # pedcbioportal.kidsfirstdrc.org
  ├── cna.txt
  └── rna_zscores.txt
```

The pipeline runs without these files using fallback values — so you can test the architecture immediately.

---

## 🛠️ Installation & Usage
```bash
git clone https://github.com/ShruthiSathya/gbmresearch.git
cd gbmresearch
pip install pandas numpy aiohttp scipy scikit-learn networkx cmapPy
python3 -m backend.pipeline.testing
```

---

## 🗂️ Module Overview
```
backend/pipeline/
  ├── data_fetcher.py            # OpenTargets GraphQL API with pagination
  ├── tissue_expression.py       # Single-cell GSC scoring
  ├── depmap_essentiality.py     # DepMap CRISPR Chronos scores
  ├── ppi_network.py             # STRING-DB protein interaction network
  ├── bbb_filter.py              # Blood-brain barrier penetrance filter
  ├── hypothesis_generator.py    # Triple combination assembly
  ├── statistical_validator.py   # Fisher's exact test for genomic co-occurrence
  ├── synergy_predictor.py       # Drug combination synergy prediction
  ├── dipg_specialization.py     # H3K27M/ACVR1 DIPG-specific scoring
  ├── tme_scorer.py              # Tumor microenvironment scoring
  └── calibration.py             # Score calibration against real trial outcomes
```

---

## 📊 Confidence Score

The confidence score is built from three **externally-grounded** signals — not from the pipeline's own composite score:

1. **DepMap Essentiality (45%)** — Broad Institute CRISPR Chronos scores
2. **BBB Penetrance (35%)** — Curated pharmacokinetic literature
3. **Target Diversity (20%)** — Jaccard distance between drug target sets

---

## 📚 Key References

- Neftel et al. (2019) *Cell* — Single-cell GBM atlas (GSE131928)
- Mackay et al. (2017) *Cancer Cell* — DIPG molecular landscape
- Grasso et al. (2015) *Nature Medicine* — DIPG therapeutic targets
- Meyers et al. (2017) *Nature Genetics* — DepMap Chronos method
- Szklarczyk et al. (2021) *Nucleic Acids Research* — STRING-DB v11.5

---

## 🧪 Latest Run Output

**Query:** DIPG | **Drugs screened:** 496 unique CNS/Oncology compounds | **Cell lines:** 52 GBM (DepMap) | **Stem cells identified:** 2,431 / 16,201

### Top Hypothesis: ABEMACICLIB + MARIZOMIB + CILENGITIDE*

| Metric | Value |
|---|---|
| Confidence | 0.80 |
| DepMap essentiality | 0.93 (52 GBM cell lines) |
| BBB penetrance | 0.50 |
| Target diversity | 1.00 (zero overlapping targets) |
| Targets | CDK4, CDK6, PSMA2, PSMB1, PSMB2 |
| Statistical validation | N/A — cohort too small (n=6) |

*\*Cilengitide failed Phase 3 GBM trial (CENTRIC 2015). Flagged for replacement in next run.* 
____________________________

## ⚠️ Disclaimer

This pipeline generates computational hypotheses for research purposes. Results require wet-lab validation before any clinical interpretation.
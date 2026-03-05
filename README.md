# GBM/DIPG AI Discovery Engine (v5.1) 🧬

An unbiased, multi-omic drug discovery pipeline designed to identify synergistic therapeutic combinations for Glioblastoma (GBM) and Diffuse Intrinsic Pontine Glioma (DIPG).

## 🚀 Overview
Unlike traditional "single-target" approaches, this engine performs a massively parallel screen of the human pharmacopeia against the specific genetic vulnerabilities of high-grade gliomas. It integrates four high-dimensional data streams to calculate a **Composite Confidence Score (0.0 - 1.0)** for novel drug combinations.

### Key Features:
- **Unbiased Screening:** Dynamically fetches ~500-1000 CNS-active drugs via the OpenTargets GraphQL API.
- **Stem-Cell Targeting:** Uses 10X Single-Cell RNA-seq (GSE131928) to prioritize drugs that eradicate Glioma Stem Cells (GSCs).
- **CRISPR Validation:** Cross-references candidates with Broad Institute DepMap Chronos scores to ensure target essentiality.
- **Mechanism Diversity:** Enforces a multi-node blockade, ensuring combinations attack distinct biological pathways (e.g., Cell Cycle + Proteasome + Microenvironment).

## 📂 Data Setup
Due to file size limits, raw omics datasets are not included in this repository. To run the pipeline, populate the `/data/raw_omics/` directory with the following:

1. **DepMap CRISPR Data:** Download `CRISPRGeneEffect.csv` and `Model.csv` from [depmap.org](https://depmap.org/portal/download/all/).
2. **Single-Cell Atlas:** Download `GSM3828673_10X_GBM_IDHwt_processed_TPM.tsv` from [NCBI GEO (GSE131928)](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE131928).
3. **Genomic Validation:** (Optional) PedcBioPortal mutation and CNA files in `.txt` format.

## 🛠️ Installation
```bash
# Clone the repository
git clone [https://github.com/ShruthiSathya/gbmresearch.git](https://github.com/ShruthiSathya/gbmresearch.git)
cd gbmresearch

# Install dependencies
pip install pandas numpy aiohttp scipy cmapPy
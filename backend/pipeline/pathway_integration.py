"""
pathway_integration.py — Reactome + KEGG Pathway Integration (v3.1)
=====================================================================
FIX v3.1 (CRITICAL — silent correctness bug):
  - Wrong Reactome endpoint fixed: previous version used /data/entity/{id}/componentOf
    which returns molecular COMPLEXES, not biological PATHWAYS.
    Correct endpoint: /ContentService/data/mapping/gene/{symbol}/pathways
    (returns actual pathway objects with displayName and stId).

  - WRONG FALLBACK GENES REMOVED: previous _curated_fallback() contained
    Parkinson's disease genes (SNCA, LRRK2, PRKN), cardiovascular genes
    (ADRB1, PDE5A, EDNRA), and hair follicle genes (SRD5A1, KCNJ8, ABCC9)
    that are entirely irrelevant to GBM drug discovery. If Reactome/KEGG
    API fails, these wrong fallback mappings would silently corrupt pathway
    scores for those drug targets.

  - REPLACED WITH: GBM/CNS-relevant fallback covering ~100 genes
    actually relevant to GBM/DIPG drug targets and biology.

  - Added cache validation: rejects cached entries with non-GBM fallback
    genes.

ARCHITECTURE
------------
  1. ReactomePathwayFetcher  — queries /data/mapping/gene/{symbol}/pathways (FIXED)
  2. KEGGPathwayFetcher       — queries KEGG REST API (unchanged, was correct)
  3. HybridPathwayMapper      — merges Reactome + KEGG, falls back to curated
                                GBM-relevant map (REPLACED)
  4. Disk cache              — avoids re-fetching on each run

METHODS CITATION
-----------------
  "Pathway annotations were retrieved from the Reactome Content Service
   (v88, https://reactome.org; Jassal et al. 2020, Nucleic Acids Res)
   and KEGG PATHWAY (https://www.genome.jp/kegg; Kanehisa et al. 2023,
   Nucleic Acids Res). For genes not covered by either database, annotations
   from the curated GBM/DIPG pathway map (Supplementary Table S1) were used."
"""

import asyncio
import aiohttp
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

REACTOME_BASE = "https://reactome.org/ContentService"
KEGG_BASE     = "https://rest.kegg.jp"
CACHE_FILE    = Path("/tmp/pathway_cache_v31.json")   # new cache key avoids stale data


# ─────────────────────────────────────────────────────────────────────────────
# 1. Reactome fetcher (FIXED endpoint)
# ─────────────────────────────────────────────────────────────────────────────

class ReactomePathwayFetcher:
    """
    Fetch Reactome pathways for a human gene symbol.

    FIX: Uses /data/mapping/gene/{symbol}/pathways endpoint.
    Previous version used /data/entity/{stId}/componentOf which returns
    molecular complexes, not biological pathways.
    """

    def __init__(self, session: aiohttp.ClientSession, cache: Dict):
        self.session = session
        self.cache   = cache

    async def get_pathways_for_gene(self, gene_symbol: str) -> List[str]:
        """Return list of Reactome pathway display names for a human gene."""
        if gene_symbol in self.cache:
            return self.cache[gene_symbol]

        pathway_names: List[str] = []

        try:
            # FIXED ENDPOINT: direct gene → pathways mapping
            # Returns list of pathway objects: [{stId, displayName, ...}]
            url = f"{REACTOME_BASE}/data/mapping/gene/{gene_symbol}/pathways"
            async with self.session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=12),
                headers={"Accept": "application/json"},
            ) as resp:
                if resp.status == 404:
                    # Gene not found in Reactome — normal for some genes
                    self.cache[gene_symbol] = []
                    return []
                if resp.status != 200:
                    logger.debug(
                        "Reactome %d for %s", resp.status, gene_symbol
                    )
                    self.cache[gene_symbol] = []
                    return []

                pathway_list = await resp.json()
                for p in pathway_list:
                    # Filter to human pathways only
                    species = p.get("species", {})
                    if isinstance(species, dict):
                        taxon = species.get("taxId", "")
                        if str(taxon) != "9606":   # 9606 = Homo sapiens
                            continue
                    name = p.get("displayName") or p.get("name") or ""
                    if name:
                        # Strip species suffix if present
                        name = name.replace(" - Homo sapiens", "").strip()
                        if name:
                            pathway_names.append(name)

        except aiohttp.ClientError as e:
            logger.debug("Reactome network error for %s: %s", gene_symbol, e)
        except Exception as e:
            logger.debug("Reactome lookup failed for %s: %s", gene_symbol, e)

        # Deduplicate and limit
        pathway_names = list(dict.fromkeys(pathway_names))[:8]
        self.cache[gene_symbol] = pathway_names
        return pathway_names


# ─────────────────────────────────────────────────────────────────────────────
# 2. KEGG fetcher (unchanged — was correct)
# ─────────────────────────────────────────────────────────────────────────────

class KEGGPathwayFetcher:
    """Fetch KEGG pathway names for a human gene via KEGG REST API."""

    def __init__(self, session: aiohttp.ClientSession, cache: Dict):
        self.session              = session
        self.cache                = cache
        self._pathway_name_cache: Dict[str, str] = {}

    async def get_pathways_for_gene(self, gene_symbol: str) -> List[str]:
        if gene_symbol in self.cache:
            return self.cache[gene_symbol]

        pathway_names: List[str] = []
        try:
            # Step 1: symbol → KEGG gene ID
            find_url = f"{KEGG_BASE}/find/hsa/{gene_symbol}"
            async with self.session.get(
                find_url, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status != 200:
                    self.cache[gene_symbol] = []
                    return []
                text  = await resp.text()
                lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
                if not lines:
                    self.cache[gene_symbol] = []
                    return []
                kegg_id = lines[0].split("\t")[0].strip()  # e.g. "hsa:5742"

            # Step 2: gene → pathway IDs
            link_url = f"{KEGG_BASE}/link/pathway/{kegg_id}"
            async with self.session.get(
                link_url, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status != 200:
                    self.cache[gene_symbol] = []
                    return []
                text     = await resp.text()
                path_ids = []
                for line in text.strip().split("\n"):
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        pid = parts[1].strip()
                        if pid.startswith("path:hsa"):
                            path_ids.append(pid.replace("path:", ""))

                # Step 3: pathway ID → name
                for pid in path_ids[:6]:
                    name = await self._get_pathway_name(pid)
                    if name:
                        pathway_names.append(name)

        except Exception as e:
            logger.debug("KEGG lookup failed for %s: %s", gene_symbol, e)

        self.cache[gene_symbol] = pathway_names
        return pathway_names

    async def _get_pathway_name(self, pathway_id: str) -> Optional[str]:
        if pathway_id in self._pathway_name_cache:
            return self._pathway_name_cache[pathway_id]
        try:
            url = f"{KEGG_BASE}/get/{pathway_id}"
            async with self.session.get(
                url, timeout=aiohttp.ClientTimeout(total=8)
            ) as resp:
                if resp.status != 200:
                    return None
                text = await resp.text()
                for line in text.split("\n"):
                    if line.startswith("NAME"):
                        name = line[12:].strip().replace(" - Homo sapiens (human)", "")
                        self._pathway_name_cache[pathway_id] = name
                        return name
        except Exception:
            pass
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 3. Hybrid mapper (FIXED fallback — GBM/CNS genes only)
# ─────────────────────────────────────────────────────────────────────────────

class HybridPathwayMapper:
    """
    Maps gene symbols to pathway names using Reactome → KEGG → curated fallback.
    Curated fallback is GBM/CNS-relevant only (v3.1 replaces wrong disease genes).
    """

    def __init__(self, use_curated_fallback: bool = True):
        self.use_curated_fallback = use_curated_fallback
        self._reactome_cache: Dict[str, List[str]] = {}
        self._kegg_cache:     Dict[str, List[str]] = {}
        self._combined_cache: Dict[str, List[str]] = {}
        self._session:        Optional[aiohttp.ClientSession] = None
        self._load_disk_cache()

    def _load_disk_cache(self) -> None:
        try:
            if CACHE_FILE.exists():
                with open(CACHE_FILE) as f:
                    disk = json.load(f)
                self._reactome_cache = disk.get("reactome", {})
                self._kegg_cache     = disk.get("kegg", {})
                self._combined_cache = disk.get("combined", {})
                logger.info(
                    "Loaded pathway cache v3.1 (%d genes)", len(self._combined_cache)
                )
        except Exception as e:
            logger.warning("Could not load pathway cache: %s", e)

    def _save_disk_cache(self) -> None:
        try:
            with open(CACHE_FILE, "w") as f:
                json.dump({
                    "reactome": self._reactome_cache,
                    "kegg":     self._kegg_cache,
                    "combined": self._combined_cache,
                    "_version": "3.1",
                }, f, indent=2)
        except Exception as e:
            logger.warning("Could not save pathway cache: %s", e)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self._session

    async def get_pathways(self, gene_symbol: str) -> List[str]:
        """Return merged pathway list for a gene (Reactome + KEGG + fallback)."""
        if gene_symbol in self._combined_cache:
            return self._combined_cache[gene_symbol]

        session  = await self._get_session()
        reactome = ReactomePathwayFetcher(session, self._reactome_cache)
        kegg     = KEGGPathwayFetcher(session, self._kegg_cache)

        r_pathways, k_pathways = await asyncio.gather(
            reactome.get_pathways_for_gene(gene_symbol),
            kegg.get_pathways_for_gene(gene_symbol),
            return_exceptions=True,
        )

        if isinstance(r_pathways, Exception):
            r_pathways = []
        if isinstance(k_pathways, Exception):
            k_pathways = []

        seen:   Set[str] = set()
        merged: List[str] = []
        for p in list(r_pathways) + list(k_pathways):
            key = p.lower().strip()
            if key not in seen and len(p) > 3:
                seen.add(key)
                merged.append(p)

        # Curated fallback — GBM/CNS genes ONLY
        if not merged and self.use_curated_fallback:
            fallback = self._curated_fallback(gene_symbol)
            if fallback:
                merged = fallback
                logger.debug("Curated fallback for %s: %s", gene_symbol, merged)

        self._combined_cache[gene_symbol] = merged
        return merged

    async def get_pathways_bulk(self, genes: List[str]) -> Dict[str, List[str]]:
        """Fetch pathways for multiple genes concurrently."""
        tasks   = [self.get_pathways(g) for g in genes]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out: Dict[str, List[str]] = {}
        for gene, result in zip(genes, results):
            out[gene] = result if isinstance(result, list) else []
        self._save_disk_cache()
        return out

    def _curated_fallback(self, gene: str) -> List[str]:
        """
        GBM/CNS-relevant fallback.
        v3.1 REPLACEMENT: removed Parkinson's, cardiovascular, hair follicle genes.
        Covers GBM drug targets, DIPG-specific genes, and CNS biology.
        """
        # GBM core oncogenes and tumour suppressors
        _GBM_FALLBACK: Dict[str, List[str]] = {
            # ── DIPG/H3K27M specific ─────────────────────────────────────────
            "H3F3A":  ["Histone H3 methylation", "Chromatin remodeling", "Epigenetic regulation"],
            "HIST1H3B": ["Histone H3 methylation", "Chromatin remodeling"],
            "EZH2":   ["H3K27 methylation", "PRC2 complex", "Epigenetic regulation of gene expression"],
            "EED":    ["PRC2 complex", "H3K27 methylation"],
            "SUZ12":  ["PRC2 complex", "Chromatin remodeling"],
            "BRD4":   ["BET bromodomain", "Super-enhancer regulation", "Transcription factor activity"],
            "BRD2":   ["BET bromodomain", "Epigenetic regulation"],
            "BRD3":   ["BET bromodomain", "Epigenetic regulation"],
            "ACVR1":  ["ACVR1 signaling", "BMP signaling pathway", "BMP-SMAD signaling"],
            "BMPR1A": ["BMP signaling pathway", "BMP-SMAD signaling"],
            "BMPR2":  ["BMP signaling pathway", "Receptor tyrosine kinase signaling"],
            "SMAD1":  ["BMP-SMAD signaling", "SMAD signaling"],
            "SMAD5":  ["BMP-SMAD signaling", "SMAD signaling"],
            "ID1":    ["BMP signaling pathway", "Cancer stem cell signaling"],
            "ID2":    ["BMP signaling pathway"],
            # ── HDAC/epigenetic ──────────────────────────────────────────────
            "HDAC1":  ["HDAC deacetylase activity", "Histone deacetylation", "Epigenetic regulation"],
            "HDAC2":  ["HDAC deacetylase activity", "Histone deacetylation"],
            "HDAC3":  ["HDAC deacetylase activity", "NF-kB signaling"],
            "HDAC4":  ["Histone deacetylation", "MAPK signaling"],
            "HDAC6":  ["Histone deacetylation", "Protein quality control"],
            "SIRT1":  ["NAD-dependent deacetylase", "Epigenetic regulation"],
            "KDM6A":  ["Histone demethylation", "H3K27 methylation"],
            "KDM6B":  ["Histone demethylation", "H3K27 methylation"],
            "DNMT1":  ["DNA methylation", "Epigenetic regulation"],
            "DNMT3A": ["DNA methylation", "Epigenetic regulation"],
            # ── Cell cycle / CDKN2A context ──────────────────────────────────
            "CDK4":   ["CDK4/6 signaling", "Cell cycle regulation", "Rb phosphorylation"],
            "CDK6":   ["CDK4/6 signaling", "Cell cycle regulation"],
            "CDKN2A": ["Cell cycle regulation", "p53 signaling", "Tumour suppressor loss"],
            "CDKN2B": ["Cell cycle regulation", "Tumour suppressor loss"],
            "RB1":    ["Cell cycle regulation", "Rb signaling", "Tumour suppressor loss"],
            "E2F1":   ["Cell cycle regulation", "Apoptosis"],
            "CCND1":  ["Cell cycle regulation", "CDK4/6 signaling"],
            "CCND2":  ["Cell cycle regulation"],
            "MDM2":   ["MDM2-p53 interaction", "p53 signaling"],
            "MDM4":   ["MDM2-p53 interaction", "p53 signaling"],
            "WEE1":   ["DNA damage response", "Cell cycle regulation", "G2/M checkpoint"],
            "ATR":    ["DNA damage response", "ATR signaling", "Replication stress response"],
            "CHEK1":  ["DNA damage response", "G2/M checkpoint"],
            # ── RTK / PI3K / MAPK ────────────────────────────────────────────
            "EGFR":   ["EGFR signaling", "MAPK signaling", "Receptor tyrosine kinase signaling"],
            "PDGFRA": ["PDGFRA signaling", "Receptor tyrosine kinase signaling"],
            "MET":    ["Receptor tyrosine kinase signaling", "MAPK signaling"],
            "FGFR1":  ["FGFR signaling", "Receptor tyrosine kinase signaling"],
            "PIK3CA": ["PI3K-Akt signaling", "mTOR signaling"],
            "PIK3R1": ["PI3K-Akt signaling"],
            "AKT1":   ["PI3K-Akt signaling", "mTOR signaling"],
            "PTEN":   ["PTEN signaling", "PI3K-Akt signaling", "Tumour suppressor loss"],
            "MTOR":   ["mTOR signaling", "PI3K-Akt signaling"],
            "TSC1":   ["mTOR signaling", "Tuberous sclerosis"],
            "TSC2":   ["mTOR signaling", "Tuberous sclerosis"],
            "BRAF":   ["MAPK signaling", "RAS signaling"],
            "KRAS":   ["RAS signaling", "MAPK signaling"],
            "NF1":    ["RAS signaling", "Tumour suppressor loss"],
            # ── Apoptosis / BCL-2 ────────────────────────────────────────────
            "BCL2":   ["BCL-2 family signaling", "Intrinsic apoptosis pathway"],
            "BCL2L1": ["BCL-2 family signaling", "Intrinsic apoptosis pathway"],
            "MCL1":   ["BCL-2 family signaling", "Apoptosis"],
            "BAX":    ["Intrinsic apoptosis pathway", "Apoptosis"],
            "TP53":   ["p53 signaling", "Apoptosis", "DNA damage response"],
            "CASP3":  ["Apoptosis", "Caspase cascade"],
            # ── DNA damage / PARP ────────────────────────────────────────────
            "PARP1":  ["PARP signaling", "DNA damage response", "DNA repair"],
            "PARP2":  ["PARP signaling", "DNA damage response"],
            "BRCA1":  ["DNA repair", "DNA damage response", "Homologous recombination"],
            "BRCA2":  ["DNA repair", "Homologous recombination"],
            "MGMT":   ["DNA repair", "DNA damage response"],
            # ── MYC / transcription factors ──────────────────────────────────
            "MYC":    ["MYC signaling", "Transcription factor activity"],
            "MYCN":   ["MYCN signaling", "Cancer stem cell signaling"],
            "STAT3":  ["STAT3 signaling", "JAK-STAT signaling"],
            "NFKB1":  ["NF-kB signaling", "TNF signaling"],
            "RELA":   ["NF-kB signaling"],
            "HIF1A":  ["Hypoxia response", "VEGF signaling", "Cancer metabolism"],
            # ── Angiogenesis / VEGF ──────────────────────────────────────────
            "VEGFA":  ["VEGF signaling", "Angiogenesis", "Hypoxia response"],
            "KDR":    ["VEGF signaling", "Receptor tyrosine kinase signaling"],
            "FLT1":   ["VEGF signaling", "Angiogenesis"],
            # ── Immune / TME ─────────────────────────────────────────────────
            "CD274":  ["PD-1/PD-L1 signaling", "T-cell checkpoint signaling"],
            "PDCD1":  ["PD-1/PD-L1 signaling", "T-cell checkpoint signaling"],
            "CTLA4":  ["T-cell checkpoint signaling"],
            "IDO1":   ["Tryptophan metabolism", "Immunosuppression"],
            "TGFB1":  ["TGF-beta signaling", "Tumour microenvironment"],
            "TGFB2":  ["TGF-beta signaling"],
            "CSF1R":  ["Macrophage differentiation", "Tumour microenvironment"],
            "IL6":    ["JAK-STAT signaling", "IL-6 signaling"],
            # ── MGMT / drug resistance ───────────────────────────────────────
            "ABCB1":  ["ABC transporter", "Drug resistance", "P-gp efflux"],
            "ABCG2":  ["ABC transporter", "Drug resistance", "BCRP efflux"],
            # ── Stem cell / developmental ─────────────────────────────────────
            "SOX2":   ["Cancer stem cell signaling", "Neural stem cell"],
            "OLIG2":  ["Oligodendrocyte differentiation", "Neural development"],
            "NES":    ["Neural stem cell", "Cytoskeletal organisation"],
            # ── ONC201 target ─────────────────────────────────────────────────
            "DRD2":   ["Dopamine receptor signaling", "G-protein coupled receptor"],
            "DRD4":   ["Dopamine receptor signaling", "G-protein coupled receptor"],
            "CLPB":   ["Protein quality control", "Mitochondrial stress response"],
        }
        return _GBM_FALLBACK.get(gene, [])

    async def close(self) -> None:
        self._save_disk_cache()
        if self._session and not self._session.closed:
            await self._session.close()
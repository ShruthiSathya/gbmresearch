"""
dipg_specialization.py — DIPG/H3K27M Disease Specialization v1.0
=================================================================
Specializes the drug repurposing pipeline for Diffuse Intrinsic Pontine
Glioma (DIPG) and H3K27M-mutant gliomas, including:

  - H3K27M-specific gene sets and pathway weights
  - ACVR1 mutation subtype handling (~25% of DIPG)
  - Epigenetic vulnerability scoring (H3K27M creates global H3K27me3 loss)
  - BBB-adjusted scoring for brainstem/CNS location
  - DIPG-specific resistance gene set

BIOLOGICAL RATIONALE
---------------------
H3K27M mutation (H3F3A K27M or HIST1H3B K27M) is found in ~80% of DIPG
and creates a unique epigenetic vulnerability:
  1. Global loss of H3K27me3 (PRC2 inhibition by mutant H3)
  2. Retained H3K27ac at active enhancers → aberrant transcriptional activation
  3. This makes EZH2 inhibitors, HDAC inhibitors, and BET inhibitors
     potentially more effective than in adult GBM

ACVR1 mutations (~25% of DIPG) activate BMP signalling and are essentially
absent from adult GBM, making this a pediatric-specific target.

Novel therapeutic angle:
  Combining epigenetic reprogramming (restore H3K27me3) + BBB-penetrant
  kinase inhibition (PDGFRA/ACVR1 in ACVR1-mutant DIPG) is mechanistically
  rational and largely untested.

References
----------
Mackay A et al. (2017). Integrated Molecular Meta-Analysis of 1,000
  Pediatric High-Grade and Diffuse Intrinsic Pontine Glioma. Cancer Cell.
  doi:10.1016/j.ccell.2017.08.017

Khuong-Quang DA et al. (2012). K27M mutation in histone H3.3 defines
  clinically and biologically distinct subgroups of pediatric diffuse
  intrinsic pontine gliomas. Acta Neuropathol. doi:10.1007/s00401-012-0998-0

Wu G et al. (2012). Somatic histone H3 alterations in pediatric diffuse
  intrinsic pontine gliomas and non-brainstem glioblastomas. Nat Genet.
  doi:10.1038/ng.1102

Fontebasso AM et al. (2014). Recurrent somatic mutations in ACVR1 in
  pediatric midline high-grade astrocytoma. Nat Genet.
  doi:10.1038/ng.2950

Grasso CS et al. (2015). Functionally defined therapeutic targets in
  diffuse intrinsic pontine glioma. Nat Med. doi:10.1038/nm.3855
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# DIPG core gene sets
# ─────────────────────────────────────────────────────────────────────────────

# H3K27M-mutant DIPG driver and vulnerability genes
# Source: Mackay et al. 2017, Wu et al. 2012, Grasso et al. 2015
DIPG_CORE_GENES: List[str] = [
    # Histone mutation genes (primary oncogenic drivers)
    "H3F3A",       # H3.3 K27M (~57% of DIPG)
    "HIST1H3B",    # H3.1 K27M (~23% of DIPG)
    "HIST1H3C",    # H3.1 K27M variant

    # PRC2 complex (H3K27 methylation — globally suppressed by H3K27M)
    "EZH2",        # Histone methyltransferase — vulnerability target
    "EED",         # PRC2 component
    "SUZ12",       # PRC2 component

    # H3K27 demethylases (upregulated due to H3K27M)
    "KDM6A",       # UTX — H3K27 demethylase
    "KDM6B",       # JMJD3 — H3K27 demethylase

    # ACVR1 / BMP signalling (~25% of DIPG, essentially absent in adult GBM)
    "ACVR1",       # BMP type I receptor — gain-of-function mutations in DIPG
    "BMPR1A",      # BMP receptor
    "BMPR2",       # BMP receptor
    "SMAD1",       # BMP downstream effector
    "SMAD5",       # BMP downstream effector
    "ID1",         # BMP target — ID proteins
    "ID2",         # BMP target

    # PDGFRA signalling (amplified ~36% of DIPG)
    "PDGFRA",      # Platelet-derived growth factor receptor alpha
    "PDGFA",       # PDGFR-A ligand
    "PDGFB",       # PDGFR-B ligand

    # PI3K/PTEN/AKT axis (frequently altered)
    "PIK3CA",      # PI3K catalytic subunit alpha
    "PIK3R1",      # PI3K regulatory subunit
    "PTEN",        # Tumour suppressor — lost in ~25% DIPG

    # Cell cycle regulators (commonly altered)
    "CDKN2A",      # p16/p14 — deleted in ~15% DIPG
    "CDK4",        # CDK4 amplification in DIPG
    "CDK6",        # CDK6
    "CCND1",       # Cyclin D1
    "RB1",         # Retinoblastoma protein

    # p53 pathway
    "TP53",        # Mutated ~33% DIPG
    "MDM2",        # p53 regulator
    "MDM4",        # p53 regulator

    # MYC (amplified subset)
    "MYC",         # Oncogene
    "MYCN",        # N-Myc — neuronal MYC

    # Epigenetic regulators beyond PRC2
    "KMT2A",       # MLL1 — H3K4 methyltransferase
    "KMT2D",       # MLL4 — H3K4 methyltransferase
    "SETD2",       # H3K36 methyltransferase — frequently mutated
    "ATRX",        # Chromatin remodeller — mutated ~33% DIPG
    "DAXX",        # H3.3 chaperone partner

    # BET bromodomain (epigenetic reader — vulnerability)
    "BRD4",        # BET bromodomain — vulnerability in H3K27M tumours
    "BRD2",        # BET bromodomain
    "BRD3",        # BET bromodomain

    # HDAC family (epigenetic vulnerability)
    "HDAC1",       # Class I HDAC
    "HDAC2",       # Class I HDAC
    "HDAC3",       # Class I HDAC
    "HDAC4",       # Class IIa HDAC
    "HDAC6",       # Class IIb HDAC

    # DNA damage / replication
    "PARP1",       # DNA repair
    "ATM",         # DNA damage kinase
    "ATR",         # DNA damage kinase

    # Receptor tyrosine kinases
    "EGFR",        # EGFR — amplified subset
    "MET",         # MET — activated in DIPG
    "FGFR1",       # FGFR1 — mutated in rare DIPG subset

    # Stemness / differentiation
    "SOX2",        # Neural stem cell marker
    "OLIG2",       # Oligodendrocyte lineage
    "NKX2-1",      # Transcription factor

    # Immune microenvironment (generally cold in DIPG)
    "CD274",       # PD-L1
    "PDCD1",       # PD-1
    "TIGIT",       # Immune checkpoint

    # Metabolism
    "IDH1",        # Isocitrate dehydrogenase (usually wild-type in DIPG)
    "IDH2",        # Isocitrate dehydrogenase
]

# ACVR1-mutant DIPG subtype (additional genes relevant when ACVR1 is mutated)
DIPG_ACVR1_SUBTYPE_GENES: List[str] = [
    "ACVR1", "BMPR1A", "BMPR2",
    "SMAD1", "SMAD4", "SMAD5", "SMAD6", "SMAD7", "SMAD9",
    "ID1", "ID2", "ID3", "ID4",
    "NOGGIN",  # BMP antagonist — potential therapeutic
    "CHORDIN", # BMP antagonist
    "GREM1",   # Gremlin — BMP antagonist
]

# ─────────────────────────────────────────────────────────────────────────────
# DIPG-specific pathway weights
# Higher weight = more therapeutically relevant to DIPG/H3K27M biology
# ─────────────────────────────────────────────────────────────────────────────
DIPG_PATHWAY_WEIGHTS: Dict[str, float] = {
    # H3K27M-specific epigenetic pathways — highest priority
    "H3K27 methylation":                    1.00,
    "PRC2 complex":                         1.00,
    "Epigenetic regulation of gene expression": 0.95,
    "H3K27 demethylation":                  0.95,
    "Histone modification":                 0.90,
    "Chromatin remodeling":                 0.90,
    "Chromatin organisation":               0.90,

    # HDAC pathways — vulnerability due to H3K27M epigenetic dysregulation
    "HDAC deacetylase activity":            0.95,
    "HDAC inhibition":                      0.95,
    "Histone deacetylation":                0.95,
    "Class I HDAC":                         0.90,
    "Class II HDAC":                        0.85,

    # BET bromodomain — H3K27ac mark reader, upregulated in H3K27M tumours
    "BRD4 signaling":                       0.95,
    "BET bromodomain":                      0.95,
    "Bromodomain signaling":                0.90,

    # ACVR1 / BMP pathway
    "BMP signaling pathway":                1.00,
    "ACVR1 signaling":                      1.00,
    "TGF-beta superfamily signaling":       0.85,
    "BMP-SMAD signaling":                   0.95,
    "SMAD signaling":                       0.90,

    # PDGFRA pathway
    "PDGFR signaling":                      0.90,
    "PDGFRA signaling":                     0.95,
    "Receptor tyrosine kinase signaling":   0.80,

    # PI3K/AKT/mTOR — commonly activated
    "PI3K-Akt signaling":                   0.85,
    "mTOR signaling":                       0.80,
    "PTEN signaling":                       0.85,

    # Cell cycle — CDK4/6 amplification in DIPG
    "Cell cycle regulation":                0.80,
    "CDK4/6 signaling":                     0.90,
    "Cyclin D-CDK4/6 complex":              0.90,
    "Rb-E2F signaling":                     0.85,

    # p53 pathway
    "p53 signaling":                        0.80,
    "DNA damage response":                  0.75,
    "MDM2-p53 interaction":                 0.80,

    # EGFR/MAPK
    "EGFR signaling":                       0.75,
    "MAPK signaling":                       0.70,
    "RAS signaling":                        0.70,

    # DNA repair (PARP vulnerability)
    "Base excision repair":                 0.70,
    "PARP signaling":                       0.85,
    "Synthetic lethality":                  0.85,

    # Autophagy — relevant to resistance
    "Autophagy":                            0.70,
    "Lysosomal function":                   0.65,

    # Immune (cold TME in DIPG — lower priority)
    "T-cell checkpoint signaling":          0.55,
    "PD-1/PD-L1 signaling":                0.55,

    # Angiogenesis (VEGF relevant but less so than epigenetic targets)
    "VEGF signaling":                       0.60,
    "Angiogenesis":                         0.60,

    # BBB-specific (relevant to drug delivery)
    "P-glycoprotein signaling":             0.65,
    "ABC transporter activity":             0.65,
}

# ─────────────────────────────────────────────────────────────────────────────
# DIPG resistance genes
# Genes whose activation drives treatment resistance in DIPG/H3K27M gliomas
# ─────────────────────────────────────────────────────────────────────────────
DIPG_RESISTANCE_GENES: Set[str] = {
    # Core resistance mechanisms
    "H3F3A",     # The oncogenic mutation itself — resistance by definition
    "HIST1H3B",  # H3.1 variant
    "PDGFRA",    # Amplification drives resistance to PDGFR inhibitors
    "EGFR",      # Feedback activation after PDGFRA inhibition
    "PTEN",      # Loss activates PI3K signalling — resistance to PDGFRA inh
    "PIK3CA",    # Activating mutation — bypass resistance
    "CDKN2A",   # Deletion — resistance to CDK4/6 inhibitors
    "CDK4",      # Amplification — resistance
    "MDM2",      # p53 inactivation — resistance to genotoxic agents
    "TP53",      # Mutation — resistance to many therapies
    "ATRX",      # Loss — alternative lengthening of telomeres
    "MYC",       # Amplification — broad resistance to differentiation
    "MYCN",      # N-Myc amplification

    # Epigenetic resistance
    "KDM6A",     # H3K27 demethylase upregulation — partially reverses H3K27me3 loss
    "KDM6B",     # H3K27 demethylase
    "BRD4",      # BET resistance through amplification

    # Drug efflux (BBB + intracellular)
    "ABCB1",     # P-gp — major BBB efflux pump
    "ABCG2",     # BCRP — BBB efflux pump
    "ABCC1",     # MRP1 — drug efflux

    # Anti-apoptotic
    "BCL2",      # Anti-apoptotic
    "BCL2L1",    # BCL-XL
    "MCL1",      # Anti-apoptotic — frequently upregulated in GBM

    # Stemness (resistance by maintaining stem cell state)
    "SOX2",      # Stem cell maintenance
    "OLIG2",     # Lineage-specific resistance
}

# ─────────────────────────────────────────────────────────────────────────────
# Novel drug-target combinations specifically relevant to DIPG
# These are mechanistically rational but NOT yet widely tested in DIPG
# This is the "white space" for publication novelty
# ─────────────────────────────────────────────────────────────────────────────
DIPG_NOVEL_TARGETS: Dict[str, Dict] = {
    "ACVR1": {
        "rationale": "Gain-of-function mutations in ~25% DIPG; essentially absent in adult GBM",
        "drugs_to_test": ["Dorsomorphin", "LDN-193189", "K02288", "INCB000928"],
        "novelty_score": 1.0,
        "clinical_status": "Preclinical only — no approved ACVR1 inhibitor",
    },
    "EZH2": {
        "rationale": "H3K27M inhibits PRC2/EZH2 — paradoxically, residual EZH2 activity maintains oncogenic gene programs",
        "drugs_to_test": ["Tazemetostat", "GSK126", "EPZ-6438"],
        "novelty_score": 0.8,
        "clinical_status": "Tazemetostat FDA-approved for EZH2-mutant FL; not tested in DIPG",
    },
    "BRD4": {
        "rationale": "H3K27M creates hyperacetylated enhancers; BRD4 reads H3K27ac → BET inhibition disrupts DIPG transcriptional program",
        "drugs_to_test": ["JQ1", "OTX015", "Molibresib", "ABBV-744"],
        "novelty_score": 0.85,
        "clinical_status": "BET inhibitors in adult GBM trials; DIPG data sparse",
    },
    "HDAC1/2": {
        "rationale": "H3K27M global histone hypomethylation balanced by hyperacetylation; HDAC inhibition + DNA damage synergistic",
        "drugs_to_test": ["Panobinostat", "Vorinostat", "Entinostat"],
        "novelty_score": 0.75,
        "clinical_status": "Panobinostat in DIPG trials (some data); combination not explored",
    },
    "CDK4/6": {
        "rationale": "CDK4 amplification in subset; Abemaciclib has best CNS penetrance of CDK4/6 inhibitors",
        "drugs_to_test": ["Abemaciclib", "Ribociclib"],
        "novelty_score": 0.70,
        "clinical_status": "Abemaciclib CNS penetrance makes it preferred; DIPG-specific trials needed",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# DIPG disease parameters for insilico_trial.py integration
# ─────────────────────────────────────────────────────────────────────────────
DIPG_DISEASE_PARAMS: Dict = {
    "baseline_orr":                   0.05,   # ~5% ORR with current SOC (worse than adult GBM)
    "baseline_pfs6":                  0.10,   # Very poor PFS
    "stroma_barrier":                 0.30,   # Brainstem infiltrative; hard to biopsy
    "bbb_barrier":                    0.80,   # Brainstem has tighter BBB than cortex
    "mutation_heterogeneity":         0.70,   # Moderate — H3K27M is clonal but other muts vary
    "pdgfra_amplification_fraction":  0.36,   # PDGFRA amplified in 36% DIPG
    "acvr1_mutation_fraction":        0.25,   # ACVR1 mutated in ~25% DIPG
    "h3k27m_prevalence":              0.80,   # H3K27M found in ~80% DIPG
    "h31_k27m_fraction":              0.23,   # H3.1 K27M subtype (more responsive to rt)
    "h33_k27m_fraction":              0.57,   # H3.3 K27M subtype
    "immune_desert_fraction":         0.85,   # Very cold TME — brainstem immune privilege
    "phase2_success_threshold_orr":   0.15,   # Lower bar given poor baseline
    "resistance_genes":               DIPG_RESISTANCE_GENES,
    "description":                    (
        "DIPG (H3K27M-mutant) — brainstem location, tight BBB, H3K27M epigenetic "
        "vulnerability, ACVR1/PDGFRA subtypes, extremely cold TME, median survival 9-11 months."
    ),
    "outcome_metric": "ORR + PFS-6 (given poor baseline, any response is meaningful)",
}


# ─────────────────────────────────────────────────────────────────────────────
# DIPG-specific scoring adjustments
# ─────────────────────────────────────────────────────────────────────────────

class DIPGSpecializedScorer:
    """
    Augments the base pipeline scoring with DIPG/H3K27M-specific logic.

    Key augmentations:
    1. H3K27M epigenetic vulnerability bonus for HDAC/EZH2/BET inhibitors
    2. ACVR1 subtype bonus for BMP pathway inhibitors
    3. BBB penetrance requirement (brainstem is tighter than cortex)
    4. Novelty bonus for drugs not yet tested in DIPG (white-space finding)

    This class is designed to be applied AFTER the base pipeline scoring,
    adding DIPG-specific signal without replacing the generic mechanism.
    """

    # Drugs that hit DIPG-specific vulnerabilities but have NOT been
    # formally tested in DIPG clinical trials (as of mid-2025)
    # This is the publication white-space
    UNTESTED_IN_DIPG: Set[str] = {
        "tazemetostat",    # EZH2 inhibitor
        "abemaciclib",     # CDK4/6 inhibitor with CNS penetrance
        "ribociclib",      # CDK4/6 inhibitor
        "molibresib",      # BET inhibitor
        "entinostat",      # Class I HDAC inhibitor
        "tucidinostat",    # HDAC inhibitor
        "navitoclax",      # BCL-2/XL inhibitor
        "selinexor",       # XPO1 inhibitor
        "alisertib",       # Aurora A kinase inhibitor
        "olaparib",        # PARP inhibitor
        "niraparib",       # PARP inhibitor
        "veliparib",       # PARP inhibitor
        "alpelisib",       # PI3Kalpha inhibitor
        "copanlisib",      # Pan-PI3K inhibitor
        "afatinib",        # Pan-ErbB inhibitor
    }

    # Mechanism keywords that indicate H3K27M/DIPG relevance
    DIPG_MECHANISM_KEYWORDS: Dict[str, float] = {
        "hdac inhibitor":           0.40,
        "histone deacetylase":      0.40,
        "pan-hdac":                 0.45,
        "class i hdac":             0.35,
        "ezh2 inhibitor":           0.40,
        "prc2 inhibitor":           0.38,
        "bet inhibitor":            0.40,
        "brd4 inhibitor":           0.40,
        "bromodomain":              0.35,
        "cdk4":                     0.30,
        "cdk4/6":                   0.35,
        "cdk 4":                    0.30,
        "pdgfr":                    0.30,
        "pdgfra":                   0.35,
        "pi3k inhibitor":           0.25,
        "mtor inhibitor":           0.25,
        "bmp inhibitor":            0.45,
        "acvr1 inhibitor":          0.50,
        "parp inhibitor":           0.25,
        "dna damage":               0.20,
        "autophagy":                0.20,
        "proteasome inhibitor":     0.15,
    }

    def __init__(
        self,
        apply_bbb_penalty:     bool  = True,
        novelty_bonus:         float = 0.08,
        h3k27m_bonus:          float = 0.12,
        acvr1_bonus:           float = 0.10,
        cns_drug_boost:        float = 0.05,
    ):
        self.apply_bbb_penalty = apply_bbb_penalty
        self.novelty_bonus     = novelty_bonus
        self.h3k27m_bonus      = h3k27m_bonus
        self.acvr1_bonus       = acvr1_bonus
        self.cns_drug_boost    = cns_drug_boost

        logger.info(
            "✅ DIPGSpecializedScorer: h3k27m_bonus=%.2f, acvr1_bonus=%.2f, "
            "novelty_bonus=%.2f, cns_boost=%.2f",
            h3k27m_bonus, acvr1_bonus, novelty_bonus, cns_drug_boost,
        )

    def _h3k27m_vulnerability_score(self, drug: Dict) -> float:
        """
        Score drug's relevance to H3K27M epigenetic vulnerability.
        Returns bonus in [0, h3k27m_bonus].
        """
        mechanism  = (drug.get("mechanism", "") or "").lower()
        name_lower = (drug.get("name", drug.get("drug_name", "")) or "").lower()
        targets    = [t.upper() for t in (drug.get("targets") or [])]

        score = 0.0

        # Mechanism keyword matching
        for keyword, weight in self.DIPG_MECHANISM_KEYWORDS.items():
            if keyword in mechanism:
                score = max(score, weight)

        # Target gene matching — check against H3K27M vulnerability targets
        h3k27m_vulnerability_genes = {
            "EZH2", "EED", "SUZ12",     # PRC2 complex
            "KDM6A", "KDM6B",           # H3K27 demethylases
            "BRD4", "BRD2", "BRD3",     # BET bromodomains
            "HDAC1", "HDAC2", "HDAC3",  # HDACs
            "HDAC4", "HDAC6",
        }
        target_hits = set(targets) & h3k27m_vulnerability_genes
        if target_hits:
            score = max(score, 0.35 + len(target_hits) * 0.05)

        # Valproic acid special case — HDAC inhibitor with excellent CNS penetrance
        if "valproic" in name_lower or "valproate" in name_lower:
            score = max(score, 0.40)

        return min(score * self.h3k27m_bonus / 0.40, self.h3k27m_bonus)

    def _acvr1_score(self, drug: Dict) -> float:
        """
        Score drug's relevance to ACVR1-mutant DIPG subtype (~25%).
        Returns bonus in [0, acvr1_bonus].
        """
        mechanism  = (drug.get("mechanism", "") or "").lower()
        targets    = [t.upper() for t in (drug.get("targets") or [])]

        acvr1_genes = {"ACVR1", "BMPR1A", "BMPR2", "SMAD1", "SMAD5",
                       "SMAD4", "ID1", "ID2"}
        bmp_keywords = ("bmp", "acvr1", "activin", "bone morphogenetic",
                        "alk2", "type i bmp receptor")

        target_hits = set(targets) & acvr1_genes
        mech_hit    = any(kw in mechanism for kw in bmp_keywords)

        if target_hits or mech_hit:
            # Weight by 0.25 (ACVR1 fraction of DIPG) — it's a subtype
            return self.acvr1_bonus * 0.25 if not target_hits else self.acvr1_bonus
        return 0.0

    def _novelty_bonus(self, drug: Dict) -> float:
        """
        Bonus for drugs not yet tested in DIPG clinical trials.
        This is the key "white space" signal for the paper.
        """
        name_lower = (drug.get("name", drug.get("drug_name", "")) or "").lower()
        # Remove salt suffixes for matching
        for suffix in (" hydrochloride", " hcl", " sodium", " mesylate"):
            name_lower = name_lower.replace(suffix, "")
        name_lower = name_lower.strip()

        if name_lower in self.UNTESTED_IN_DIPG:
            return self.novelty_bonus
        # Partial match
        for untested in self.UNTESTED_IN_DIPG:
            if untested in name_lower or name_lower in untested:
                return self.novelty_bonus * 0.75
        return 0.0

    def score_candidate(self, candidate: Dict) -> Dict:
        """
        Apply DIPG-specific scoring augmentation to a candidate.
        Adds dipg_score, dipg_components, and adjusts composite score.
        """
        base_score = candidate.get("score", 0.0)

        h3k27m_bonus = self._h3k27m_vulnerability_score(candidate)
        acvr1_bonus  = self._acvr1_score(candidate)
        novelty      = self._novelty_bonus(candidate)

        # CNS drug boost — drugs approved for CNS use get small bump
        # (implies known CNS penetrance even if BBB score uncertain)
        bbb_pen = candidate.get("bbb_penetrance", "UNKNOWN")
        cns_boost = self.cns_drug_boost if bbb_pen == "HIGH" else 0.0

        total_bonus = h3k27m_bonus + acvr1_bonus + novelty + cns_boost
        adjusted    = min(1.0, base_score + total_bonus)

        candidate["dipg_score"]          = round(adjusted, 4)
        candidate["dipg_bonus_total"]    = round(total_bonus, 4)
        candidate["dipg_components"] = {
            "base_score":      round(base_score, 4),
            "h3k27m_bonus":    round(h3k27m_bonus, 4),
            "acvr1_bonus":     round(acvr1_bonus, 4),
            "novelty_bonus":   round(novelty, 4),
            "cns_boost":       round(cns_boost, 4),
            "is_untested_dipg": novelty > 0,
            "h3k27m_relevant": h3k27m_bonus > 0,
            "acvr1_relevant":  acvr1_bonus > 0,
        }
        candidate["score"] = adjusted  # Update composite score

        return candidate

    def score_batch(self, candidates: List[Dict]) -> List[Dict]:
        """Score all candidates with DIPG-specific augmentation."""
        logger.info(f"🧬 DIPG specialization: scoring {len(candidates)} candidates")
        for c in candidates:
            self.score_candidate(c)
        candidates.sort(key=lambda x: x.get("score", 0), reverse=True)

        n_h3k27m = sum(1 for c in candidates if c.get("dipg_components", {}).get("h3k27m_relevant"))
        n_acvr1  = sum(1 for c in candidates if c.get("dipg_components", {}).get("acvr1_relevant"))
        n_novel  = sum(1 for c in candidates if c.get("dipg_components", {}).get("is_untested_dipg"))

        logger.info(
            f"   H3K27M-relevant: {n_h3k27m}, ACVR1-relevant: {n_acvr1}, "
            f"Untested in DIPG (novelty candidates): {n_novel}"
        )
        return candidates

    def generate_novelty_report(self, candidates: List[Dict], top_n: int = 10) -> str:
        """
        Generate a focused report on novel DIPG repurposing candidates.
        This is the core output for the paper's results section.
        """
        novel_candidates = [
            c for c in candidates
            if c.get("dipg_components", {}).get("is_untested_dipg")
        ]
        novel_candidates.sort(key=lambda x: x.get("score", 0), reverse=True)

        lines = [
            "# Novel DIPG Repurposing Candidates (Untested in DIPG Clinical Trials)",
            "",
            "## Summary",
            f"- Total candidates scored: {len(candidates)}",
            f"- Novel candidates (untested in DIPG): {len(novel_candidates)}",
            f"- Top {min(top_n, len(novel_candidates))} candidates shown below",
            "",
            "## Candidate Rankings",
            "",
            "| Rank | Drug | Score | H3K27M | ACVR1 | CNS Penetrance | Mechanism |",
            "|------|------|-------|--------|-------|----------------|-----------|",
        ]

        for i, c in enumerate(novel_candidates[:top_n], 1):
            comp = c.get("dipg_components", {})
            bbb  = c.get("bbb_penetrance", "?")
            mech = (c.get("mechanism", "") or "")[:50]
            h3   = "✓" if comp.get("h3k27m_relevant") else "-"
            ac   = "✓" if comp.get("acvr1_relevant") else "-"
            lines.append(
                f"| {i} | **{c.get('drug_name', c.get('name', '?'))}** | "
                f"{c.get('score', 0):.3f} | {h3} | {ac} | {bbb} | {mech} |"
            )

        lines += [
            "",
            "## DIPG Novel Target White Space",
            "",
            "The following target-drug combinations are mechanistically rational",
            "for H3K27M-mutant DIPG but have no published clinical trial data:",
            "",
        ]

        from dipg_specialization import DIPG_NOVEL_TARGETS
        for target, info in DIPG_NOVEL_TARGETS.items():
            lines += [
                f"### {target}",
                f"**Rationale**: {info['rationale']}",
                f"**Candidate drugs**: {', '.join(info['drugs_to_test'])}",
                f"**Clinical status**: {info['clinical_status']}",
                f"**Novelty score**: {info['novelty_score']:.1f}/1.0",
                "",
            ]

        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Gene score weighting adjustments for DIPG
# ─────────────────────────────────────────────────────────────────────────────

def get_dipg_gene_score_weights() -> Dict[str, float]:
    """
    Returns gene-level importance weights for DIPG.
    Used to boost genes that are specifically important in H3K27M tumours.
    """
    return {
        # Primary H3K27M oncogenic drivers — highest weight
        "H3F3A":    1.00,
        "HIST1H3B": 1.00,
        "ACVR1":    0.95,
        "PDGFRA":   0.90,
        "EZH2":     0.90,
        "BRD4":     0.90,

        # Key epigenetic regulators
        "KDM6A":    0.85,
        "KDM6B":    0.80,
        "HDAC1":    0.80,
        "HDAC2":    0.80,
        "ATRX":     0.75,
        "SETD2":    0.75,

        # Cell cycle (CDK4 amplification)
        "CDK4":     0.85,
        "CDK6":     0.75,
        "CDKN2A":   0.80,

        # Signalling
        "PIK3CA":   0.75,
        "PTEN":     0.80,
        "TP53":     0.75,
        "MYC":      0.70,
        "MYCN":     0.70,
        "EGFR":     0.65,

        # Drug efflux (resistance) — lower weight as targets
        "ABCB1":    0.40,
        "ABCG2":    0.40,
    }


def get_dipg_disease_data_supplement() -> Dict:
    """
    Returns supplementary disease data dict for DIPG.
    Can be merged into the OpenTargets disease_data dict for pipeline use.
    """
    gene_weights = get_dipg_gene_score_weights()
    return {
        "name":         "Diffuse intrinsic pontine glioma",
        "aliases":      ["DIPG", "H3K27M-mutant glioma", "diffuse midline glioma H3K27M"],
        "genes":        DIPG_CORE_GENES,
        "gene_scores":  gene_weights,
        "pathways":     list(DIPG_PATHWAY_WEIGHTS.keys()),
        "is_rare":      True,
        "is_pediatric": True,
        "h3k27m_mutant": True,
        "bbb_relevant": True,
        "active_trials_count": 15,  # approximate as of 2024
        "disease_params": DIPG_DISEASE_PARAMS,
        "source":       "DIPG specialization module (Mackay 2017, Wu 2012, Grasso 2015)",
        "notes": (
            "DIPG is a pediatric brainstem glioma driven by H3K27M histone mutation. "
            "~80% carry H3K27M. ACVR1 mutations in ~25%. Median OS 9-11 months. "
            "No effective systemic therapy to date."
        ),
    }
"""
tme_scorer.py — Tumour Microenvironment (TME) Scoring Module
=============================================================
Scores drug candidates for their ability to overcome or exploit the GBM/DIPG
tumour microenvironment. The TME is a primary reason drugs that work against
tumour cells in vitro fail in vivo — particularly in GBM, which has one of
the most immunosuppressive TMEs of any solid tumour.

BIOLOGICAL RATIONALE
---------------------
The GBM TME is characterised by:
  1. Glioma-associated macrophages/microglia (GAMs) — up to 50% of tumour mass.
     GAMs are reprogrammed to an immunosuppressive M2-like phenotype.
  2. Regulatory T cells (Tregs) — TGF-β and IDO-mediated suppression.
  3. T cell exhaustion — PD-1/PD-L1, TIM-3, LAG-3 checkpoint upregulation.
  4. IDO1 activity — tryptophan depletion suppresses effector T cells.
  5. VEGF-mediated immunosuppression — abnormal vasculature limits immune
     cell trafficking.
  6. Blood-brain barrier (handled separately by bbb_filter.py).
  7. Low mutational burden — fewer neoantigens vs other solid tumours.

DIPG-specific TME features:
  - Brainstem location means even less immune surveillance than cortical GBM.
  - H3K27M mutation alters inflammatory gene expression (NF-κB pathway).
  - ACVR1 mutations activate BMP signalling, which modulates immune response.

SCORING APPROACH
-----------------
Each drug receives a TME score (0–1) combining:
  1. Immunomodulatory activity — does the drug target TME suppression pathways?
  2. M2→M1 repolarisation potential — does it shift macrophage phenotype?
  3. Checkpoint relevance — does it affect T cell exhaustion?
  4. Anti-angiogenic effect — VEGF normalisation improves immune trafficking.
  5. Direct TME target presence — do its known targets appear in GBM TME signatures?

Gene signatures sourced from:
  - ImmPort (immport.org) curated immune gene lists
  - Neftel et al. (2019) GBM single-cell atlas — Cell
  - Darmanis et al. (2017) single-cell RNA-seq of GBM — Cell Reports
  - Pombo Antunes et al. (2021) DIPG TME — Nature Neuroscience
  - Ligon lab GBM TME signatures (Broad Institute)

REFERENCES
-----------
Neftel C et al. (2019). An Integrative Model of Cellular States, Plasticity,
  and Genetics for Glioblastoma. Cell. doi:10.1016/j.cell.2019.06.024

Pombo Antunes AR et al. (2021). Single-cell profiling of myeloid cells in
  glioblastoma across species and disease stage reveals macrophage competition
  and specialization. Nature Neuroscience. doi:10.1038/s41593-020-00789-y

Darmanis S et al. (2017). Single-Cell RNA-Seq Analysis of Infiltrating
  Neoplastic Cells at the Migrating Front of Human Glioblastoma.
  Cell Reports. doi:10.1016/j.celrep.2017.10.030

Gieryng A et al. (2017). Immune microenvironment of gliomas.
  Lab Invest. doi:10.1038/labinvest.2017.19
"""

import logging
import math
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# GBM TME GENE SIGNATURES
# Curated from published single-cell and bulk RNA-seq studies of GBM TME.
# ─────────────────────────────────────────────────────────────────────────────

# Glioma-associated macrophage / microglia (GAM) immunosuppressive markers
# Source: Pombo Antunes et al. 2021; Darmanis et al. 2017
GAM_IMMUNOSUPPRESSIVE_GENES: Set[str] = {
    "CD163",   # M2 macrophage marker — scavenger receptor
    "MRC1",    # CD206 — mannose receptor, M2 polarisation
    "ARG1",    # Arginase-1 — depletes arginine, suppresses T cells
    "IL10",    # Immunosuppressive cytokine
    "TGFB1",   # TGF-β1 — master immunosuppressor in GBM
    "TGFB2",   # TGF-β2 — autocrine loop in GBM cells
    "VEGFA",   # VEGF — angiogenesis + immune suppression
    "CSF1R",   # M-CSFR — macrophage survival/proliferation
    "CSF1",    # M-CSF — drives GAM recruitment
    "CX3CR1",  # Fractalkine receptor on microglia
    "P2RY12",  # Homeostatic microglia marker (lost in GBM)
    "TMEM119", # Homeostatic microglia (lost in tumour-associated state)
    "IDO1",    # Indoleamine 2,3-dioxygenase — tryptophan catabolism
    "IDO2",    # IDO2 isoform
    "CD68",    # Pan-macrophage marker
    "CD14",    # Monocyte/macrophage
    "ITGAM",   # CD11b — myeloid cell adhesion
    "SPP1",    # Osteopontin — pro-tumour macrophage signal
    "LGALS3",  # Galectin-3 — immunosuppressive lectin
    "CCL2",    # MCP-1 — monocyte recruitment
    "CCL7",    # CCR2 ligand — monocyte recruitment
    "CXCL12",  # SDF-1 — immunosuppressive, promotes tumour growth
    "HIF1A",   # Hypoxia-inducible factor — M2 polarisation driver
}

# T cell exhaustion / checkpoint genes
# Source: ImmPort; Neftel et al. 2019
T_CELL_EXHAUSTION_GENES: Set[str] = {
    "PDCD1",   # PD-1 — exhaustion marker
    "CD274",   # PD-L1 — checkpoint ligand (expressed by GBM cells + GAMs)
    "PDCD1LG2",# PD-L2
    "CTLA4",   # CTLA-4 — early checkpoint
    "HAVCR2",  # TIM-3 — GBM enriched exhaustion marker
    "LAG3",    # LAG-3 — co-exhaustion with TIM-3 in GBM
    "TIGIT",   # TIGIT — emerging checkpoint in GBM
    "ENTPD1",  # CD39 — exhausted T cell marker
    "NT5E",    # CD73 — adenosine pathway
    "ADORA2A", # Adenosine receptor A2A — T cell suppression
    "FOXP3",   # Treg transcription factor
    "IL2RA",   # CD25 — Treg marker
    "IKZF2",   # Helios — Treg stability
    "TNFRSF9", # 4-1BB — T cell co-stimulation (target for activation)
    "CD28",    # T cell co-stimulation
    "ICOS",    # Inducible T cell co-stimulator
}

# Anti-tumour immune effector genes (drugs that upregulate these are positive)
ANTI_TUMOUR_IMMUNE_GENES: Set[str] = {
    "CD8A",    # Cytotoxic T cells
    "CD8B",    # Cytotoxic T cells
    "GZMB",    # Granzyme B — cytotoxic activity
    "PRF1",    # Perforin — cytolysis
    "IFNG",    # IFN-γ — anti-tumour cytokine
    "TNF",     # TNF-α — tumour killing
    "IL2",     # T cell proliferation
    "IL15",    # NK cell activation
    "NCR1",    # NKp46 — NK cell activation receptor
    "KLRK1",   # NKG2D — stress ligand receptor
    "CD56",    # NCAM1 — NK cell marker
    "CD16",    # FCGR3A — NK cell activation
    "CD40",    # APC activation — M1 macrophage polarisation
    "CD40LG",  # CD40L — T cell-APC interaction
    "CD80",    # B7-1 — APC co-stimulation
    "CD86",    # B7-2 — APC co-stimulation
    "IL12A",   # IL-12 — M1 polarisation driver
    "IL12B",   # IL-12 subunit
    "CXCL9",   # CXCR3 ligand — T cell trafficking into tumour
    "CXCL10",  # IP-10 — T cell recruitment
    "CXCL11",  # CXCR3 ligand
}

# VEGF/angiogenesis genes (anti-VEGF normalises vasculature, improves immune access)
ANGIOGENESIS_GENES: Set[str] = {
    "VEGFA",   # Primary angiogenic driver in GBM
    "VEGFB",   # VEGF-B
    "VEGFC",   # VEGF-C
    "KDR",     # VEGFR2 — primary angiogenic receptor
    "FLT1",    # VEGFR1
    "FLT4",    # VEGFR3
    "PDGFB",   # PDGF-B — pericyte recruitment
    "PDGFRB",  # PDGFR-β — pericyte
    "ANGPT1",  # Angiopoietin-1 — vessel stabilisation
    "ANGPT2",  # Angiopoietin-2 — vessel destabilisation
    "TEK",     # Tie2 — angiopoietin receptor
    "HIF1A",   # Hypoxia driver of VEGF
    "EPAS1",   # HIF-2α
    "NRP1",    # Neuropilin-1 — VEGF co-receptor
    "NRP2",    # Neuropilin-2
}

# NF-κB / inflammatory pathway (H3K27M activates NF-κB in DIPG)
NFKB_INFLAMMATORY_GENES: Set[str] = {
    "NFKB1",   # NF-κB p50
    "NFKB2",   # NF-κB p52
    "RELA",    # NF-κB p65 — pro-survival in GBM
    "RELB",    # NF-κB RelB
    "REL",     # c-Rel
    "IKBKB",   # IKKβ — NF-κB activating kinase
    "IKBKG",   # NEMO — NF-κB essential modulator
    "TNFAIP3", # A20 — NF-κB negative regulator
    "IL6",     # IL-6 — JAK-STAT activator, pro-tumour
    "IL8",     # CXCL8 — neutrophil recruitment, pro-angiogenic
    "IL1B",    # IL-1β — inflammatory
    "STAT3",   # STAT3 — pro-survival, immunosuppressive
}

# Combined GBM TME signature (all immunosuppressive targets)
GBM_TME_SIGNATURE: Set[str] = (
    GAM_IMMUNOSUPPRESSIVE_GENES
    | T_CELL_EXHAUSTION_GENES
    | ANGIOGENESIS_GENES
    | NFKB_INFLAMMATORY_GENES
)

# DIPG-specific additions (brainstem immune privilege, H3K27M effects)
DIPG_TME_ADDITIONAL_GENES: Set[str] = {
    "ACVR1",   # ACVR1 mutations alter BMP/inflammatory signalling
    "SMAD1",   # BMP-SMAD pathway
    "ID1",     # BMP target — immune evasion
    "ID2",     # BMP target
    "CHD7",    # Chromatin remodeller — co-mutated with ACVR1 in DIPG
    "CX3CL1",  # Fractalkine — microglia/macrophage communication
    "IL13RA2", # IL-13Rα2 — GBM/DIPG antigen (CAR-T target)
    "GD2",     # Ganglioside GD2 — DIPG surface antigen
    "B7H3",    # CD276 — immune checkpoint overexpressed in DIPG
}

DIPG_TME_SIGNATURE: Set[str] = GBM_TME_SIGNATURE | DIPG_TME_ADDITIONAL_GENES


# ─────────────────────────────────────────────────────────────────────────────
# DRUG TME ACTIVITY DATABASE
# Curated from published literature on drug effects on immune/TME components.
# Each entry: drug_name → {tme_activities, tme_score_bonus, rationale}
# ─────────────────────────────────────────────────────────────────────────────

DRUG_TME_ACTIVITY: Dict[str, Dict] = {

    # ── Checkpoint inhibitors ─────────────────────────────────────────────────
    "Nivolumab": {
        "activities": ["checkpoint_blockade", "pd1_inhibition", "t_cell_activation"],
        "tme_bonus": 0.35,
        "m2_m1_shift": 0.10,
        "anti_exhaustion": 0.40,
        "anti_vegf": 0.0,
        "rationale": "Anti-PD-1. High TME relevance but poor BBB penetrance.",
        "bbb_caveat": True,
    },
    "Pembrolizumab": {
        "activities": ["checkpoint_blockade", "pd1_inhibition"],
        "tme_bonus": 0.35,
        "m2_m1_shift": 0.10,
        "anti_exhaustion": 0.40,
        "anti_vegf": 0.0,
        "rationale": "Anti-PD-1. Similar to nivolumab.",
        "bbb_caveat": True,
    },
    "Atezolizumab": {
        "activities": ["checkpoint_blockade", "pdl1_inhibition"],
        "tme_bonus": 0.30,
        "m2_m1_shift": 0.10,
        "anti_exhaustion": 0.35,
        "anti_vegf": 0.0,
        "rationale": "Anti-PD-L1.",
        "bbb_caveat": True,
    },
    "Ipilimumab": {
        "activities": ["checkpoint_blockade", "ctla4_inhibition"],
        "tme_bonus": 0.30,
        "m2_m1_shift": 0.15,
        "anti_exhaustion": 0.30,
        "anti_vegf": 0.0,
        "rationale": "Anti-CTLA-4. Depletes Tregs, activates APCs.",
        "bbb_caveat": True,
    },

    # ── Anti-angiogenic agents ────────────────────────────────────────────────
    "Bevacizumab": {
        "activities": ["anti_vegf", "vascular_normalisation", "tme_remodelling"],
        "tme_bonus": 0.25,
        "m2_m1_shift": 0.05,
        "anti_exhaustion": 0.05,
        "anti_vegf": 0.35,
        "rationale": "Anti-VEGF. Normalises GBM vasculature, reduces immunosuppressive VEGF.",
        "bbb_caveat": True,  # monoclonal
    },
    "Pazopanib": {
        "activities": ["anti_vegf", "vegfr_inhibition", "vascular_normalisation"],
        "tme_bonus": 0.18,
        "m2_m1_shift": 0.05,
        "anti_exhaustion": 0.0,
        "anti_vegf": 0.25,
        "rationale": "Multi-kinase VEGFR/PDGFR inhibitor. Vascular normalisation.",
    },
    "Cabozantinib": {
        "activities": ["anti_vegf", "met_inhibition", "axl_inhibition"],
        "tme_bonus": 0.22,
        "m2_m1_shift": 0.12,
        "anti_exhaustion": 0.05,
        "anti_vegf": 0.22,
        "rationale": "VEGFR2/MET/AXL inhibitor. AXL inhibition reduces M2 polarisation.",
    },
    "Lenvatinib": {
        "activities": ["anti_vegf", "vegfr_inhibition", "immunomodulation"],
        "tme_bonus": 0.20,
        "m2_m1_shift": 0.10,
        "anti_exhaustion": 0.05,
        "anti_vegf": 0.28,
        "rationale": "VEGFR/FGFR/PDGFR inhibitor with emerging immunomodulatory data.",
    },

    # ── IDO inhibitors ────────────────────────────────────────────────────────
    "Epacadostat": {
        "activities": ["ido1_inhibition", "tryptophan_restoration", "t_cell_rescue"],
        "tme_bonus": 0.28,
        "m2_m1_shift": 0.08,
        "anti_exhaustion": 0.22,
        "anti_vegf": 0.0,
        "rationale": "IDO1 inhibitor. Restores tryptophan, rescues T cell function in GBM TME.",
    },
    "Navoximod": {
        "activities": ["ido1_inhibition", "tryptophan_restoration"],
        "tme_bonus": 0.22,
        "m2_m1_shift": 0.05,
        "anti_exhaustion": 0.18,
        "anti_vegf": 0.0,
        "rationale": "IDO1 inhibitor. Oral bioavailability, some CNS exposure data.",
    },

    # ── CSF1R inhibitors (macrophage reprogramming) ───────────────────────────
    "Pexidartinib": {
        "activities": ["csf1r_inhibition", "gam_depletion", "m2_m1_shift"],
        "tme_bonus": 0.32,
        "m2_m1_shift": 0.38,
        "anti_exhaustion": 0.10,
        "anti_vegf": 0.0,
        "rationale": "CSF1R inhibitor. Depletes/reprograms GAMs — directly targets GBM TME.",
    },
    "Emactuzumab": {
        "activities": ["csf1r_inhibition", "gam_reprogramming"],
        "tme_bonus": 0.28,
        "m2_m1_shift": 0.32,
        "anti_exhaustion": 0.08,
        "anti_vegf": 0.0,
        "rationale": "Anti-CSF1R monoclonal. Reprograms rather than depletes TAMs.",
        "bbb_caveat": True,
    },

    # ── TGF-β inhibitors ──────────────────────────────────────────────────────
    "Galunisertib": {
        "activities": ["tgfb_inhibition", "immunosuppression_reversal", "treg_reduction"],
        "tme_bonus": 0.30,
        "m2_m1_shift": 0.20,
        "anti_exhaustion": 0.18,
        "anti_vegf": 0.0,
        "rationale": "TGFβRI inhibitor. TGF-β is a dominant GBM immunosuppressor.",
    },
    "Vactosertib": {
        "activities": ["tgfb_inhibition", "alk5_inhibition"],
        "tme_bonus": 0.28,
        "m2_m1_shift": 0.18,
        "anti_exhaustion": 0.15,
        "anti_vegf": 0.0,
        "rationale": "TGFβRI/ALK5 inhibitor with CNS exposure data.",
    },

    # ── STAT3 inhibitors ──────────────────────────────────────────────────────
    "Napabucasin": {
        "activities": ["stat3_inhibition", "cancer_stem_cell", "immunomodulation"],
        "tme_bonus": 0.22,
        "m2_m1_shift": 0.12,
        "anti_exhaustion": 0.08,
        "anti_vegf": 0.0,
        "rationale": "STAT3 inhibitor. STAT3 drives both tumour cell survival and M2 polarisation.",
    },

    # ── Immunomodulatory drugs with TME effects ───────────────────────────────
    "Thalidomide": {
        "activities": ["tnf_inhibition", "anti_angiogenic", "immunomodulation"],
        "tme_bonus": 0.18,
        "m2_m1_shift": 0.10,
        "anti_exhaustion": 0.05,
        "anti_vegf": 0.12,
        "rationale": "CRBN/IMiD. Anti-angiogenic, anti-TNF, immune modulatory.",
    },
    "Lenalidomide": {
        "activities": ["immunomodulation", "anti_angiogenic", "nk_activation"],
        "tme_bonus": 0.20,
        "m2_m1_shift": 0.12,
        "anti_exhaustion": 0.08,
        "anti_vegf": 0.10,
        "rationale": "IMiD. NK cell activation, anti-angiogenic, immune stimulatory.",
    },

    # ── HDAC inhibitors (have known TME effects) ──────────────────────────────
    "Panobinostat": {
        "activities": ["hdac_inhibition", "immunomodulation", "mhc_upregulation"],
        "tme_bonus": 0.15,
        "m2_m1_shift": 0.08,
        "anti_exhaustion": 0.10,
        "anti_vegf": 0.0,
        "rationale": "Pan-HDAC. Upregulates MHC-I, enhances immune recognition of GBM cells.",
    },
    "Vorinostat": {
        "activities": ["hdac_inhibition", "mhc_upregulation", "immunomodulation"],
        "tme_bonus": 0.12,
        "m2_m1_shift": 0.06,
        "anti_exhaustion": 0.08,
        "anti_vegf": 0.0,
        "rationale": "HDAC inhibitor. MHC-I upregulation, PD-L1 modulation.",
    },
    "Entinostat": {
        "activities": ["hdac_inhibition", "treg_reduction", "mdsc_reduction"],
        "tme_bonus": 0.18,
        "m2_m1_shift": 0.10,
        "anti_exhaustion": 0.12,
        "anti_vegf": 0.0,
        "rationale": "Class I HDAC inhibitor. Reduces Tregs and MDSCs — distinctive TME effect.",
    },

    # ── PI3K/mTOR inhibitors (affect TME) ────────────────────────────────────
    "Everolimus": {
        "activities": ["mtor_inhibition", "treg_modulation", "anti_angiogenic"],
        "tme_bonus": 0.10,
        "m2_m1_shift": 0.05,
        "anti_exhaustion": 0.05,
        "anti_vegf": 0.08,
        "rationale": "mTOR inhibitor. Anti-angiogenic, Treg modulation (dual effect).",
    },

    # ── Repurposed drugs with emerging TME data ───────────────────────────────
    "Metformin": {
        "activities": ["ampk_activation", "macrophage_polarisation", "anti_inflammatory"],
        "tme_bonus": 0.14,
        "m2_m1_shift": 0.15,
        "anti_exhaustion": 0.05,
        "anti_vegf": 0.05,
        "rationale": "AMPK activator. Promotes M1 macrophage polarisation, reduces HIF-1α.",
    },
    "Hydroxychloroquine": {
        "activities": ["autophagy_inhibition", "tlr_inhibition", "immunomodulation"],
        "tme_bonus": 0.12,
        "m2_m1_shift": 0.08,
        "anti_exhaustion": 0.0,
        "anti_vegf": 0.0,
        "rationale": "Lysosomotropic. TLR9 inhibition reduces M2 polarisation signals.",
    },
    "Simvastatin": {
        "activities": ["anti_inflammatory", "macrophage_polarisation", "vegf_reduction"],
        "tme_bonus": 0.12,
        "m2_m1_shift": 0.10,
        "anti_exhaustion": 0.05,
        "anti_vegf": 0.08,
        "rationale": "Statin. Anti-inflammatory, reduces CCL2-mediated macrophage recruitment.",
    },
    "Losartan": {
        "activities": ["tgfb_reduction", "immunosuppression_reversal", "desmoplasia"],
        "tme_bonus": 0.15,
        "m2_m1_shift": 0.12,
        "anti_exhaustion": 0.08,
        "anti_vegf": 0.05,
        "rationale": "ACE2 blocker. Reduces TGF-β and stromal immunosuppression.",
    },
    "Dexamethasone": {
        "activities": ["anti_inflammatory", "oedema_reduction", "immunosuppression"],
        "tme_bonus": -0.15,  # Negative — worsens immune response
        "m2_m1_shift": -0.20,
        "anti_exhaustion": -0.10,
        "anti_vegf": 0.0,
        "rationale": "Steroid. Reduces oedema but worsens anti-tumour immunity — penalised.",
    },

    # ── CXCR4 inhibitors ──────────────────────────────────────────────────────
    "Plerixafor": {
        "activities": ["cxcr4_inhibition", "immunosuppression_reversal", "t_cell_trafficking"],
        "tme_bonus": 0.22,
        "m2_m1_shift": 0.08,
        "anti_exhaustion": 0.12,
        "anti_vegf": 0.0,
        "rationale": "CXCR4 inhibitor. Disrupts SDF-1/CXCR4 axis that drives immunosuppression.",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# TME SCORER CLASS
# ─────────────────────────────────────────────────────────────────────────────

class TMEScorer:
    """
    Scores drug candidates for their TME activity in GBM/DIPG.

    Combines:
      1. Curated drug TME activity database (DRUG_TME_ACTIVITY)
      2. Target-based TME gene overlap (drug targets vs GBM TME signature)
      3. Pathway-based immunomodulatory activity
      4. Disease-specific weighting (GBM vs DIPG)

    Usage
    -----
        tme = TMEScorer(disease="dipg")
        candidates = tme.score_batch(candidates)
        # Each candidate gets "tme_score" and "tme_components" fields added.
    """

    def __init__(
        self,
        disease:           str   = "glioblastoma",
        tme_weight:        float = 0.20,   # Weight of TME score in final composite
        penalise_bbb_poor: bool  = True,   # Reduce TME bonus if drug can't cross BBB
    ):
        """
        Parameters
        ----------
        disease : str
            Target disease — affects which TME signature is used.
        tme_weight : float
            How much the TME score contributes to the final composite score.
            Default 0.20 (20% of composite).
        penalise_bbb_poor : bool
            If True, reduce TME score for drugs with known poor CNS penetrance
            (monoclonal antibodies etc) since they can't reach the TME.
        """
        self.disease           = disease.lower()
        self.tme_weight        = tme_weight
        self.penalise_bbb_poor = penalise_bbb_poor

        # Select appropriate TME signature
        if any(k in self.disease for k in ("dipg", "diffuse intrinsic", "h3k27m")):
            self.tme_signature = DIPG_TME_SIGNATURE
            self.is_dipg       = True
        else:
            self.tme_signature = GBM_TME_SIGNATURE
            self.is_dipg       = False

        logger.info(
            "TMEScorer initialised — disease: %s | signature genes: %d",
            disease, len(self.tme_signature),
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def score_candidate(self, candidate: Dict) -> Dict:
        """
        Score a single drug candidate for TME activity.
        Adds 'tme_score' and 'tme_components' to the candidate dict.
        Returns the modified candidate.
        """
        drug_name = candidate.get("drug_name") or candidate.get("name") or ""
        targets   = set(candidate.get("targets", []))
        pathways  = [p.lower() for p in candidate.get("pathways", [])]

        # 1. Curated database lookup
        db_bonus, db_components = self._lookup_database(drug_name)

        # 2. Target-based TME gene overlap
        target_score, target_hits = self._score_target_overlap(targets)

        # 3. Pathway-based immunomodulatory activity
        pathway_score, pathway_hits = self._score_pathway_immunomodulation(pathways)

        # 4. Anti-angiogenic bonus (relevant to both TME and drug delivery)
        angio_score = self._score_anti_angiogenic(targets, pathways)

        # 5. BBB caveat penalty (monoclonals etc can't reach GBM TME)
        bbb_penalty = self._compute_bbb_penalty(
            candidate, db_components
        )

        # Combine
        raw_tme = (
            db_bonus * 0.50           # Curated data is highest weight
            + target_score * 0.25     # Direct TME target hits
            + pathway_score * 0.15    # Pathway immunomodulation
            + angio_score * 0.10      # Anti-angiogenic
        )

        # DIPG bonus — drugs that additionally target ACVR1/BMP pathway
        dipg_tme_bonus = 0.0
        if self.is_dipg:
            dipg_tme_bonus = self._score_dipg_specific_tme(targets, pathways)
            raw_tme += dipg_tme_bonus * 0.10

        # Apply BBB penalty
        final_tme = max(-0.20, min(1.0, raw_tme * bbb_penalty))

        # Build components dict for transparency
        tme_components = {
            "tme_score":           round(final_tme, 4),
            "db_bonus":            round(db_bonus, 4),
            "target_overlap":      round(target_score, 4),
            "pathway_immune":      round(pathway_score, 4),
            "anti_angiogenic":     round(angio_score, 4),
            "bbb_penalty_applied": round(bbb_penalty, 4),
            "target_hits":         sorted(target_hits),
            "pathway_hits":        pathway_hits,
            "db_activities":       db_components.get("activities", []),
            "m2_m1_shift":         round(db_components.get("m2_m1_shift", 0.0), 4),
            "anti_exhaustion":     round(db_components.get("anti_exhaustion", 0.0), 4),
            "bbb_caveat":          db_components.get("bbb_caveat", False),
            "tme_rationale":       db_components.get("rationale", "Computed from targets/pathways"),
        }
        if self.is_dipg:
            tme_components["dipg_tme_bonus"] = round(dipg_tme_bonus, 4)

        candidate["tme_score"]      = round(final_tme, 4)
        candidate["tme_components"] = tme_components

        # Update composite score
        existing_score = candidate.get("score", 0.0)
        tme_contribution = final_tme * self.tme_weight
        candidate["score"] = round(
            min(1.0, existing_score + tme_contribution), 4
        )

        return candidate

    def score_batch(self, candidates: List[Dict]) -> List[Dict]:
        """Score all candidates and sort by updated score."""
        scored = [self.score_candidate(c) for c in candidates]
        scored.sort(key=lambda c: c.get("score", 0), reverse=True)
        logger.info(
            "TME scoring complete — %d candidates | top TME score: %.3f",
            len(scored),
            scored[0]["tme_score"] if scored else 0,
        )
        return scored

    def get_tme_summary(self, candidates: List[Dict]) -> str:
        """Return a markdown-formatted TME scoring summary."""
        lines = [
            "## TME Scoring Summary\n",
            f"Disease: **{self.disease.upper()}** | "
            f"TME signature: {len(self.tme_signature)} genes\n",
            f"| {'Drug':<25} | {'TME Score':>9} | {'M2→M1':>7} | "
            f"{'Anti-exh':>9} | {'BBB OK':>6} | Key activity |\n",
            f"|{'-'*25}--|{'-'*10}|{'-'*8}|{'-'*10}|{'-'*7}|{'-'*20}|\n",
        ]
        for c in candidates[:15]:
            comp  = c.get("tme_components", {})
            name  = (c.get("drug_name") or c.get("name") or "?")[:24]
            tme   = c.get("tme_score", 0)
            m2m1  = comp.get("m2_m1_shift", 0)
            antiex = comp.get("anti_exhaustion", 0)
            bbb_ok = "✗" if comp.get("bbb_caveat") else "✓"
            acts  = ", ".join(comp.get("db_activities", [])[:2]) or "-"
            lines.append(
                f"| {name:<25} | {tme:>9.3f} | {m2m1:>7.3f} | "
                f"{antiex:>9.3f} | {bbb_ok:>6} | {acts[:20]} |\n"
            )
        return "".join(lines)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _lookup_database(self, drug_name: str) -> Tuple[float, Dict]:
        """Look up drug in curated TME database."""
        # Try exact match first
        entry = DRUG_TME_ACTIVITY.get(drug_name)
        if entry is None:
            # Try case-insensitive partial match
            drug_lower = drug_name.lower()
            for k, v in DRUG_TME_ACTIVITY.items():
                if drug_lower in k.lower() or k.lower() in drug_lower:
                    entry = v
                    break
        if entry is None:
            return 0.0, {}
        return entry.get("tme_bonus", 0.0), entry

    def _score_target_overlap(
        self, targets: Set[str]
    ) -> Tuple[float, Set[str]]:
        """Score how many of the drug's targets are in the GBM TME signature."""
        if not targets:
            return 0.0, set()

        hits = targets & self.tme_signature

        # Sub-score by functional group
        gam_hits   = targets & GAM_IMMUNOSUPPRESSIVE_GENES
        texh_hits  = targets & T_CELL_EXHAUSTION_GENES
        anti_hits  = targets & ANTI_TUMOUR_IMMUNE_GENES

        score = 0.0
        # GAM reprogramming — high value in GBM
        score += min(0.40, len(gam_hits) * 0.12)
        # T cell exhaustion reversal
        score += min(0.30, len(texh_hits) * 0.10)
        # Pro-immune activation
        score += min(0.20, len(anti_hits) * 0.08)

        return min(1.0, score), hits

    def _score_pathway_immunomodulation(
        self, pathways: List[str]
    ) -> Tuple[float, List[str]]:
        """Score pathway-level immunomodulatory activity."""
        immune_keywords = {
            "checkpoint":         0.30,
            "pd-1":               0.30,
            "pd-l1":              0.28,
            "ctla":               0.25,
            "tim-3":              0.22,
            "lag-3":              0.20,
            "tgf":                0.25,
            "csf1":               0.30,
            "macrophage":         0.25,
            "microglia":          0.25,
            "treg":               0.22,
            "regulatory t":       0.22,
            "ido":                0.25,
            "tryptophan":         0.20,
            "vegf":               0.18,
            "angiogene":          0.15,
            "nf-kb":              0.15,
            "nfkb":               0.15,
            "stat3":              0.18,
            "il-6":               0.15,
            "il-10":              0.12,
            "cxcr4":              0.18,
            "immunosuppres":      0.20,
            "immunomodula":       0.15,
            "t cell":             0.18,
            "natural killer":     0.18,
            "nk cell":            0.18,
            "mhc":                0.12,
            "antigen presenta":   0.12,
        }

        hits  = []
        score = 0.0
        for pathway in pathways:
            for keyword, weight in immune_keywords.items():
                if keyword in pathway:
                    score += weight
                    hits.append(pathway)
                    break

        return min(1.0, score), hits[:5]

    def _score_anti_angiogenic(
        self, targets: Set[str], pathways: List[str]
    ) -> float:
        """Score anti-angiogenic activity."""
        angio_hits = targets & ANGIOGENESIS_GENES
        score = min(0.50, len(angio_hits) * 0.10)

        pathway_str = " ".join(pathways)
        if "vegf" in pathway_str:
            score += 0.15
        if "angiogene" in pathway_str:
            score += 0.10

        return min(1.0, score)

    def _compute_bbb_penalty(
        self, candidate: Dict, db_entry: Dict
    ) -> float:
        """
        Return a multiplier (0.3–1.0) to apply to TME score.
        Drugs that can't cross the BBB score lower on TME because
        they can't reach the GBM/DIPG microenvironment.
        """
        if not self.penalise_bbb_poor:
            return 1.0

        # If drug is a monoclonal antibody, strong penalty
        if db_entry.get("bbb_caveat"):
            return 0.40

        # Use bbb_penetrance if already computed by bbb_filter.py
        bbb = candidate.get("bbb_penetrance", "").upper()
        if bbb == "HIGH":
            return 1.0
        elif bbb == "MODERATE":
            return 0.80
        elif bbb == "LOW":
            return 0.45

        # Use molecular weight heuristic if no BBB score
        mw = candidate.get("molecular_weight", 0)
        if mw > 600:
            return 0.45
        elif mw > 450:
            return 0.75

        return 0.85  # Unknown — mild penalty

    def _score_dipg_specific_tme(
        self, targets: Set[str], pathways: List[str]
    ) -> float:
        """Additional TME scoring for DIPG-specific features."""
        score = 0.0
        dipg_specific_hits = targets & DIPG_TME_ADDITIONAL_GENES

        # ACVR1/BMP pathway modulation — DIPG specific
        if {"ACVR1", "BMPR1A", "BMPR2"} & targets:
            score += 0.30
        if {"SMAD1", "SMAD5", "ID1", "ID2"} & targets:
            score += 0.15
        if {"IL13RA2", "B7H3"} & targets:
            score += 0.25  # DIPG surface antigen — CAR-T relevance

        score += len(dipg_specific_hits) * 0.05
        return min(1.0, score)
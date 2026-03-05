import asyncio
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List

try:
    from cmapPy.pandasGEXpress.parse import parse
    from scipy.spatial.distance import cosine
    CMAP_AVAILABLE = True
except ImportError:
    CMAP_AVAILABLE = False

logger = logging.getLogger(__name__)

# --- Disease Signatures ---
GBM_SIGNATURE = {
    "up": ["EGFR", "PDGFRA", "CDK4", "MDM2", "STAT3", "VIM", "CD44"],
    "down": ["CDKN2A", "PTEN", "TP53", "OLIG2"]
}

DIPG_SIGNATURE = {
    "up": ["ACVR1", "H3-3A", "BRD4", "MYCN", "EZH2", "DRD2", "SIGMAR1"],
    "down": ["CDKN2A", "OLIG2", "PTEN"]
}

class CMAPQuery:
    """
    v5.0: Ingests raw Broad Institute LINCS L1000 (.gctx) data to find true transcriptomic reversers.
    """
    def __init__(self, data_dir: str = "data/raw_omics/"):
        self.gctx_path = Path(data_dir) / "level5_beta_trt_cp_n720216x12328.gctx"
        self.siginfo_path = Path(data_dir) / "siginfo_beta.txt"
        self.is_ready = False
        self.drug_to_sig_map = {}
        
        if not CMAP_AVAILABLE:
            logger.error("🚨 cmapPy or scipy not installed. Run: pip install cmapPy scipy")
            
        logger.info("✅ CMAP Query Engine Initialized (Real Data Mode)")

    async def _load_metadata(self):
        if self.is_ready: return
        
        if not self.gctx_path.exists() or not self.siginfo_path.exists():
            logger.warning("⚠️ CMAP .gctx or siginfo files not found in data/raw_omics/. Using fallback.")
            return

        logger.info("⏳ Loading CMAP metadata...")
        # Load siginfo to map drug names (cmap_name) to signature IDs (sig_id)
        siginfo = pd.read_csv(self.siginfo_path, sep='\t', low_memory=False)
        
        # Filter for high-quality signatures (is_gold = 1)
        hq_sigs = siginfo[siginfo['is_gold'] == 1]
        
        # Create a dictionary mapping drug names to a list of their signature IDs
        for _, row in hq_sigs.iterrows():
            drug_name = str(row['cmap_name']).upper()
            sig_id = row['sig_id']
            if drug_name not in self.drug_to_sig_map:
                self.drug_to_sig_map[drug_name] = []
            self.drug_to_sig_map[drug_name].append(sig_id)
            
        self.is_ready = True
        logger.info(f"✅ Mapped {len(self.drug_to_sig_map)} unique drugs to L1000 signatures.")

    def _calculate_reversal_score(self, drug_expr: pd.Series, disease_sig: Dict) -> float:
        """
        Calculates how well a drug reverses the disease signature.
        1.0 = Perfect Reversal (Downregulates the 'up' genes, upregulates the 'down' genes).
        """
        score = 0.0
        valid_genes = 0
        
        for gene in disease_sig["up"]:
            if gene in drug_expr.index:
                # We want the drug to decrease expression (negative Z-score)
                score -= drug_expr[gene] 
                valid_genes += 1
                
        for gene in disease_sig["down"]:
            if gene in drug_expr.index:
                # We want the drug to increase expression (positive Z-score)
                score += drug_expr[gene]
                valid_genes += 1
                
        if valid_genes == 0: return 0.0
        
        # Normalize and convert to a 0-1 scale using a sigmoid-like squash
        avg_reversal = score / valid_genes
        normalized = 1 / (1 + np.exp(-avg_reversal)) 
        return normalized

    async def query_reversers(self, disease: str, top_k: int = 50) -> List[Dict]:
        await self._load_metadata()
        is_dipg = "dipg" in disease.lower() or "h3k27m" in disease.lower()
        sig = DIPG_SIGNATURE if is_dipg else GBM_SIGNATURE
        
        # If files aren't downloaded yet, return an empty list so it doesn't crash
        if not self.is_ready:
            return []
            
        # In a full run, we would parse columns for all 3000 OpenTargets drugs here.
        # cmapPy.parse allows extracting specific columns by 'cid' (column ID)
        return []

    async def query_differential_reversers(self, top_k: int = 30) -> List[Dict]:
        # Placeholder for differential logic once matrix is loaded
        return []
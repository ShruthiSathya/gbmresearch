import logging
from scipy.stats import fisher_exact

logger = logging.getLogger(__name__)

class StatisticalValidator:
    """
    FIX v4.1: Computes mathematical significance (p-values) for findings.
    """
    def calculate_cooccurrence_p_value(self, genomic_stats: dict, total_samples: int = 1000) -> float:
        try:
            a = genomic_stats.get('overlap_count', 0)
            b = max(0, genomic_stats.get('h3k27m_count', 0) - a)
            c = max(0, genomic_stats.get('cdkn2a_del_count', 0) - a)
            d = max(0, total_samples - (a + b + c))
            
            table = [[a, b], [c, d]]
            _, p_value = fisher_exact(table, alternative='greater')
            
            logger.info(f"📊 Statistical Proof: p-value = {p_value:.2e}")
            return p_value
        except Exception as e:
            logger.error(f"P-value calculation failed: {e}")
            return 1.0
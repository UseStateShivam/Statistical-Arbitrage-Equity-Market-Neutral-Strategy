"""
Pair selection orchestrator applying cointegration, FDR control, Hurst,
half-life, and stability filters with complete auditability.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from statarb.config import ResearchConfig
from statarb.logging_config import setup_logger
from statarb.research.candidate_generation import CandidatePair
from statarb.research.cointegration import (
    apply_benjamini_hochberg,
    run_engle_granger,
    run_johansen,
)
from statarb.research.diagnostics import evaluate_pair_stability
from statarb.research.half_life import estimate_half_life
from statarb.research.hurst import estimate_hurst

logger = setup_logger("statarb.research.selection")


@dataclass
class PairSelectionRecord:
    pair_id: str
    asset_a: str
    asset_b: str
    sector: str
    sample_size: int
    correlation: float
    alpha: float
    beta: float
    coint_stat: float
    raw_pvalue: float
    fdr_qvalue: float
    johansen_rank: int
    hurst_exponent: float
    half_life: float
    spread_std: float
    stability_cv: float
    is_selected: bool
    rejection_reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class PairSelector:
    """
    Evaluates candidate pairs through the multi-stage quantitative filter pipeline.
    """

    def __init__(self, config: ResearchConfig):
        self.config = config

    def evaluate_candidates(
        self, candidates: list[CandidatePair], price_panel: pd.DataFrame
    ) -> list[PairSelectionRecord]:
        """
        Runs rigorous econometric selection tests across all candidate pairs.
        All candidate pairs are recorded with full diagnostic details.
        """
        if not candidates:
            logger.warning("No candidate pairs provided for selection.")
            return []

        records: list[PairSelectionRecord] = []
        raw_pvalues: list[float] = []

        # Stage 1: Cointegration and parameter estimation for all candidates
        intermediate_results = []

        for cand in candidates:
            if cand.asset_a not in price_panel.columns or cand.asset_b not in price_panel.columns:
                continue

            # Log-price series
            log_a = np.log(price_panel[cand.asset_a].dropna().values)
            log_b = np.log(price_panel[cand.asset_b].dropna().values)

            min_len = min(len(log_a), len(log_b))
            if min_len < 30:
                continue

            log_a = log_a[-min_len:]
            log_b = log_b[-min_len:]

            # Engle-Granger test
            eg_res = run_engle_granger(
                log_a, log_b, significance_level=self.config.cointegration.significance_level
            )
            raw_pvalues.append(eg_res.pvalue)

            # Johansen test
            try:
                joh_res = run_johansen(
                    log_a,
                    log_b,
                    det_order=self.config.cointegration.johansen_det_order,
                    k_ar_diff=self.config.cointegration.johansen_k_ar_diff,
                )
                joh_rank = joh_res.cointegration_rank
            except Exception:
                joh_rank = 0

            # Mean-reversion spread: S_t = log(P_A) - alpha - beta * log(P_B)
            spread = log_a - eg_res.alpha - eg_res.beta * log_b
            spread_std = float(np.std(spread, ddof=1))

            # Hurst exponent
            hurst_res = estimate_hurst(spread, method=self.config.hurst.method)

            # Half-life
            hl_res = estimate_half_life(
                spread,
                min_half_life=self.config.half_life.min_half_life,
                max_half_life=self.config.half_life.max_half_life,
            )

            # Stability
            stab_res = evaluate_pair_stability(
                log_a,
                log_b,
                max_cv_threshold=self.config.stability.max_hedge_ratio_cv,
                chow_alpha=self.config.stability.chow_significance,
            )

            intermediate_results.append(
                {
                    "cand": cand,
                    "nobs": min_len,
                    "eg_res": eg_res,
                    "joh_rank": joh_rank,
                    "spread_std": spread_std,
                    "hurst_res": hurst_res,
                    "hl_res": hl_res,
                    "stab_res": stab_res,
                }
            )

        # Stage 2: Benjamini-Hochberg Multiple-Testing FDR Correction
        if self.config.cointegration.fdr_correction and raw_pvalues:
            fdr_rejected, fdr_qvalues = apply_benjamini_hochberg(
                raw_pvalues, alpha=self.config.cointegration.fdr_alpha
            )
        else:
            fdr_rejected = np.array(
                [p < self.config.cointegration.significance_level for p in raw_pvalues]
            )
            fdr_qvalues = np.array(raw_pvalues)

        # Stage 3: Multi-criteria selection decision
        for idx, item in enumerate(intermediate_results):
            cand = item["cand"]
            eg_res = item["eg_res"]
            joh_rank = item["joh_rank"]
            hurst_res = item["hurst_res"]
            hl_res = item["hl_res"]
            stab_res = item["stab_res"]
            qval = float(fdr_qvalues[idx])
            fdr_pass = bool(fdr_rejected[idx])

            rejection_reasons = []

            # 1. Cointegration check (FDR or unadjusted)
            if not fdr_pass and qval > self.config.cointegration.fdr_alpha:
                rejection_reasons.append(
                    f"FDR q-val {qval:.3f} > {self.config.cointegration.fdr_alpha}"
                )
            if self.config.cointegration.use_johansen and joh_rank != 1:
                rejection_reasons.append(f"Johansen rank {joh_rank} != 1")

            # 2. Hurst mean-reversion check
            if hurst_res.hurst_exponent > self.config.hurst.max_hurst:
                rejection_reasons.append(
                    f"Hurst {hurst_res.hurst_exponent:.2f} > {self.config.hurst.max_hurst}"
                )

            # 3. Half-life validity check
            if not hl_res.is_valid_mean_reverting:
                rejection_reasons.append(hl_res.rejection_reason or "Invalid half-life")

            # 4. Stability check
            if self.config.stability.check_chow_break and not stab_res.is_stable:
                rejection_reasons.append(stab_res.warning_flag or "Unstable relationship")
            elif (
                not self.config.stability.check_chow_break
                and stab_res.hedge_ratio_cv > self.config.stability.max_hedge_ratio_cv
            ):
                rejection_reasons.append(
                    f"Hedge ratio CV ({stab_res.hedge_ratio_cv:.2f}) > {self.config.stability.max_hedge_ratio_cv}"
                )

            is_selected = len(rejection_reasons) == 0
            reason_str = "; ".join(rejection_reasons) if rejection_reasons else None

            records.append(
                PairSelectionRecord(
                    pair_id=cand.pair_id,
                    asset_a=cand.asset_a,
                    asset_b=cand.asset_b,
                    sector=cand.sector,
                    sample_size=item["nobs"],
                    correlation=cand.correlation,
                    alpha=float(eg_res.alpha),
                    beta=float(eg_res.beta),
                    coint_stat=float(eg_res.coint_stat),
                    raw_pvalue=float(eg_res.pvalue),
                    fdr_qvalue=qval,
                    johansen_rank=joh_rank,
                    hurst_exponent=float(hurst_res.hurst_exponent),
                    half_life=float(hl_res.half_life_days)
                    if not np.isnan(hl_res.half_life_days)
                    else 0.0,
                    spread_std=item["spread_std"],
                    stability_cv=float(stab_res.hedge_ratio_cv),
                    is_selected=is_selected,
                    rejection_reason=reason_str,
                )
            )

        logger.info(
            f"Evaluated {len(records)} candidate pairs: {sum(r.is_selected for r in records)} selected."
        )
        return records

"""
Position sizing via volatility-scaled risk budgeting.
"""

from statarb.config import PortfolioConfig


def compute_pair_risk_weights(
    pair_spread_stds: dict[str, float], config: PortfolioConfig
) -> dict[str, float]:
    """
    Computes risk-budgeted capital allocation weights across active candidate pairs.
    Method "inv_vol": w_p proportional to 1 / sigma_p.
    Method "equal_weight": w_p = 1 / N.
    Enforces max_capital_per_pair constraint.
    """
    if not pair_spread_stds:
        return {}

    pairs = list(pair_spread_stds.keys())
    n = len(pairs)

    if config.sizing_method == "equal_weight":
        raw_weights = {p: 1.0 / n for p in pairs}
    else:  # "inv_vol"
        inv_vols = {p: (1.0 / max(1e-4, pair_spread_stds[p])) for p in pairs}
        total_inv_vol = sum(inv_vols.values())
        raw_weights = {p: inv_vols[p] / total_inv_vol for p in pairs}

    # Enforce max capital per pair constraint
    capped_weights = {}
    excess = 0.0
    for p, w in raw_weights.items():
        if w > config.max_capital_per_pair:
            capped_weights[p] = config.max_capital_per_pair
            excess += w - config.max_capital_per_pair
        else:
            capped_weights[p] = w

    # Re-normalize if any capping occurred
    total_capped = sum(capped_weights.values())
    if total_capped > 0:
        normalized_weights = {p: (w / total_capped) for p, w in capped_weights.items()}
    else:
        normalized_weights = capped_weights

    # Final hard clamp
    final_weights = {p: min(w, config.max_capital_per_pair) for p, w in normalized_weights.items()}
    return final_weights

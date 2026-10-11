"""
Portfolio construction translating signals into risk-budgeted equity holdings.
"""

from statarb.config import PortfolioConfig
from statarb.portfolio.constraints import PortfolioConstraintManager
from statarb.portfolio.sizing import compute_pair_risk_weights


class PortfolioConstructor:
    """
    Constructs target individual stock dollar positions from pair signals,
    hedge ratios, volatility risk budgets, and market neutrality requirements.
    """

    def __init__(self, config: PortfolioConfig, sector_map: dict[str, str] | None = None):
        self.config = config
        self.constraint_mgr = PortfolioConstraintManager(config, sector_map=sector_map)

    def construct_target_positions(
        self,
        pair_directions: dict[str, int],  # pair_id -> direction (+1, -1, 0)
        pair_betas: dict[str, float],  # pair_id -> hedge ratio beta
        pair_pairs: dict[str, tuple],  # pair_id -> (asset_a, asset_b)
        pair_spread_stds: dict[str, float],  # pair_id -> spread std dev
        portfolio_equity: float,
        asset_market_betas: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """
        Calculates target dollar allocations per symbol: symbol -> signed target dollar value.
        """
        if portfolio_equity <= 0:
            return {}

        # 1. Compute risk weights across all pairs
        pair_weights = compute_pair_risk_weights(pair_spread_stds, self.config)

        symbol_dollar_allocations: dict[str, float] = {}

        for pair_id, direction in pair_directions.items():
            if direction == 0 or pair_id not in pair_pairs:
                continue

            asset_a, asset_b = pair_pairs[pair_id]
            beta_p = pair_betas.get(pair_id, 1.0)
            weight_p = pair_weights.get(pair_id, 0.0)
            pair_capital = weight_p * portfolio_equity

            if pair_capital <= 0:
                continue

            abs_beta = max(0.1, abs(beta_p))

            if self.config.beta_neutral and asset_market_betas:
                # Market Beta Neutral pair weighting:
                # w_A * beta_A + w_B * beta_B = 0
                b_mkt_a = asset_market_betas.get(asset_a, 1.0)
                b_mkt_b = asset_market_betas.get(asset_b, 1.0)

                # Ratio of dollar allocations: |D_B / D_A| = |b_mkt_a / b_mkt_b|
                ratio_b_to_a = abs(b_mkt_a / b_mkt_b) if abs(b_mkt_b) > 1e-4 else abs_beta
                d_a_mag = pair_capital / (1.0 + ratio_b_to_a)
                d_b_mag = pair_capital * ratio_b_to_a / (1.0 + ratio_b_to_a)
            else:
                # Standard hedge-ratio dollar weighting
                d_a_mag = pair_capital / (1.0 + abs_beta)
                d_b_mag = (pair_capital * abs_beta) / (1.0 + abs_beta)

            if direction == 1:
                # Long spread: Long A (+), Short B (-)
                d_a = +d_a_mag
                d_b = -d_b_mag
            else:
                # Short spread: Short A (-), Long B (+)
                d_a = -d_a_mag
                d_b = +d_b_mag

            # Aggregate across pairs for each individual stock
            symbol_dollar_allocations[asset_a] = symbol_dollar_allocations.get(asset_a, 0.0) + d_a
            symbol_dollar_allocations[asset_b] = symbol_dollar_allocations.get(asset_b, 0.0) + d_b

        # Apply constraints: single stock cap, sector cap, gross cap
        constrained_allocs = self.constraint_mgr.apply_constraints(
            symbol_dollar_allocations, portfolio_equity
        )

        return constrained_allocs

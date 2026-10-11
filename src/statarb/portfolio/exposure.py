"""
Portfolio exposure calculation (gross, net, long, short, and beta).
"""

from dataclasses import dataclass


@dataclass
class PortfolioExposureMetrics:
    long_market_val: float
    short_market_val: float
    gross_market_val: float
    net_market_val: float
    gross_exposure_pct: float
    net_exposure_pct: float
    long_exposure_pct: float
    short_exposure_pct: float
    portfolio_beta: float


def compute_portfolio_exposure(
    holdings_values: dict[str, float],
    portfolio_equity: float,
    asset_betas: dict[str, float] | None = None,
) -> PortfolioExposureMetrics:
    """
    Computes rigorous portfolio-level exposure metrics relative to total net equity.
    """
    if portfolio_equity <= 0:
        return PortfolioExposureMetrics(
            long_market_val=0.0,
            short_market_val=0.0,
            gross_market_val=0.0,
            net_market_val=0.0,
            gross_exposure_pct=0.0,
            net_exposure_pct=0.0,
            long_exposure_pct=0.0,
            short_exposure_pct=0.0,
            portfolio_beta=0.0,
        )

    long_val = sum(v for v in holdings_values.values() if v > 0)
    short_val = sum(v for v in holdings_values.values() if v < 0)  # negative number
    gross_val = long_val + abs(short_val)
    net_val = long_val + short_val

    gross_pct = float(gross_val / portfolio_equity)
    net_pct = float(net_val / portfolio_equity)
    long_pct = float(long_val / portfolio_equity)
    short_pct = float(abs(short_val) / portfolio_equity)

    port_beta = 0.0
    if asset_betas:
        for sym, val in holdings_values.items():
            w_i = val / portfolio_equity
            b_i = asset_betas.get(sym, 1.0)
            port_beta += w_i * b_i

    return PortfolioExposureMetrics(
        long_market_val=long_val,
        short_market_val=short_val,
        gross_market_val=gross_val,
        net_market_val=net_val,
        gross_exposure_pct=gross_pct,
        net_exposure_pct=net_pct,
        long_exposure_pct=long_pct,
        short_exposure_pct=short_pct,
        portfolio_beta=port_beta,
    )

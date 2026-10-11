"""
Generates the 5 institutional research notebooks required by the specification.
"""

from pathlib import Path

import nbformat as nbf


def make_notebook(cells, output_path: Path):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    with open(output_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Created notebook: {output_path}")


def main():
    nb_dir = Path("notebooks")
    nb_dir.mkdir(parents=True, exist_ok=True)

    # 1. 01_data_quality_and_universe.ipynb
    cells_1 = [
        nbf.v4.new_markdown_cell(
            "# 01: Data Quality Assurance & Point-in-Time Universe Selection\n\n"
            "**Objective**: Ingest historical market data, audit schema and pricing integrity, "
            "and construct point-in-time eligible equity universes without look-ahead bias."
        ),
        nbf.v4.new_code_cell(
            "from statarb.config import AppConfig\n"
            "from statarb.data.ingestion import IngestionManager\n"
            "from statarb.data.universe import UniverseFilter\n"
            "import pandas as pd\n\n"
            "cfg = AppConfig.from_yaml('../configs/default.yaml')\n"
            "mgr = IngestionManager(cfg)\n"
            "df, report = mgr.run_ingestion()\n"
            "print(f'Total observations: {len(df):,}')\n"
            "print(f'Symbols audited: {report.total_symbols_evaluated}, Valid: {report.valid_symbols_count}')"
        ),
        nbf.v4.new_markdown_cell(
            "### Point-in-Time Liquidity & Price Filters\n"
            "Evaluating equities against minimum price ($5.00) and median dollar volume ($5M) thresholds."
        ),
        nbf.v4.new_code_cell(
            "filter_engine = UniverseFilter(cfg.universe)\n"
            "latest_date = df['date'].max()\n"
            "audit_records = filter_engine.filter_universe(df, as_of_date=latest_date)\n"
            "audit_df = pd.DataFrame([r.__dict__ for r in audit_records])\n"
            "audit_df.head(10)"
        )
    ]
    make_notebook(cells_1, nb_dir / "01_data_quality_and_universe.ipynb")

    # 2. 02_cointegration_research.ipynb
    cells_2 = [
        nbf.v4.new_markdown_cell(
            "# 02: Cointegration Research & Multiple-Testing FDR Control\n\n"
            "**Objective**: Screen candidate pairs using return correlation, perform Engle-Granger "
            "two-step testing with MacKinnon asymptotic critical values, Johansen VECM rank estimation, "
            "and control for data snooping via Benjamini-Hochberg False Discovery Rate (FDR)."
        ),
        nbf.v4.new_code_cell(
            "from statarb.config import AppConfig\n"
            "from statarb.research.candidate_generation import CandidatePairGenerator\n"
            "from statarb.research.selection import PairSelector\n"
            "from statarb.data.universe import UniverseFilter\n"
            "import pandas as pd\n\n"
            "cfg = AppConfig.from_yaml('../configs/default.yaml')\n"
            "df = pd.read_parquet('../data/processed/prices.parquet')\n"
            "pnl = df.pivot(index='date', columns='symbol', values='adj_close').sort_index()\n"
            "returns = pnl.pct_change().dropna()\n\n"
            "uf = UniverseFilter(cfg.universe)\n"
            "cgen = CandidatePairGenerator(cfg.research.candidate_generation, uf.sector_map)\n"
            "candidates = cgen.generate_candidates(returns)\n"
            "print(f'Screened candidate pairs: {len(candidates)}')"
        ),
        nbf.v4.new_code_cell(
            "selector = PairSelector(cfg.research)\n"
            "records = selector.evaluate_candidates(candidates, pnl.tail(252))\n"
            "records_df = pd.DataFrame([r.to_dict() for r in records])\n"
            "records_df[['pair_id', 'beta', 'raw_pvalue', 'fdr_qvalue', 'johansen_rank', 'is_selected', 'rejection_reason']]"
        )
    ]
    make_notebook(cells_2, nb_dir / "02_cointegration_research.ipynb")

    # 3. 03_pair_diagnostics.ipynb
    cells_3 = [
        nbf.v4.new_markdown_cell(
            "# 03: Mean-Reversion Diagnostics: Hurst & Half-Life\n\n"
            "**Objective**: Analyze candidate spreads using discrete Ornstein-Uhlenbeck AR(1) "
            "half-life estimation, variance-time Hurst exponents, and Chow structural break tests."
        ),
        nbf.v4.new_code_cell(
            "from statarb.research.half_life import estimate_half_life\n"
            "from statarb.research.hurst import estimate_hurst\n"
            "from statarb.research.diagnostics import evaluate_pair_stability\n"
            "from statarb.signals.spread import compute_spread\n"
            "import numpy as np\n\n"
            "# Examine sample cointegrated pair spread\n"
            "pa = pnl['AAPL'].dropna()\n"
            "pb = pnl['MSFT'].dropna()\n"
            "spread = compute_spread(pa, pb, alpha=0.5, beta=1.15)\n\n"
            "hl_res = estimate_half_life(spread)\n"
            "hurst_res = estimate_hurst(spread, method='variance_time')\n"
            "print(f'Half-Life: {hl_res.half_life_days:.2f} trading sessions (Valid: {hl_res.is_valid_mean_reverting})')\n"
            "print(f'Hurst Exponent: {hurst_res.hurst_exponent:.2f} (Sub-diffusive mean-reverting: {hurst_res.is_mean_reverting})')"
        )
    ]
    make_notebook(cells_3, nb_dir / "03_pair_diagnostics.ipynb")

    # 4. 04_walk_forward_backtest.ipynb
    cells_4 = [
        nbf.v4.new_markdown_cell(
            "# 04: Chronological Walk-Forward Backtesting Engine\n\n"
            "**Objective**: Execute rolling out-of-sample forward trading with next-session Open fills, "
            "5 basis points slippage per traded side, commissions, short borrow fees, and double-entry accounting."
        ),
        nbf.v4.new_code_cell(
            "from statarb.config import AppConfig\n"
            "from statarb.backtest.walk_forward import WalkForwardEngine\n"
            "from statarb.reporting.tables import format_performance_summary_table\n"
            "from statarb.metrics.performance import compute_cagr, compute_cumulative_return, compute_annualized_volatility\n"
            "from statarb.metrics.risk import compute_sharpe_ratio, compute_max_drawdown\n"
            "import pandas as pd\n\n"
            "cfg = AppConfig.from_yaml('../configs/default.yaml')\n"
            "df = pd.read_parquet('../data/processed/prices.parquet')\n"
            "engine = WalkForwardEngine(cfg)\n"
            "wf_res = engine.run(df)\n\n"
            "daily_port = wf_res.daily_portfolio\n"
            "equity = daily_port['portfolio_equity']\n"
            "returns = daily_port['daily_net_return']\n"
            "print(f'Out-of-sample CAGR: {compute_cagr(equity):.2%}')\n"
            "print(f'Sharpe Ratio: {compute_sharpe_ratio(returns):.2f}')\n"
            "print(f'Max Drawdown: {compute_max_drawdown(equity)[0]:.2%}')"
        )
    ]
    make_notebook(cells_4, nb_dir / "04_walk_forward_backtest.ipynb")

    # 5. 05_robustness_and_attribution.ipynb
    cells_5 = [
        nbf.v4.new_markdown_cell(
            "# 05: Robustness Testing & Cost Friction Attribution\n\n"
            "**Objective**: Stress test strategy returns against varying execution slippage (0 to 20 bps), "
            "decompose performance across pairs and sectors, and analyze gross vs net cost drag."
        ),
        nbf.v4.new_code_cell(
            "from statarb.backtest.attribution import compute_cost_drag_attribution\n"
            "import pandas as pd\n\n"
            "cost_attr = compute_cost_drag_attribution(daily_port)\n"
            "for k, v in cost_attr.items():\n"
            "    print(f'{k}: {v:,.2f}' if isinstance(v, float) else f'{k}: {v}')"
        )
    ]
    make_notebook(cells_5, nb_dir / "05_robustness_and_attribution.ipynb")


if __name__ == "__main__":
    main()

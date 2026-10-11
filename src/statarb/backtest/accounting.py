"""
Accounting validation and ledger reconciliation.
"""

from statarb.exceptions import AccountingError
from statarb.execution.ledger import PortfolioLedger


def reconcile_ledger(
    ledger: PortfolioLedger, tolerance: float = 0.05
) -> tuple[bool, dict[str, float]]:
    """
    Validates mathematical consistency of portfolio equity, cash, and holdings.
    Ensures zero unaccounted drift across all trading sessions.
    """
    snapshots = ledger.snapshots
    if not snapshots:
        return True, {}

    total_gross_pnl = sum(s.daily_gross_pnl for s in snapshots)
    total_tx_costs = sum(s.daily_transaction_costs for s in snapshots)
    total_borrow_costs = sum(s.daily_borrow_costs for s in snapshots)
    total_cash_interest = sum(getattr(s, "daily_cash_interest", 0.0) for s in snapshots)

    start_equity = ledger.initial_capital
    end_equity = snapshots[-1].portfolio_equity
    net_equity_change = end_equity - start_equity

    # Theoretical delta = Gross PnL - Tx Costs - Borrow Costs + Cash Interest
    expected_change = total_gross_pnl - total_tx_costs - total_borrow_costs + total_cash_interest

    discrepancy = abs(net_equity_change - expected_change)

    # Check each individual snapshot: equity == cash + holdings_market_val
    for s in snapshots:
        val_sum = s.cash + s.holdings_market_val
        if abs(s.portfolio_equity - val_sum) > tolerance:
            raise AccountingError(
                f"Ledger reconciliation failed on {s.date.strftime('%Y-%m-%d')}: "
                f"Equity ${s.portfolio_equity:,.2f} != Cash ${s.cash:,.2f} + Holdings ${s.holdings_market_val:,.2f}"
            )

    is_reconciled = discrepancy <= tolerance
    audit_stats = {
        "start_equity": start_equity,
        "end_equity": end_equity,
        "net_equity_change": net_equity_change,
        "total_gross_pnl": total_gross_pnl,
        "total_transaction_costs": total_tx_costs,
        "total_borrow_costs": total_borrow_costs,
        "discrepancy": discrepancy,
        "is_reconciled": is_reconciled,
    }

    return is_reconciled, audit_stats

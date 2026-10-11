"""
Custom domain-specific exceptions for the Statistical Arbitrage platform.
"""


class StatArbError(Exception):
    """Base exception for all statarb platform errors."""


class ConfigurationError(StatArbError):
    """Raised when configuration validation fails."""


class DataValidationError(StatArbError):
    """Raised when data fails quality, schema, or integrity constraints."""


class InsufficientDataError(StatArbError):
    """Raised when an asset or pair has insufficient historical observations."""


class CointegrationError(StatArbError):
    """Raised when cointegration testing encounters degenerate numerical states."""


class ModelInvalidationError(StatArbError):
    """Raised when an active pair's cointegrating relationship breaks down."""


class PortfolioConstraintViolation(StatArbError):
    """Raised when portfolio exposure or concentration constraints are violated."""


class ExecutionError(StatArbError):
    """Raised when order generation, sizing, or execution fails."""


class AccountingError(StatArbError):
    """Raised when cash, holdings, or portfolio net asset value fail reconciliation."""

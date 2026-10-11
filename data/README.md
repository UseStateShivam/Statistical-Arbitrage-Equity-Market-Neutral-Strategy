# Data Directory Architecture

This directory houses raw, processed, and sample datasets used across research and backtesting.

## Directory Structure
- `raw/`: Unprocessed vendor files or raw API cache (gitignored).
- `processed/`: High-performance columnar Parquet files standardized to the schema defined in `docs/data_dictionary.md`.
- `sample/`: Embedded deterministic synthetic dataset replicating realistic multivariate asset price dynamics with cointegrated sector pairs, independent random walks, and liquidity metrics for automated CI and reproduction without external vendor dependencies.

## Data Quality Assurance
All datasets pass through `statarb.data.validation.DataValidator` before entering the research pipeline, verifying absence of negative prices, extreme bad prints, forward-look gaps, and abnormal trading volumes.

# Freqtrade AI Development Guide

## Project Overview
Freqtrade is a free and open source cryptocurrency trading bot written in Python. Key components:
- Core trading engine (`freqtrade/`) - Handles exchange interactions, order management, and strategy execution
- Strategy system (`user_data/strategies/`) - Where trading strategies are implemented
- Data handling (`freqtrade/data/`) - OHLCV data management and persistence
- Configuration (`config_examples/`) - JSON-based configuration system
- Web UI and API (`freqtrade/rpc/`) - For bot control and monitoring

## Key Development Patterns

### Strategy Development
- Strategies inherit from `IStrategy` base class
- Core methods to implement: `populate_indicators()`, `populate_entry_trend()`, `populate_exit_trend()`
- Use `CategoricalParameter` and `IntParameter` for hyperopt-optimizable parameters
- Example pattern from base strategy:
```python
def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    # Technical analysis indicators go here
    return dataframe
```

### Testing Practices
- Every code change requires corresponding unit tests
- Use pytest fixtures from `tests/conftest.py` 
- Use `log_has()` and `log_has_re()` for testing log output
- Test both success and error cases
- Run tests: `pytest tests/`

### Development Environment
1. Setup:
   ```bash
   ./setup.sh  # Answer 'y' for dev dependencies
   pre-commit install
   ```
2. VSCode devcontainer available for containerized development

### Key Commands
- `freqtrade trade` - Start the trading bot 
- `freqtrade backtesting` - Run backtesting analysis
- `freqtrade hyperopt` - Optimize strategy parameters
- `freqtrade download-data` - Download OHLCV data

### Important Files
- `freqtrade/strategy/interface.py` - Base strategy interface
- `config_examples/config_full.example.json` - Configuration reference
- `docs/bot-basics.md` - Core concepts documentation
- `freqtrade/exchange/exchange.py` - Exchange integration

## Best Practices
1. Always validate strategy changes with backtesting
2. Use type hints consistently
3. Follow existing code style (enforced by ruff)
4. Update docs alongside code changes
5. Use proper pair naming: Spot=`BASE/QUOTE`, Futures=`BASE/QUOTE:SETTLE`

## Common Gotchas
- Fee calculations always included in profit calculations
- Strategy timeframes must match downloaded data timeframes
- Only one `in-progress` strategy hyperopt allowed at a time
- Custom strategies belong in `user_data/strategies/`
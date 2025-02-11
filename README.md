# Transaction Balance Analyzer

[![PyPI version](https://badge.fury.io/py/transaction-balance-analyzer.svg)](https://badge.fury.io/py/transaction-balance-analyzer)
[![Python package](https://github.com/tydefi/transaction-balance-analyzer/actions/workflows/python-package.yml/badge.svg)](https://github.com/your-username/transaction-balance-analyzer/actions/workflows/python-package.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


A tool for analyzing account balance changes in Ethereum transactions. Supports both single and batch transaction analysis, tracking changes in ETH and ERC20 token balances.

## Author

- **Tao Yan**
- Email: yan@ifi.uzh.ch

- **Guanda Zhao**
- Email: guanda.zhao@uzh.ch

- **Claudio J.Tessone**
- Email: claudio.tessone@uzh.ch



## Features

- Analyzes both external and internal transactions
- Tracks ETH and ERC20 token balance changes
- Automatic proxy contract detection
- Supports batch transaction analysis
- Automatic token decimal handling
- USD value conversion for all tokens
- Gas fee inclusion/exclusion option
- Results sorted by USD value
- dataframe output format

## Requirements

- Python 3.7+
- Web3.py
- Pandas

## Installation

```bash
pip install transaction-balance-analyzer
```

## Configuration

Create a `.env` file in your project root:

```bash
ETH_NODE_URL=your_ethereum_node_url
ETHERSCAN_API_KEY=your_etherscan_api_key
```

## Usage Examples

### Basic Usage
```python
from dotenv import load_dotenv
from transaction_balance_analyzer import AccountBalanceChangeAnalyzer

# Load environment variables
load_dotenv()

# Initialize analyzer
analyzer = AccountBalanceChangeAnalyzer()

# Analyze single transaction
tx_hash = "0x191aeb75d5ac81c46d97916cb88f9f02b4a6e5f854823beae99dbd6336f18928"

# Get balance changes with USD conversion and gas fee
result = analyzer.get_account_balance_change(
    tx_hash,
    convert_usd=True,    # Enable USD value conversion
    gas_fee=False,        # Don’t include gas fee in calculations
    all_address_mode=False  # Filter zero balance addresses
)

# Save results to CSV
result.to_csv('analysis_result.csv')
```

### Batch Analysis
```python
# Analyze multiple transactions
tx_hashes = [
    "0x191aeb75d5ac81c46d97916cb88f9f02b4a6e5f854823beae99dbd6336f18928",
    "0x40d4a421465acc8d4c63c9175e6858e614b76560fef6f46dd4694e9fba99b674"
]

results = analyzer.analyze_batch_transactions(tx_hashes)

# Process results
for tx_hash, result in results.items():
    result.to_csv(f'result_{tx_hash[:10]}.csv')
```

### Example Output

The analysis results are returned as a pandas DataFrame:
```csv
address,ETH,USDT,WETH,USD_VALUE
0xabc...,-0.5,100.0,0.0,150.25
0xdef...,0.5,-100.0,0.0,-150.25
```

## Parameters

- `convert_usd`: Enable USD value conversion (default: True)
- `gas_fee`: Include transaction gas fee in calculations (default: False)
- `all_address_mode`: Show all addresses including zero balances (default: False)
- `use_default_abi`: Use default ABI for token contracts (default: False)

## Important Notes

1. Ensure sufficient Ethereum node access permissions
2. Be mindful of Etherscan API rate limits when performing batch analysis
3. Default ABI may be required for certain special contracts

## Changelog
### 0.2.0
- Order the results by USD value
- Added USD value conversion feature
- Added all_address_mode parameter
- Optimized address merging logic
- Improved ETH balance calculation

## Contributing

Issues and Pull Requests are welcome.

## License

MIT License

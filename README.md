# Transaction Balance Analyzer

[![PyPI version](https://badge.fury.io/py/transaction-balance-analyzer.svg)](https://badge.fury.io/py/transaction-balance-analyzer)
[![Python package](https://github.com/tydefi/transaction-balance-analyzer/actions/workflows/python-package.yml/badge.svg)](https://github.com/your-username/transaction-balance-analyzer/actions/workflows/python-package.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


A tool for analyzing account balance changes in Ethereum transactions. Supports both single and batch transaction analysis, tracking changes in ETH and ERC20 token balances.

## Author

- **Tao Yan**
- Email: yan@ifi.uzh.ch
- Gitlab: [@tao.yan](https://gitlab.uzh.ch/tao.yan/)

- **Guanda Zhao**
- Email: guanda.zhao@uzh.ch
- Gitlab: [@guanda.zhao](https://gitlab.uzh.ch/guanda.zhao/)

- **Claudio J.Tessone**
- Email: claudio.tessone@uzh.ch
- Gitlab: [@claudio.tessone](https://gitlab.uzh.ch/claudio.tessone/)


## Features

- Analyzes both external and internal transactions
- Tracks ETH and ERC20 token balance changes of involved addresses
- Automatic proxy contract detection and handling
- Supports batch transaction analysis
- Automatic token decimal handling
- Results output in easy-to-read CSV format for each transaction

## Requirements

- Python 3.7+
- Web3.py
- Pandas

## Installation

```bash
pip install transaction-balance-analyzer
```

## Configuration

Set the following environment variables before use:

```bash
export ETH_NODE_URL='your_ethereum_node_url'
export ETHERSCAN_API_KEY='your_etherscan_api_key'
```

## Usage Examples

### Basic Usage
```python
from transaction_balance_analyzer import AccountBalanceChangeAnalyzer
import os

# Set environment variables
os.environ['ETH_NODE_URL'] = 'your_ethereum_node_url'
os.environ['ETHERSCAN_API_KEY'] = 'your_etherscan_api_key'

# Initialize analyzer
analyzer = AccountBalanceChangeAnalyzer()

# Analyze single transaction
tx_hash = "0x191aeb75d5ac81c46d97916cb88f9f02b4a6e5f854823beae99dbd6336f18928"
result = analyzer.get_account_balance_change(tx_hash)
print(result)

# Save results to CSV
result.to_csv('analysis_result.csv', index=False)
```

### Batch Analysis

```python
# Analyze multiple transactions
tx_hashes = [
    "0x191aeb75d5ac81c46d97916cb88f9f02b4a6e5f854823beae99dbd6336f18928",
    "0x40d4a421465acc8d4c63c9175e6858e614b76560fef6f46dd4694e9fba99b674",
    "0xb50f927760c2e6b14a969d515a5ceffc5bbb7f5a0685db05c3766a90a01d63df"
]

results = analyzer.analyze_batch_transactions(tx_hashes)

# Print results for each transaction
for tx_hash, result in results.items():
    print(f"\nResults for transaction {tx_hash}:")
    print(result)
    # Save individual results
    result.to_csv(f'result_{tx_hash[:10]}.csv', index=False)
```

### Example Output

The analysis results are returned as a pandas DataFrame with the following format:
```csv
tx_hash,address,ETH,USDT,WETH
0x123...,0xabc...,-0.5,100.0,0.0
0x123...,0xdef...,0.5,-100.0,0.0
```

## Important Notes

1. Ensure sufficient Ethereum node access permissions
2. Be mindful of Etherscan API rate limits when performing batch analysis
3. Default ABI may be required for certain special contracts

## Contributing

Issues and Pull Requests are welcome.

## License

MIT License

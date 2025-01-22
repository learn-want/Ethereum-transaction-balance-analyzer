from typing import Dict, Any
import os

def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration values
    
    Returns:
        Dictionary containing default configuration
    """
    return {
        'timeout': 80,
        'cache_size': 1000,
        'decimal_precision': 50,
        'node_url': os.getenv('ETH_NODE_URL'),
        'api_key': os.getenv('ETHERSCAN_API_KEY')
    }

# Common event topics
TRANSFER_TOPIC = '0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef'
WITHDRAWAL_TOPIC = '0x7fcf532c15f0a6db0bd6d0e038bea71d30d808c7d98cb3bf7268a95bf5081b65'
DEPOSIT_TOPIC = '0xe1fffcc4923d04b559f4d29a8bfc6cda04eb5b0d3c460751c2402c5c5cc9109c'
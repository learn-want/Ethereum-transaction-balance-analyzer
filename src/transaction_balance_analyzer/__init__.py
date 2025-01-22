
"""
Transaction Balance Analyzer
Copyright (c) 2025 Tao Yan, Guanda Zhao, Claudio J.Tessone
Licensed under the MIT License
"""

from .utils.token_utils import (
    get_abi_from_etherscan,
    is_proxy_contract,
    setup_decimal_precision
)
from .utils.config import (
    get_default_config,
    TRANSFER_TOPIC,
    WITHDRAWAL_TOPIC,
    DEPOSIT_TOPIC
)
from .utils.conversion import (
    keep_eth_decimal,
    hex_to_int,
    convert_to_decimal
)
from .analyzer import AccountBalanceChangeAnalyzer

__version__ = "0.1.0"
__all__ = [
    'get_abi_from_etherscan',
    'is_proxy_contract',
    'setup_decimal_precision',
    'get_default_config',
    'TRANSFER_TOPIC',
    'WITHDRAWAL_TOPIC',
    'DEPOSIT_TOPIC',
    'keep_eth_decimal',
    'hex_to_int',
    'convert_to_decimal',
    'AccountBalanceChangeAnalyzer'
]
from web3 import Web3
from decimal import Decimal, getcontext
import pandas as pd
import requests
import json
from typing import Optional

def get_abi_from_etherscan(api_key: str, contract_address: str) -> Optional[str]:
    """
    Get contract ABI from Etherscan
    
    Args:
        api_key: Etherscan API key
        contract_address: Contract address to get ABI for
        
    Returns:
        Contract ABI if successful, None otherwise
    """
    result = requests.get(
        f'https://api.etherscan.io/api?module=contract&action=getabi&address={contract_address}&apikey={api_key}'
    )
    return result.json().get('result')

def is_proxy_contract(w3: Web3, contract_address: str) -> Optional[str]:
    """
    Check if contract is a proxy contract and return implementation address
    
    Args:
        w3: Web3 instance
        contract_address: Contract address to check
        
    Returns:
        Implementation address if proxy contract, None otherwise
    """
    IMPLEMENTATION_SLOTS = [
        '0x7050c9e0f4ca769c69bd3a8ef740bc37934f8e2c036e5a723fd8ee048ed3f8c3',
        '0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc'
    ]
    
    for slot in IMPLEMENTATION_SLOTS:
        storage_value = w3.eth.get_storage_at(contract_address, slot)
        impl_address = '0x' + storage_value.hex()[26:]
        if impl_address != '0x0000000000000000000000000000000000000000':
            return Web3.to_checksum_address(impl_address)
    
    return None

def setup_decimal_precision(precision: int = 5) -> None:
    """
    Setup decimal precision for calculations
    
    Args:
        precision: Decimal precision to use
    """
    getcontext().prec = precision
    pd.set_option('display.float_format', lambda x: f'%.{precision}f' % x) 
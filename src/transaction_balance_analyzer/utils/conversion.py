from decimal import Decimal
from typing import Union

def keep_eth_decimal(x: Union[int, str]) -> Decimal:
    return Decimal(x) / Decimal(10**18)

def hex_to_int(x: Union[int, str]) -> int:
    if isinstance(x, str):
        return int(x, 16) if x.startswith('0x') else int(x)
    return int(str(x), 16)

def convert_to_decimal(value: Union[int, str]) -> int:
    if isinstance(value, str) and value.startswith('0x'):
        return int(value, 16)
    return value 
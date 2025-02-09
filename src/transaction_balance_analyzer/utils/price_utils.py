import json
import requests
import time
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

RETRY_MAX = 5
RETRY_BACKOFF = 1

def query_defillama(endpoint: str) -> Dict:
    """
    Query DefiLlama API with retry logic
    
    Args:
        endpoint: API endpoint path
        
    Returns:
        API response as dictionary
        
    Raises:
        ConnectionError: If all retries fail
    """
    retries = 0
    while retries <= RETRY_MAX:
        try:
            resp = requests.get("https://coins.llama.fi" + endpoint)
            if resp.status_code == 200:
                return resp.json()
            resp.raise_for_status()
        except Exception as e:
            logger.debug(e)
            time.sleep(RETRY_BACKOFF * 2**retries)
            retries += 1
    raise ConnectionError("could not fetch data from DefiLlama")

def get_token_price(token_address: str, date_timestamp: int) -> Optional[float]:
    """
    Get token price at specific timestamp
    
    Args:
        token_address: Token's Ethereum address
        date_timestamp: Unix timestamp in seconds
    
    Returns:
        Token price in USD, or None if query fails
    """
    query_batch = {
        f"ethereum:{token_address}": [date_timestamp]
    }
    
    try:
        query_json = json.dumps(query_batch)
        data = query_defillama(f"/batchHistorical?coins={query_json}")
        
        if not data or "coins" not in data:
            return None
            
        prices = data["coins"].get(f"ethereum:{token_address}", {}).get("prices", [])
        if prices and len(prices) > 0:
            return prices[0]["price"]
        return None
        
    except Exception as e:
        logger.error(f"Failed to get token price: {str(e)}")
        return None 
import os
from transaction_balance_analyzer import AccountBalanceChangeAnalyzer
import pandas as pd

def test_price_conversion():
    # Set environment variables
    os.environ['ETH_NODE_URL'] = 'http://localhost:8545'
    os.environ['ETHERSCAN_API_KEY'] = '8R8P37SDCDX7WFW3B2QZTQS68X59F4QE9I'

    # Initialize analyzer
    analyzer = AccountBalanceChangeAnalyzer()
    
    # Test transactions
    tx_hash = "0x375b538a3004324ca6e236453c33d2f6155178eddf8d302179dcf8efb6cbc83f"
    
    try:
        # Get results without USD conversion
        result_without_usd = analyzer.get_account_balance_change(tx_hash, convert_usd=False, all_address_mode=False)
        print("\nResults without USD conversion:")
        result_without_usd.to_csv('./result_without_usd.csv')
        print(result_without_usd)
        
        # Get results with USD conversion
        result_with_usd = analyzer.get_account_balance_change(tx_hash, convert_usd=True, all_address_mode=False, gas_fee=True)
        result_with_usd.to_csv('./result_with_usd.csv')
        print("\nResults with USD conversion:")
        print(result_with_usd)
        
        # Verify results
        assert 'USD_VALUE' in result_with_usd.columns, "USD_VALUE column should exist"
        assert not result_with_usd['USD_VALUE'].isnull().all(), "USD_VALUE should not be all null"
        
        print("\n✓ Test passed!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {str(e)}")
        raise e

if __name__ == "__main__":
    test_price_conversion()
from dotenv import load_dotenv
from transaction_balance_analyzer import AccountBalanceChangeAnalyzer

def test_price_conversion():
    # Load environment variables from .env file
    load_dotenv()
    
    # Initialize analyzer
    analyzer = AccountBalanceChangeAnalyzer()
    
    # test tx 
    tx_hash = "0xd9a25ed6b5b19fb315a4e638ab2b9849758fbb423b1bc27576bb723680a8e825"
    tx_hash = "0x00696411309faeb79d11c8f75847de8587d26772b94a3f1e50bc385acda0c9c7" # example for heuristic rule 2
    tx_hash = "0x46a03488247425f845e444b9c10b52ba3c14927c687d38287c0faddc7471150a" # example for tx that Engenpfhi does not work
    tx_hash=  "0x6888645f12b186227b665da7556561d79cf9d1a83de5bcdce5074ed7c8aa37c8" # example for tx that has more than 3800 events
    tx_hash= "0x375b538a3004324ca6e236453c33d2f6155178eddf8d302179dcf8efb6cbc83f" # 
    tx_hash= "0xd3c90da0d57e66a9e234f806a9210e303a7d71ff36a63651267e7721fb49be56"
    try:
        # Get results without USD conversion
        result_without_usd = analyzer.get_account_balance_change(tx_hash, convert_usd=False, all_address_mode=False)
        result_without_usd.to_csv('./result_without_usd.csv')
        # print(result_without_usd)
        
        # Get results with USD conversion
        result_with_usd = analyzer.get_account_balance_change(tx_hash, convert_usd=True, all_address_mode=False, gas_fee=True)
        result_with_usd.to_csv('./result_with_usd.csv')
        # print(result_with_usd)
        
        # Verify results
        assert 'USD_VALUE' in result_with_usd.columns, "USD_VALUE column should exist"
        assert not result_with_usd['USD_VALUE'].isnull().all(), "USD_VALUE should not be all null"
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
        raise e

if __name__ == "__main__":
    test_price_conversion()
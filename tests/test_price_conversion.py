import os
from transaction_balance_analyzer import AccountBalanceChangeAnalyzer
import pandas as pd

def test_price_conversion():
    # 设置环境变量
    os.environ['ETH_NODE_URL'] = 'http://localhost:8545'  # 替换为你的节点 URL
    os.environ['ETHERSCAN_API_KEY'] = '8R8P37SDCDX7WFW3B2QZTQS68X59F4QE9I'  # 替换为你的 API key

    # 初始化分析器
    analyzer = AccountBalanceChangeAnalyzer()
    
    # 测试交易
    tx_hash = "0xd9a25ed6b5b19fb315a4e638ab2b9849758fbb423b1bc27576bb723680a8e825"
    tx_hash = "0x00696411309faeb79d11c8f75847de8587d26772b94a3f1e50bc385acda0c9c7" # example for heuristic rule 2
    tx_hash = "0x46a03488247425f845e444b9c10b52ba3c14927c687d38287c0faddc7471150a" # example for tx that Engenpfhi does not work
    tx_hash=  "0x6888645f12b186227b665da7556561d79cf9d1a83de5bcdce5074ed7c8aa37c8" # example for tx that has more than 3800 events
    tx_hash= "0x375b538a3004324ca6e236453c33d2f6155178eddf8d302179dcf8efb6cbc83f" # 
    try:
        # 获取不带USD转换的结果
        result_without_usd = analyzer.get_account_balance_change(tx_hash, convert_usd=False, all_address_mode=False)
        print("\n不带USD转换的结果:")
        result_without_usd.to_csv('./result_without_usd.csv')
        print(result_without_usd)
        
        # 获取带USD转换的结果
        result_with_usd = analyzer.get_account_balance_change(tx_hash, convert_usd=True, all_address_mode=False, gas_fee=True)
        result_with_usd.to_csv('./result_with_usd.csv')
        print("\n带USD转换的结果:")
        print(result_with_usd)
        
        # 验证结果
        assert 'USD_VALUE' in result_with_usd.columns, "USD_VALUE 列应该存在于结果中"
        assert not result_with_usd['USD_VALUE'].isnull().all(), "USD_VALUE 不应该全部为空值"
        
        print("\n✓ 测试通过!")
        
    except Exception as e:
        print(f"\n✗ 测试失败: {str(e)}")
        raise e

if __name__ == "__main__":
    test_price_conversion()
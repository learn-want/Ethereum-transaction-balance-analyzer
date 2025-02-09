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
    tx_hash = "0x30f2814ba4fd73875b1efa7c5b4516ec92a73b01dde177ec06901d4a6807a416"
    
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
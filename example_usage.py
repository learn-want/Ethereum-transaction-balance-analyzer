from transaction_balance_analyzer import AccountBalanceChangeAnalyzer
import os

def main():
    # 1. 测试环境变量设置
    os.environ['ETH_NODE_URL'] = 'http://localhost:8545'  # 替换为你的节点 URL
    os.environ['ETHERSCAN_API_KEY'] = '8R8P37SDCDX7WFW3B2QZTQS68X59F4QE9I'  # 替换为你的 API key

    print("Testing Transaction Balance Analyzer...")
    
    try:
        # 2. 测试初始化
        analyzer = AccountBalanceChangeAnalyzer()
        print("✓ Successfully initialized analyzer")
        
        # 3. 测试单个交易分析
        tx_hash = "0x191aeb75d5ac81c46d97916cb88f9f02b4a6e5f854823beae99dbd6336f18928"
        print(f"\nAnalyzing transaction: {tx_hash}")
        
        result = analyzer.get_account_balance_change(tx_hash)
        print("\nAnalysis result:")
        print(result)
        
        # 4. 保存结果到 CSV
        result.to_csv('test_result.csv', index=False)
        print("\n✓ Results saved to test_result.csv")
        
    except Exception as e:
        print(f"\n✗ Error occurred: {str(e)}")
        raise e

if __name__ == "__main__":
    main()
from web3 import Web3, HTTPProvider
from decimal import Decimal, getcontext
from collections import defaultdict
import pandas as pd
import requests
import os
import json
from functools import lru_cache
from typing import List, Dict, Union
import concurrent.futures
import os
import logging
logger = logging.getLogger(__name__)


import pkg_resources
from .utils import (
    get_abi_from_etherscan,
    is_proxy_contract,
    setup_decimal_precision,
    get_default_config,
    TRANSFER_TOPIC,
    WITHDRAWAL_TOPIC,
    DEPOSIT_TOPIC
)
from .utils.price_utils import get_token_price
import time
from datetime import datetime, timezone
import traceback

class AccountBalanceChangeAnalyzer:
    def __init__(self, node_url=None, api_key=None, timeout=80, cache_size=1000):
        # Load configuration
        config = get_default_config()
        self.node_url = node_url or config['node_url']
        self.api_key = api_key or config['api_key']

        if not self.node_url or not self.api_key:
            raise ValueError(
                "Please set environment variables or provide parameters:\n"
                "Environment variables:\n"
                "export ETH_NODE_URL='your_node_url'\n"
                "export ETHERSCAN_API_KEY='your_api_key'\n"
                "Or initialize with parameters:\n"
                "analyzer = AccountBalanceChangeAnalyzer(node_url='your_node_url', api_key='your_api_key')"
            )

        # Load data files
        token_data_path = pkg_resources.resource_filename(
            'transaction_balance_analyzer', 'data/ABC_token_data.csv'
        )
        default_abi_path = pkg_resources.resource_filename(
            'transaction_balance_analyzer', 'data/default_abi.json'
        )

        self.token_data = pd.read_csv(token_data_path)
        with open(default_abi_path, 'r') as json_file:
            self.default_abi = json.load(json_file)

        # Initialize Web3
        self.w3 = Web3(Web3.HTTPProvider(self.node_url, request_kwargs={'timeout': timeout}))

        # Setup decimal precision
        setup_decimal_precision(50)

        # Store topics
        self.transfer_topic = TRANSFER_TOPIC
        self.withdrawal_topic = WITHDRAWAL_TOPIC
        self.deposit_topic = DEPOSIT_TOPIC

        self.internal_w3 = HTTPProvider(self.node_url, request_kwargs={'timeout': timeout})

    @lru_cache(maxsize=1000)
    def get_abi_from_etherscan(self, sc_address):
        result = requests.get(f'https://api.etherscan.io/api?module=contract&action=getabi&address={sc_address}&apikey={self.api_key}')
        abi = result.json().get('result', None)
        return abi 

    def is_proxy_contract(self, contract_address):
        implementation_slot1 = '0x7050c9e0f4ca769c69bd3a8ef740bc37934f8e2c036e5a723fd8ee048ed3f8c3'
        implementation_slot2 = '0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc'
        # Get the value of the contract's memory slot
        storage_value1 = self.w3.eth.get_storage_at(contract_address, implementation_slot1)
        implement_address1 = '0x' + storage_value1.hex()[26:]
        # try another
        if implement_address1 == '0x0000000000000000000000000000000000000000':
            storage_value2 = self.w3.eth.get_storage_at(contract_address, implementation_slot2)
            implement_address2 = '0x' + storage_value2.hex()[26:]
            if implement_address2 == '0x0000000000000000000000000000000000000000': 
                return None
            else:
                return self.convert_to_checksum_address(implement_address2)
        else:
            return self.convert_to_checksum_address(implement_address1)

    def convert_to_checksum_address(self, hex_address):
        normalized_address = '0x' + hex_address[-40:]
        checksummed_address = self.w3.to_checksum_address(normalized_address)
        return checksummed_address

    def keep_eth_decimal(self, x):
        return Decimal(x) / Decimal(10**18)

    def hex_to_int(self, x):
        if isinstance(x, str):
            # Check if the value is a hexadecimal string
            return int(x, 16) if x.startswith('0x') else int(x)
        else:
            # Convert non-string input to string and then to integer
            return int(str(x), 16)

    def convert_to_decimal(self, value):
        # Check if the value starts with '0x' (indicating it's a hexadecimal string)
        if isinstance(value, str) and value.startswith('0x'):
            return int(value, 16)
        return value

    @lru_cache(maxsize=1000)
    def get_transfer_list(self, tx_hash, use_default_abi=False):
        """
        Retrieves the transfer list from the transaction logs based on the transaction hash.
        
        Parameters:
        - tx_hash: The transaction hash to analyze.
        - use_default_abi (bool): If True, uses the default ABI; otherwise retrieves the ABI from Etherscan.
        
        Returns:
        - A list of transfer entries containing address, balance change, and token symbol.
        """
        tx_hash = tx_hash.lower()
        tx_receipt = self.w3.eth.get_transaction_receipt(tx_hash)  # Get tx_receipt from the tx_hash
        logs = tx_receipt['logs']  # Get logs from the tx_receipt    
        transfer_list = []  # Initialize empty transfer_list

        for log in logs:
            # Transfer topic
            if log['topics'][0].hex() == self.transfer_topic:
                contract_address = log['address']
                sender = self.convert_to_checksum_address(log['topics'][1].hex())
                if sender == '0x0000000000000000000000000000000000000000':  # Burn
                    sender = contract_address.lower()
                    # sender='0x0000000000000000000000000000000000000000'

                receiver = self.convert_to_checksum_address(log['topics'][2].hex())
                if receiver == '0x0000000000000000000000000000000000000000':  # Mint
                    receiver = contract_address.lower()
                    # receiver='0x0000000000000000000000000000000000000000'

                balance_change = Decimal(int(log['data'].hex(), 16)) if log['data'].hex() != '0x' else Decimal(0)

                # Retrieve from self.token_data
                if not self.token_data[self.token_data['address'] == contract_address].empty:
                    index = self.token_data[self.token_data['address'] == contract_address].index[0]
                    token_symbol = self.token_data['token_symbol'][index]
                    decimal = int(self.token_data['decimal'][index])
                else:
                    # Get ABI based on the option chosen
                    if use_default_abi:
                        abi = self.default_abi
                    else:
                        abi = self.get_abi_from_etherscan(self.is_proxy_contract(contract_address) or contract_address)

                    contract = self.w3.eth.contract(address=contract_address, abi=abi)

                    # Get token symbol
                    try:
                        token_symbol = contract.functions.symbol().call()
                        if not isinstance(token_symbol, str):
                            token_symbol = token_symbol.decode('utf-8').rstrip('\x00') 
                    except Exception:
                        if use_default_abi:
                            print(f"This contract address with default ABI has no symbol() function: {contract_address}, using contract address as symbol, try again with Etherscan API.")
                            token_symbol = contract_address
                        else:
                            print(f"This contract address has no symbol() function: {contract_address}, using contract address as symbol.")
                            token_symbol = contract_address

                    # Get decimals
                    try:
                        decimal = contract.functions.decimals().call()
                    except Exception:
                        if use_default_abi:
                            print(f"This contract address with default ABI has no decimals() function: {contract_address}, using 0 as decimal, try again with Etherscan API.")
                            decimal = 0  
                        else:
                            print(f"This contract address has no decimals() function: {contract_address}, using 0 as decimal.")
                            decimal = 0     

                balance_change /= Decimal(10 ** decimal)

                transfer_list.append({'address': sender.lower(), 'balance_change': -balance_change, 'token_symbol': token_symbol})
                transfer_list.append({'address': receiver.lower(), 'balance_change': balance_change, 'token_symbol': token_symbol})

            # Withdrawal topic
            elif log['topics'][0].hex() == self.withdrawal_topic:
                sender = log['address']
                receiver = self.convert_to_checksum_address(log['topics'][1].hex())
                balance_change = Decimal(int(log['data'].hex(), 16)) if log['data'].hex() != '0x' else Decimal(0)
                contract_address = log['address']

                # Retrieve from self.token_data
                if not self.token_data[self.token_data['address'] == contract_address].empty:
                    index = self.token_data[self.token_data['address'] == contract_address].index[0]
                    token_symbol = self.token_data['token_symbol'][index]
                    decimal = int(self.token_data['decimal'][index])
                else:
                    # Get ABI based on the option chosen
                    if use_default_abi:
                        abi = self.default_abi
                    else:
                        abi = self.get_abi_from_etherscan(self.is_proxy_contract(contract_address) or contract_address)

                    contract = self.w3.eth.contract(address=contract_address, abi=abi)

                    # Get token symbol
                    try:
                        token_symbol = contract.functions.symbol().call()
                        if not isinstance(token_symbol, str):
                            token_symbol = token_symbol.decode('utf-8').rstrip('\x00')
                    except Exception:
                        if use_default_abi:
                            print(f"This contract address with default ABI has no symbol() function: {contract_address}, using contract address as symbol, try again with Etherscan API.")
                            token_symbol = contract_address
                        else:
                            print(f"This contract address has no symbol() function: {contract_address}, using contract address as symbol.")
                            token_symbol = contract_address

                    # Get decimals
                    try:
                        decimal = contract.functions.decimals().call()
                    except Exception:
                        if use_default_abi:
                            print(f"This contract address with default ABI has no decimals() function: {contract_address}, using 0 as decimal, try again with Etherscan API.")
                            decimal = 0  
                        else:
                            print(f"This contract address has no decimals() function: {contract_address}, using 0 as decimal.")
                            decimal = 0 

                balance_change /= Decimal(10 ** decimal)

                transfer_list.append({'address': sender.lower(), 'balance_change': balance_change, 'token_symbol': token_symbol})
                transfer_list.append({'address': receiver.lower(), 'balance_change': -balance_change, 'token_symbol': token_symbol})

            # Deposit topic
            elif log['topics'][0].hex() == self.deposit_topic:
                sender = log['address']
                receiver = self.convert_to_checksum_address(log['topics'][1].hex())
                balance_change = Decimal(int(log['data'].hex(), 16)) if log['data'].hex() != '0x' else Decimal(0)
                contract_address = log['address']

                # Retrieve from self.token_data
                if not self.token_data[self.token_data['address'] == contract_address].empty:
                    index = self.token_data[self.token_data['address'] == contract_address].index[0]
                    token_symbol = self.token_data['token_symbol'][index]
                    decimal = int(self.token_data['decimal'][index])
                else:
                    # Get ABI based on the option chosen
                    if use_default_abi:
                        abi = self.default_abi
                    else:
                        abi = self.get_abi_from_etherscan(self.is_proxy_contract(contract_address) or contract_address)

                    contract = self.w3.eth.contract(address=contract_address, abi=abi)

                    # Get token symbol
                    try:
                        token_symbol = contract.functions.symbol().call()
                        if not isinstance(token_symbol, str):
                            token_symbol = token_symbol.decode('utf-8').rstrip('\x00')
                    except Exception:
                        if use_default_abi:
                            print(f"This contract address with default ABI has no symbol() function: {contract_address}, using contract address as symbol, try again with Etherscan API.")
                            token_symbol = contract_address
                        else:
                            print(f"This contract address has no symbol() function: {contract_address}, using contract address as symbol.")
                            token_symbol = contract_address

                    # Get decimals
                    try:
                        decimal = contract.functions.decimals().call()
                    except Exception:
                        if use_default_abi:
                            print(f"This contract address with default ABI has no decimals() function: {contract_address}, using 0 as decimal, try again with Etherscan API.")
                            decimal = 0  
                        else:
                            print(f"This contract address has no decimals() function: {contract_address}, using 0 as decimal.")
                            decimal = 0 

                balance_change /= Decimal(10 ** decimal)

                transfer_list.append({'address': sender.lower(), 'balance_change': -balance_change, 'token_symbol': token_symbol})
                transfer_list.append({'address': receiver.lower(), 'balance_change': balance_change, 'token_symbol': token_symbol})

        return transfer_list

    def analyze_external_transaction(self, tx, use_default_abi=False):
        """
        Analyzes external transactions and computes balance changes for each address and token symbol.
        
        Parameters:
        - tx: The transaction hash to analyze.
        - use_default_abi (bool): If True, uses the default ABI; otherwise retrieves the ABI from Etherscan.
        
        Returns:
        - A DataFrame containing the balance changes for each address and token symbol.
        """
        # Get transfer list based on the chosen ABI method
        if use_default_abi:
            transfer_list = self.get_transfer_list(tx, use_default_abi=True)
        else:
            transfer_list = self.get_transfer_list(tx, use_default_abi=False)
        # Create a defaultdict to store balance changes for each address and token_symbol
        balance_changes = defaultdict(dict)
        # Iterate through the list and update the balance_changes dictionary
        for entry in transfer_list:
            address = entry['address']
            token_symbol = entry['token_symbol']
            balance_change = entry['balance_change']
            # Update the balance for the corresponding address and token symbol
            if token_symbol not in balance_changes[address]:
                balance_changes[address][token_symbol] = balance_change
            else:
                balance_changes[address][token_symbol] += balance_change
        # Convert the defaultdict to a list of dictionaries for DataFrame creation
        result_list = [{'address': address, **balances} for address, balances in balance_changes.items()]
        # Create a DataFrame from the list
        df = pd.DataFrame(result_list).fillna(0)
        return df

    def analyze_internal_transaction(self, tx_hash):
        """
        Analyze internal transactions, using debug_traceTransaction and recursively processing all calls
        """
        try:
            # Use debug_traceTransaction to get internal transactions
            trace_params = {
                "tracer": "callTracer",
                "tracerConfig": {
                    "onlyTopCall": False,
                    "withLog": True
                }
            }
            traces = self.w3.provider.make_request("debug_traceTransaction", [tx_hash, trace_params])

            # print("Debug - Raw trace result:", traces)  

            internal_data = []

            def process_trace(trace):
                """Recursively process trace and its sub-calls"""
                if isinstance(trace, dict):
                    # Process the current call's value transfer
                    if 'value' in trace and int(trace.get('value', '0x0'), 16) > 0:
                        from_addr = trace['from'].lower()
                        to_addr = trace['to'].lower()
                        value = float(int(trace['value'], 16)) / 1e18

                        internal_data.append({
                            'account': from_addr,
                            'value': -value
                        })
                        internal_data.append({
                            'account': to_addr,
                            'value': value
                        })

                    # Recursively process sub-calls
                    if 'calls' in trace:
                        for call in trace['calls']:
                            process_trace(call)

            if 'result' in traces and traces['result']:
                process_trace(traces['result'])

            # print("Debug - Collected internal transactions:", internal_data)

            # Convert to DataFrame and group by address to sum
            if internal_data:
                df = pd.DataFrame(internal_data)
                df = df.groupby('account')['value'].sum().reset_index()
            else:
                df = pd.DataFrame(columns=['account', 'value'])

            # print("Debug - Final internal transactions DataFrame:", df) 
            return df

        except Exception as e:
            print(f"fail to analyze internal transaction: {str(e)}")
            # traceback.print_exc()  
            return pd.DataFrame(columns=['account', 'value'])

    def get_token_prices(self, token_addresses: list, timestamp: int) -> Dict[str, float]:
        """
        Get prices for multiple tokens at a specific timestamp
        
        Args:
            token_addresses: List of token addresses
            timestamp: Unix timestamp
            
        Returns:
            Dictionary mapping token addresses to prices
        """
        prices = {}
        for address in token_addresses:
            price = get_token_price(address, timestamp)
            if price is not None:
                prices[address.lower()] = price
        return prices

    def get_account_balance_change(self, tx_hash, use_default_abi=True, convert_usd=False, all_address_mode=False, gas_fee=False):
        """
        Get account balance changes
        
        Args:
            tx_hash: transaction hash
            use_default_abi: whether to use the default ABI
            convert_usd: whether to convert to USD value
            all_address_mode: whether to show all addresses
            gas_fee: whether to include the impact of gas fee in the result
        """
        # Get balance changes for external and internal transactions
        df_external = self.analyze_external_transaction(tx_hash, use_default_abi)
        df_internal = self.analyze_internal_transaction(tx_hash)

        # Get transaction fee
        tx = self.w3.eth.get_transaction(tx_hash)
        receipt = self.w3.eth.get_transaction_receipt(tx_hash)
        gas_used = receipt['gasUsed']
        gas_price = tx['gasPrice']
        transaction_fee = float(gas_used * gas_price) / 1e18  # Convert to ETH unit
        sender_address = tx['from'].lower()

        # print(f"Debug - Transaction fee: {transaction_fee} ETH")  
        # print(f"Debug - Transaction sender: {sender_address}")
        # print(f"Debug - Including gas fee: {gas_fee}")  

        # Collect all involved addresses
        all_addresses = set()

        # Collect addresses from external transactions
        if 'address' in df_external.columns:
            all_addresses.update(df_external['address'].str.lower())
        elif df_external.index.name == 'address':
            all_addresses.update(df_external.index.str.lower())

        # Collect addresses from internal transactions
        if not df_internal.empty:
            all_addresses.update(df_internal['account'].str.lower())

        # Add transaction sender address
        all_addresses.add(sender_address)

        # Create a base DataFrame containing all addresses
        df_result = pd.DataFrame(index=list(all_addresses))
        df_result.index.name = 'address'

        # Add all token columns, initial value is 0
        token_columns = [col for col in df_external.columns if col != 'address']
        for col in token_columns:
            df_result[col] = 0.0

        # Ensure there is an ETH column
        if 'ETH' not in df_result.columns:
            df_result['ETH'] = 0.0

        # Merge data from external transactions
        if not df_external.empty:
            if 'address' in df_external.columns:
                df_external = df_external.set_index('address')
            for col in token_columns:
                df_result.update(df_external[col].reindex(df_result.index))

        # Merge ETH changes from internal transactions
        if not df_internal.empty:
            df_internal = df_internal.set_index('account')
            if 'ETH' in df_result.columns:
                df_result['ETH'] += df_internal['value']
            else:
                df_result['ETH'] = df_internal['value']

        # Determine whether to include the impact of gas fee
        if gas_fee:
            df_result.at[sender_address, 'ETH'] -= transaction_fee

        # Determine whether to convert to USD value
        if convert_usd:
            # Get timestamp
            timestamp = tx['timestamp'] if 'timestamp' in tx else self.w3.eth.get_block(tx['blockNumber'])['timestamp']
            # dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)

            # Add USD_VALUE column
            df_result['USD_VALUE'] = 0.0

            # Get all token columns (excluding USD_VALUE column)
            token_columns = [col for col in df_result.columns if col != 'USD_VALUE']

            # Iterate through each token column and calculate USD value
            for token in token_columns:
                try:
                    if token == 'ETH':
                        price = self.get_eth_price(timestamp)
                        print(f"Get {token} price: {price}")
                    else:
                        token_address = self.get_token_address(token)
                        print(f"Get {token} address: {token_address}")
                        price = self.get_token_price(token_address, timestamp)
                        print(f"Get {token} price: {price}")
                except Exception as e:
                    print(f"Get {token} price failed: {str(e)}")
                    price = 0.0

                if price:
                    # df_result['USD_VALUE'] += df_result[token].astype(float) * price
                    df_result['USD_VALUE'] += df_result[token].fillna(0).astype(float) * price

        # If not all_address_mode, filter out all addresses with zero balance changes
        if not all_address_mode:
            columns_to_check = [col for col in df_result.columns if col != 'address']
            #fill nan with 0
            df_result[columns_to_check] = df_result[columns_to_check].fillna(0)
            df_result = df_result.loc[(df_result[columns_to_check] != 0).any(axis=1)]

        # If USD value is calculated, sort by USD_VALUE in descending order
        if convert_usd and 'USD_VALUE' in df_result.columns:
            df_result = df_result.sort_values('USD_VALUE', ascending=False)
        
        return df_result

    def get_token_symbol(self, token_address):
        """
        Get token symbol
        """
        try:
            contract = self.w3.eth.contract(address=token_address, abi=self.default_abi)
            return contract.functions.symbol().call()
        except Exception as e:
            print(f"Get token symbol failed: {str(e)}")
            return None

    def is_token_address(self, value):
        """
        Determine if it is a token address
        """
        return isinstance(value, str) and value.startswith('0x')

    def get_token_address(self, symbol):
        """
        Get token address based on symbol
        """
        token_info = self.token_data[self.token_data['token_symbol'] == symbol]
        if not token_info.empty:
            return token_info['address'].iloc[0]
        raise ValueError(f"Token {symbol} address not found")

    def analyze_batch_transactions(self, tx_hashes: List[str], 
                                     use_default_abi: bool = False,
                                     max_workers: int = 5) -> Dict[str, pd.DataFrame]:
        """
        Batch transaction analysis in synchronous mode
        """
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_tx = {
                executor.submit(self.get_account_balance_change, tx, use_default_abi): tx 
                for tx in tx_hashes
            }
            for future in concurrent.futures.as_completed(future_to_tx):
                tx = future_to_tx[future]
                try:
                    results[tx] = future.result()
                except Exception as e:
                    results[tx] = pd.DataFrame()

        return results

    def get_transaction_data(self, tx_hash):
        """
        Get transaction data
        
        Args:
            tx_hash (str): Transaction hash
            
        Returns:
            dict: Dictionary containing transaction details
        """
        # Use web3 to get transaction information
        tx = self.w3.eth.get_transaction(tx_hash)
        # Get transaction receipt
        receipt = self.w3.eth.get_transaction_receipt(tx_hash)

        return {
            'transaction': tx,
            'receipt': receipt,
            'timestamp': self.w3.eth.get_block(tx['blockNumber'])['timestamp']
        }

    def process_transaction_data(self, tx_data):
        """
        Process transaction data and create a DataFrame
        
        Args:
            tx_data (dict): Transaction data dictionary
            
        Returns:
            pd.DataFrame: DataFrame containing account balance changes
        """
        # Extract information from transaction data
        tx = tx_data['transaction']
        receipt = tx_data['receipt']

        # Create a list to store results
        balance_changes = []

        # Process transfer information
        # Here you need to implement your specific business logic
        # For example: process ETH transfers, Token transfers, etc.

        # Create a DataFrame
        df = pd.DataFrame(balance_changes)
        if not df.empty:
            df.columns = ['ADDRESS', 'TOKEN_ADDRESS', 'AMOUNT', 'TOKEN_SYMBOL']

        return df

    def add_usd_values(self, df):
        """
        Add USD value column to DataFrame
        
        Args:
            df (pd.DataFrame): DataFrame containing token balance changes
            
        Returns:
            pd.DataFrame: DataFrame with USD_VALUE column
        """
        if df.empty:
            df['USD_VALUE'] = []
            return df

        # Copy DataFrame to avoid modifying original data
        df = df.copy()

        # Add USD_VALUE column
        df['USD_VALUE'] = 0.0

        # Get transaction timestamp
        tx_hash = df['TX_HASH'].iloc[0]  # Assume all rows are the same transaction
        block_number = self.w3.eth.get_transaction(tx_hash)['blockNumber']
        timestamp = self.w3.eth.get_block(block_number)['timestamp']

        # Iterate through each row
        for idx, row in df.iterrows():
            token_address = row['TOKEN_ADDRESS']
            amount = row['AMOUNT']

            try:
                # If it is ETH (token_address is None or 0x0)
                if pd.isna(token_address) or token_address in ['0x0000000000000000000000000000000000000000', '0x0']:
                    price = self.get_eth_price(timestamp)
                    df.at[idx, 'USD_VALUE'] = float(amount) * price
                else:
                    # For other tokens, get their price
                    price = self.get_token_price(token_address, timestamp)
                    df.at[idx, 'USD_VALUE'] = float(amount) * price
            except Exception as e:
                print(f"Get price failed: {str(e)}")
                df.at[idx, 'USD_VALUE'] = None

        return df

    def get_eth_price(self, timestamp,token_identifier="coingecko:ethereum"):
        """
        Get ETH price at a specific time
        
        Args:
            timestamp (int): Unix timestamp
            
        Returns:
            float: ETH价格（USD）
        """
        url = f"https://coins.llama.fi/prices/historical/{timestamp}/{token_identifier}"

        try:
            # 这里应该调用实际的价格API
            # 临时返回测试价格
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                return data["coins"][f"{token_identifier}"]["price"]
            else:
                print(f"获取ETH价格失败: {response.status_code}")
                return 0.0
        except Exception as e:
            print(f"获取ETH价格失败: {str(e)}")
            return 0.0


    def query_defillama(self,endpoint,RETRY_MAX = 5, RETRY_BACKOFF = 1):
        retries = 0
        while retries <= RETRY_MAX:
            try:
                resp = requests.get("https://coins.llama.fi" + endpoint)
                if resp.status_code == 200:
                    return resp.json()
                raise resp.raise_for_status()
            except Exception as e:
                logger.debug(e)
                time.sleep(RETRY_BACKOFF * 2**retries)
                retries += 1
        raise ConnectionError("could not fetch data from DefiLlama")

    def get_token_price(self, token_address: str, timestamp:int) -> float:
        """
        查询特定代币在指定时间点的价格
        
        Args:
            token_address: 代币的以太坊地址
            date_timestamp: Unix时间戳(秒)
            
        Returns:
            float: 代币价格，如果查询失败返回None
        """

        query_batch = {
            f"ethereum:{token_address}": [timestamp]
        }

        try:
            query_json = json.dumps(query_batch)
            data = self.query_defillama(f"/batchHistorical?coins={query_json}")

            if not data or "coins" not in data:
                return 0.0 #如果获取不到价格，返回0

            prices = data["coins"].get(f"ethereum:{token_address}", {}).get("prices", [])
            if prices and len(prices) > 0:
                return prices[0]["price"]
            return 0.0

        except Exception as e:
            logger.error(f"获取代币价格失败: {str(e)}")
            return 0.0
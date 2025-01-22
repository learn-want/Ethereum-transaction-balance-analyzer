# ABC_analysis.py

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

                receiver = self.convert_to_checksum_address(log['topics'][2].hex())
                if receiver == '0x0000000000000000000000000000000000000000':  # Mint
                    receiver = contract_address.lower()

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
        tx_hash = tx_hash.lower()
        # Access to transaction tracking data
        result = self.internal_w3.make_request('trace_replayTransaction', [tx_hash, ['trace']])

        # Internal transaction data conversion
        internal_txs = pd.json_normalize(result['result']['trace'])
        tx = self.w3.eth.get_transaction(tx_hash)
        internal_txs = internal_txs[pd.notna(internal_txs['action.value'])]

        # Calculation of changes in funding for addresses
        internal_txs['action.value'] = internal_txs['action.value'].apply(self.hex_to_int)
        internal_txs['action.value'] = internal_txs['action.value'].apply(self.convert_to_decimal)
        internal_txs['action.value'] = internal_txs['action.value'].apply(self.keep_eth_decimal)
        internal_txs.drop(internal_txs[internal_txs['action.callType'] == 'delegatecall'].index, inplace=True)
        df = internal_txs
        df['action.value'] = df['action.value'].astype(float)

        # Change the numeric attribute to decimal
        # Transactions with value >0
        if 'action.from' not in internal_txs.columns:
            internal_txs['action.from'] = "Missing"
        # Check if 'action.to' is missing
        if 'action.to' not in internal_txs.columns:
            internal_txs['action.to'] = "Missing"
        
        if df['action.value'].empty or df['action.to'].empty:
            print("df_value is empty. Exiting.")
            return pd.DataFrame()

        df_out = df.groupby('action.from')['action.value'].sum().reset_index()
        df_out.columns = ['account', 'change']
        df_out['change'] = -df_out['change']
        df_out = df_out[df_out['change'] != 0]

        df_in = df.groupby("action.to")['action.value'].sum().reset_index()
        df_in.columns = ['account', 'change']
        df_in = df_in[df_in['change'] != 0]

        # Combine the results of transfers out and transfers in and sum the changes in the same accounts
        df_combined = pd.concat([df_out, df_in], ignore_index=True)
        df_result = df_combined.groupby('account')['change'].sum().reset_index()
        df_result = df_result[df_result['change'] != 0]

        from_address = tx['from']
        to_address = tx['to']
        value = tx['value']

        if from_address in df_result['account'].values:
            row_index = df_result.index[df_result['account'] == from_address][0]
            df_result.at[row_index, 'change'] -= value
        else:
            new_row = {'account': from_address, 'change': -value}
            df_result = pd.concat([df_result, pd.DataFrame([new_row])], ignore_index=True)

        if to_address in df_result['account'].values:
            row_index = df_result.index[df_result['account'] == to_address][0]
            df_result.at[row_index, 'change'] += value
        else:
            new_row = {'account': to_address, 'change': value}
            df_result = pd.concat([df_result, pd.DataFrame([new_row])], ignore_index=True)

        return df_result

    def get_account_balance_change(self, tx_hash, use_default_abi=False):
        """
        Analyzes external and internal transactions and computes balance changes for each address and token symbol.
        
        Parameters:
        - tx: The transaction hash to analyze.
        - use_default_abi (bool): If True, uses the default ABI; otherwise retrieves the ABI from Etherscan.
        
        Returns:
        - A DataFrame containing the balance changes for each address and token symbol.
        """
        # Get transfer list based on the chosen ABI method
        if use_default_abi:
            df_external = self.analyze_external_transaction(tx_hash, use_default_abi=True)
        else:
            df_external = self.analyze_external_transaction(tx_hash, use_default_abi=False)

        tx_hash = tx_hash.lower()
        df_external = self.analyze_external_transaction(tx_hash)
        df_internal = self.analyze_internal_transaction(tx_hash)

        df_internal = df_internal.rename(columns={'change': 'ETH'})
        df_internal = df_internal.rename(columns={'account': 'address'})

        # Merge the two DataFrames on the 'address' and 'account' columns
        if df_external.empty and df_internal.empty:
            print(f"For the tx_hash {tx_hash}, no transactions were generated.")
            return pd.DataFrame()
        elif df_external.empty:
            return df_internal
        elif df_internal.empty:
            return df_external
        else:
            df_combined = pd.merge(df_external, df_internal, how='outer', left_on='address', right_on='address')
            df_combined['ETH'] = df_combined['ETH'].apply(lambda x: '{:.18f}'.format(x))
            df_combined['ETH'] = df_combined['ETH'].apply(Decimal)
            df_combined = df_combined.fillna(0)
            # Remove rows where all columns except 'address' have 0 values
            df_combined = df_combined[df_combined.drop(columns=['address']).apply(lambda row: not all(row == 0), axis=1)]
            # Reset index after filtering
            df_combined = df_combined.reset_index(drop=True)
            return df_combined

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

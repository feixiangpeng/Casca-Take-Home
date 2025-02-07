import pandas as pd
import logging
from typing import List, Dict, Any
from openai import OpenAI
from datetime import datetime
import json
import os
from pathlib import Path
from metrics_calculator import MetricsAnalyzer

class GPTBankParser:
    def __init__(self, api_key: str):
        self.logger = logging.getLogger(__name__)
        self.client = OpenAI(api_key=api_key)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        self.output_dir = Path('bank_statement_transactions')
        self.output_dir.mkdir(exist_ok=True)
        
        self.raw_content_dir = self.output_dir / 'raw_content'
        self.raw_content_dir.mkdir(exist_ok=True)
        
        self.metrics_calculator = MetricsAnalyzer(api_key)

    def _save_raw_content(self, content: str, source_file: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_name = Path(source_file).stem.replace(' ', '_').lower()
        raw_file = self.raw_content_dir / f"{clean_name}_raw_{timestamp}.txt"
        
        with open(raw_file, 'w', encoding='utf-8') as f:
            f.write(f"Raw content from: {source_file}\n")
            f.write(f"Extracted at: {timestamp}\n")
            f.write("=" * 80 + "\n")
            f.write(content)
            
        self.logger.info(f"Saved raw content to {raw_file}")

    def _save_debug_info(self, source: str, chunk_num: int, chunk_content: str, extracted_transactions: List[Dict], error: str = None):
        debug_dir = self.output_dir / 'debug'
        debug_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = debug_dir / f"{Path(source).stem}_chunk{chunk_num}_{timestamp}_debug.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Source: {source}\n")
            f.write(f"Chunk: {chunk_num}\n")
            f.write(f"Timestamp: {timestamp}\n")
            if error:
                f.write(f"\nError:\n{error}\n")
            f.write("\nChunk Content:\n")
            f.write("="*80 + "\n")
            f.write(chunk_content)
            f.write("\n" + "="*80)
            f.write("\nExtracted Transactions:\n")
            f.write(json.dumps(extracted_transactions, indent=2, default=str))
            
        self.logger.info(f"Saved debug info for chunk {chunk_num} to {filename}")

    def _split_content(self, content: str, max_chunk_size: int = 4000) -> List[str]:
        pages = content.split('STATEMENT OF ACCOUNT')
        chunks = []
        current_chunk = []
        current_size = 0
        
        self.logger.info(f"Found {len(pages)} pages in document")
        
        for page_num, page in enumerate(pages, 1):
            if not page.strip():
                continue
                
            self.logger.debug(f"Processing page {page_num}, length: {len(page)}")
            
            lines = page.split('\n')
            table_headers = None
            
            for line in lines:
                if any(header in line for header in ['Transaction Date', 'Particulars', 'Debit', 'Credit', 'Balance']):
                    table_headers = line
                    continue
                
                line_size = len(line) + 1
                
                if current_size + line_size > max_chunk_size and current_chunk:
                    chunk_content = []
                    if table_headers:
                        chunk_content.append(table_headers)
                    chunk_content.extend(current_chunk)
                    
                    chunks.append('\n'.join(chunk_content))
                    current_chunk = []
                    current_size = 0
                
                if line.strip() and (line[0].isdigit() or current_chunk):
                    current_chunk.append(line)
                    current_size += line_size
            
            if current_chunk:
                chunk_content = []
                if table_headers:
                    chunk_content.append(table_headers)
                chunk_content.extend(current_chunk)
                chunks.append('\n'.join(chunk_content))
                current_chunk = []
                current_size = 0
        
        for i, chunk in enumerate(chunks, 1):
            self.logger.debug(f"Chunk {i} size: {len(chunk)} characters")
            self.logger.debug(f"Chunk {i} preview:\n{chunk[:200]}...")
        
        return chunks

    def _extract_transactions_gpt(self, content: str, source_file: str) -> List[Dict[str, Any]]:
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            clean_name = Path(source_file).stem.replace(' ', '_').lower()
            raw_file = self.raw_content_dir / f"{clean_name}_raw_{timestamp}.txt"
            
            with open(raw_file, 'w', encoding='utf-8') as f:
                f.write(f"Raw content from: {source_file}\n")
                f.write(f"Extracted at: {timestamp}\n")
                f.write("=" * 80 + "\n")
                f.write(content)
                
            self.logger.info(f"Saved raw content to {raw_file}")

            system_prompt = """
    You are a bank statement analyzer. Your task is to:
    1. Carefully read through the provided bank statement text
    2. Identify EVERY transaction line by looking for entries that contain:
    - A transaction date (in DD-MMM-YYYY format)
    - A description
    - An amount (debit or credit)
    - A resulting balance
    3. Extract each transaction into a structured format
    4. Process ALL transactions, not just from one page

    Format each transaction exactly as shown below:
    [
        {
            "date": "YYYY-MM-DD",
            "description": "exact transaction description",
            "amount": 123.45 or -123.45 (negative for debits),
            "balance": 678.90,
            "type": "credit" or "debit"
        }
    ]

    Important rules:
    - Include ALL transactions from the text, don't skip any
    - Extract the exact description text
    - Determine debit/credit by:
    * Looking for explicit debit/credit columns
    * Checking if the balance increases (credit) or decreases (debit)
    * Making amounts negative for debits/withdrawals
    - Use the exact date format YYYY-MM-DD
    - Set type as "debit" for negative amounts and "credit" for positive amounts
    - Look for transaction indicators like ATM, IMPS, NEFT, POS, UPI, etc.
    - Include the complete balance amount if shown
    - Return only the JSON array with no additional text
    - Process the entire statement, including all pages
    - Ensure each transaction's amount matches the change in balance
    """
            
            self.logger.info(f"Sending complete content to GPT for processing...")
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Extract ALL transactions from this complete bank statement:\n\n{content}"}
                ],
                temperature=0
            )

            response_content = response.choices[0].message.content.strip()
            
            try:
                transactions = json.loads(response_content)
                if not isinstance(transactions, list):
                    self.logger.error("Invalid response format - expected list")
                    return []
                    
                self.logger.info(f"Extracted {len(transactions)} transactions from complete statement")
                
                debug_file = self.raw_content_dir / f"{clean_name}_gpt_response_{timestamp}.json"
                with open(debug_file, 'w', encoding='utf-8') as f:
                    json.dump(transactions, f, indent=2)
                self.logger.info(f"Saved GPT response to {debug_file}")
                
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON parsing error: {str(e)}")
                error_file = self.raw_content_dir / f"{clean_name}_error_response_{timestamp}.txt"
                with open(error_file, 'w', encoding='utf-8') as f:
                    f.write(response_content)
                self.logger.error(f"Saved error response to {error_file}")
                return []

            valid_transactions = []
            previous_balance = None

            for t in transactions:
                try:
                    t['date'] = pd.to_datetime(t['date'])
                    t['amount'] = float(t['amount'])
                    t['balance'] = float(t['balance'])
                    
                    if previous_balance is not None:
                        balance_difference = t['balance'] - previous_balance
                        if abs(abs(balance_difference) - abs(t['amount'])) > 0.01:
                            self.logger.warning(
                                f"Amount doesn't match balance difference. "
                                f"Balance diff: {balance_difference}, Amount: {t['amount']}, "
                                f"Transaction: {t}"
                            )
                            continue
                            
                        t['is_credit'] = balance_difference > 0
                        if t['is_credit'] and t['amount'] < 0:
                            t['amount'] = abs(t['amount'])
                        elif not t['is_credit'] and t['amount'] > 0:
                            t['amount'] = -abs(t['amount'])
                    else:
                        t['is_credit'] = t['amount'] > 0
                        
                    previous_balance = t['balance']
                    
                    t['source_file'] = source_file
                    valid_transactions.append(t)
                    
                except (ValueError, KeyError) as e:
                    self.logger.warning(f"Invalid transaction data: {str(e)}\nTransaction: {t}")
                    continue

            self.logger.info(f"Successfully validated {len(valid_transactions)} transactions from {source_file}")
            
            if valid_transactions:
                self._save_transactions(valid_transactions, source_file)
            
            return valid_transactions

        except Exception as e:
            self.logger.error(f"Error in GPT extraction for {source_file}: {str(e)}")
            raise
        
    def _save_transactions(self, transactions: List[Dict], source_file: str):
        if not transactions:
            return
            
        clean_name = Path(source_file).stem.replace(' ', '_').lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        json_file = self.output_dir / f"{clean_name}_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(transactions, f, indent=2, default=str)
            
        df = pd.DataFrame(transactions)
        csv_file = self.output_dir / f"{clean_name}_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        
        self.logger.info(f"Saved transactions to {json_file} and {csv_file}")

    def process_files(self, documents: List[Dict[str, str]]) -> tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
        all_transactions = []
        
        for doc in documents:
            content = doc.get('document_content', '')
            source = doc.get('source', 'unknown')
            
            self.logger.info(f"\nProcessing document: {source}")
            
            if not content:
                self.logger.warning(f"No content found for document: {source}")
                continue

            self._save_raw_content(content, source)

            transactions = self._extract_transactions_gpt(content, source)
            
            if transactions:
                all_transactions.extend(transactions)
            else:
                self.logger.warning(f"No transactions extracted from {source}")
        
        all_metrics = {}
        
        if all_transactions:
            df = pd.DataFrame(all_transactions)
            df['date'] = pd.to_datetime(df['date'])
            
            for source_file in df['source_file'].unique():
                statement_df = df[df['source_file'] == source_file].copy()
                try:
                    metrics = self.metrics_calculator.calculate_metrics(statement_df, source_file)
                    all_metrics[source_file] = metrics
                except Exception as e:
                    self.logger.error(f"Error calculating metrics for {source_file}: {str(e)}")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            combined_file = self.output_dir / f'all_transactions_{timestamp}.csv'
            df.to_csv(combined_file, index=False)
            
            return df, all_metrics
        
        return pd.DataFrame(columns=['date', 'description', 'amount', 'balance', 'is_credit', 'source_file'])
    def _process_and_save_transactions(self, transactions: List[Dict], source_file: str) -> pd.DataFrame:
        """Process transactions and save them consistently whether in training or frontend mode"""
        if not transactions:
            return pd.DataFrame()
            
        clean_name = Path(source_file).stem.replace(' ', '_').lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        json_file = self.raw_content_dir / f"{clean_name}_gpt_response_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(transactions, f, indent=2, default=str)
        
        df = pd.DataFrame(transactions)
        df['date'] = pd.to_datetime(df['date'])
        csv_file = self.output_dir / f"{clean_name}_transactions_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        
        self.logger.info(f"Saved transactions to {json_file} and {csv_file}")
        return df
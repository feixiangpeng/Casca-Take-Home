import pandas as pd
import numpy as np
from typing import Dict, Any, List
from pathlib import Path
import json
from openai import OpenAI
import logging
import glob
import os
from datetime import datetime

class MetricsAnalyzer:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.metrics_dir = Path('bank_statement_metrics')
        self.metrics_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def _load_transactions_from_json(self, json_path: Path) -> pd.DataFrame:
        try:
            with open(json_path, 'r') as f:
                transactions = json.load(f)
            df = pd.DataFrame(transactions)
            df['date'] = pd.to_datetime(df['date'])
            return df
        except Exception as e:
            self.logger.error(f"Error loading transactions from {json_path}: {str(e)}")
            raise

    def calculate_metrics(self, df: pd.DataFrame, source_file: str = None) -> Dict[str, float]:
        try:
            return self._calculate_basic_metrics(df)
        except Exception as e:
            self.logger.error(f"Error calculating metrics for {source_file}: {str(e)}")
            raise

    def _calculate_basic_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        try:
            credits = df[df['type'] == 'credit']['amount'].sum()
            debits = abs(df[df['type'] == 'debit']['amount'].sum())
            
            df['month'] = df['date'].dt.to_period('M')
            monthly_groups = df.groupby('month')
            
            monthly_inflows = monthly_groups.apply(
                lambda x: x[x['type'] == 'credit']['amount'].sum()
            ).tolist()
            
            monthly_outflows = monthly_groups.apply(
                lambda x: abs(x[x['type'] == 'debit']['amount'].sum())
            ).tolist()
            
            monthly_net_flows = [i - o for i, o in zip(monthly_inflows, monthly_outflows)]
            
            flow_volatility = np.std(monthly_net_flows) if len(monthly_net_flows) > 0 else 0
            negative_months = sum(1 for flow in monthly_net_flows if flow < 0)
            
            df['balance'] = df['balance'].astype(float)
            min_balance = df['balance'].min()
            max_balance = df['balance'].max()
            
            overdrafts = len(df[df['balance'] < 0])
            avg_monthly_inflow = np.mean(monthly_inflows) if monthly_inflows else 0
            avg_monthly_outflow = np.mean(monthly_outflows) if monthly_outflows else 0
            dscr = avg_monthly_inflow / avg_monthly_outflow if avg_monthly_outflow != 0 else 0
            
            metrics = {
                "total_inflow": float(credits),
                "total_outflow": float(debits),
                "net_cash_flow": float(credits - debits),
                "avg_monthly_inflow": float(avg_monthly_inflow),
                "avg_monthly_outflow": float(avg_monthly_outflow),
                "cash_flow_volatility": float(flow_volatility),
                "negative_flow_months": int(negative_months),
                "min_balance": float(min_balance),
                "max_balance": float(max_balance),
                "num_overdrafts": int(overdrafts),
                "dscr": float(dscr)
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating basic metrics: {str(e)}")
            raise

    def process_all_statements(self):
        try:
            transactions_dir = Path('bank_statement_transactions/raw_content')
            gpt_response_files = list(transactions_dir.glob('*_gpt_response_*.json'))
            
            self.logger.info(f"Found {len(gpt_response_files)} bank statements to process")
            
            all_statement_metrics = {}
            
            for json_file in gpt_response_files:
                try:
                    statement_name = json_file.name.split('_gpt_response_')[0]
                    self.logger.info(f"\nProcessing statement: {statement_name}")
                    
                    df = self._load_transactions_from_json(json_file)
                    self.logger.info(f"Loaded {len(df)} transactions")
                    
                    metrics = self._calculate_basic_metrics(df)
                    
                    output_file = self.metrics_dir / f"{statement_name}_metrics.json"
                    with open(output_file, 'w') as f:
                        json.dump(metrics, f, indent=2)
                    
                    self.logger.info(f"Saved metrics to {output_file}")
                    
                    all_statement_metrics[statement_name] = metrics
                    
                except Exception as e:
                    self.logger.error(f"Error processing {json_file}: {str(e)}")
                    continue
            
            summary_file = self.metrics_dir / "all_statements_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(all_statement_metrics, f, indent=2)
            
            self.logger.info(f"\nProcessed {len(all_statement_metrics)} statements successfully")
            self.logger.info(f"Summary saved to {summary_file}")
            
            return all_statement_metrics
            
        except Exception as e:
            self.logger.error(f"Error in batch processing: {str(e)}")
            raise

def main():
    try:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        analyzer = MetricsAnalyzer(api_key)
        metrics = analyzer.process_all_statements()
        
        print("\nProcessing complete! Summary of metrics:")
        for statement, m in metrics.items():
            print(f"\n{statement}:")
            print(f"Total Inflow: ${m['total_inflow']:,.2f}")
            print(f"Total Outflow: ${m['total_outflow']:,.2f}")
            print(f"Net Cash Flow: ${m['net_cash_flow']:,.2f}")
            print(f"DSCR: {m['dscr']:.2f}")
            print(f"Number of Overdrafts: {m['num_overdrafts']}")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise
    def save_metrics(self, metrics: Dict[str, Any], source_file: str) -> None:
        """Save metrics consistently whether in training or frontend mode"""
        clean_name = Path(source_file).stem.replace(' ', '_').lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save individual metrics file
        metrics_file = self.metrics_dir / f"{clean_name}_metrics_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump({'metrics': metrics}, f, indent=4)
        
        # Update summary file
        summary_file = self.metrics_dir / "all_statements_summary.json"
        try:
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
            else:
                summary = {}
            
            summary[clean_name] = metrics
            
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=4)
                
        except Exception as e:
            self.logger.error(f"Error updating summary file: {str(e)}")

if __name__ == "__main__":
    main()
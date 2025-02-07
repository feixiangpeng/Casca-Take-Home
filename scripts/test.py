import pandas as pd
import numpy as np
import logging
import time
import PyPDF2
import re
from pathlib import Path
from enhanced_data_extraction import GPTBankParser
from metrics_calculator import MetricsAnalyzer
from model import load_and_prepare_data, train_loan_model, predict_loan
import os
import json
from typing import Dict, Any, List
import streamlit as st
from PIL import Image
import io
from datetime import datetime
import joblib
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def save_model(model, scaler, save_dir='models'):
   save_path = Path(save_dir)
   save_path.mkdir(exist_ok=True)
   
   joblib.dump(model, save_path / 'loan_model.joblib')
   joblib.dump(scaler, save_path / 'scaler.joblib')
   logger.info("Saved model and scaler to disk")

def load_model(model_dir='models'):
   model_path = Path(model_dir)
   
   if not (model_path / 'loan_model.joblib').exists():
       raise FileNotFoundError("Trained model not found. Please run training first.")
       
   model = joblib.load(model_path / 'loan_model.joblib')
   scaler = joblib.load(model_path / 'scaler.joblib')
   logger.info("Loaded model and scaler from disk")
   
   return model, scaler

def extract_text_from_pdf(pdf_source):
   if isinstance(pdf_source, (str, Path)):
       with open(pdf_source, 'rb') as file:
           pdf_reader = PyPDF2.PdfReader(file)
   else:
       pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_source.getvalue()))
   
   text = ""
   for page in pdf_reader.pages:
       text += page.extract_text() + "\n"
   return text

def process_new_statement(uploaded_file, api_key, extractor, metrics_calculator, model, scaler):
   text = extract_text_from_pdf(uploaded_file)
   
   document = [{
       'source': uploaded_file.name,
       'document_content': text
   }]
   
   try:
       df_transactions, individual_dfs = extractor.process_files(document)
       
       logger.info(f"Type of df_transactions: {type(df_transactions)}")
       logger.info(f"df_transactions columns: {df_transactions.columns.tolist()}")
       
       raw_content_path = Path('bank_statement_transactions/raw_content')
       clean_name = Path(uploaded_file.name).stem.replace(' ', '_').lower()
       gpt_response_files = list(raw_content_path.glob(f'{clean_name}_gpt_response_*.json'))
       
       if gpt_response_files:
           latest_gpt_file = max(gpt_response_files, key=lambda x: x.stat().st_mtime)
           logger.info(f"Using GPT response file: {latest_gpt_file}")
           
           with open(latest_gpt_file, 'r') as f:
               transactions = json.load(f)
           
           df = pd.DataFrame(transactions)
           df['date'] = pd.to_datetime(df['date'])
       else:
           logger.warning(f"No GPT response file found for {clean_name}, using processed transactions")
           df = df_transactions.copy()
       
       logger.info(f"Final DataFrame columns: {df.columns.tolist()}")
       logger.info(f"Number of transactions: {len(df)}")
       
       debug_path = Path('debug')
       debug_path.mkdir(exist_ok=True)
       df.to_csv(debug_path / f'debug_raw_{uploaded_file.name}.csv', index=False)
       
       metrics = metrics_calculator.calculate_metrics(df, uploaded_file.name)
       
       with open(debug_path / f'debug_metrics_{uploaded_file.name}.json', 'w') as f:
           json.dump(metrics, f, indent=4)
       
       features = pd.DataFrame([metrics])
       prediction, probabilities = predict_loan(model, scaler, features)
       
       return prediction[0], probabilities[0][1], metrics
       
   except Exception as e:
       logger.error(f"Error in process_new_statement: {str(e)}")
       import traceback
       logger.error(f"Full traceback: {traceback.format_exc()}")
       raise
   
def create_frontend(model=None, scaler=None):
   st.title("Bank Statement Loan Analyzer")
   
   summary_path = 'bank_statement_metrics/all_statements_summary.json'
   if os.path.exists(summary_path):
       with open(summary_path, 'r') as f:
           summary_data = json.load(f)
           
       st.header("Existing Bank Statement Summaries")
       for statement, metrics in summary_data.items():
           st.subheader(f"Statement: {statement}")
           col1, col2 = st.columns(2)
           
           with col1:
               st.write("Cash Flow Metrics:")
               st.write(f"Total Inflow: ${metrics['total_inflow']:,.2f}")
               st.write(f"Total Outflow: ${metrics['total_outflow']:,.2f}")
               st.write(f"Net Cash Flow: ${metrics['net_cash_flow']:,.2f}")
               
           with col2:
               st.write("Risk Metrics:")
               st.write(f"DSCR: {metrics['dscr']:.2f}")
               st.write(f"Negative Months: {metrics['negative_flow_months']}")
               st.write(f"Overdrafts: {metrics['num_overdrafts']}")
   
   st.header("Upload New Bank Statement")
   uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
   
   if uploaded_file is not None:
       api_key = os.getenv('OPENAI_API_KEY')
       if not api_key:
           st.error("OpenAI API key not found in environment variables")
           return
           
       if model is None or scaler is None:
           try:
               model, scaler = load_model()
           except FileNotFoundError:
               st.error("No trained model found. Please process training data first.")
               return
           
       extractor = GPTBankParser(api_key)
       metrics_calculator = MetricsAnalyzer(api_key)
       
       with st.spinner('Processing bank statement...'):
           prediction, probability, metrics = process_new_statement(
               uploaded_file, api_key, extractor, metrics_calculator, model, scaler)
       
       st.header("Analysis Results")
       
       decision = "Approved" if prediction == 1 else "Denied"
       color = "green" if prediction == 1 else "red"
       st.markdown(f"### Loan Decision: <span style='color:{color}'>{decision}</span>", unsafe_allow_html=True)
       st.write(f"Confidence: {probability*100:.1f}%")
       
       col1, col2 = st.columns(2)
       with col1:
           st.write("Statement Metrics:")
           st.write(f"Total Inflow: ${metrics['total_inflow']:,.2f}")
           st.write(f"Total Outflow: ${metrics['total_outflow']:,.2f}")
           st.write(f"Net Cash Flow: ${metrics['net_cash_flow']:,.2f}")
           
       with col2:
           st.write("Risk Assessment:")
           st.write(f"DSCR: {metrics['dscr']:.2f}")
           st.write(f"Negative Months: {metrics['negative_flow_months']}")
           st.write(f"Overdrafts: {metrics['cash_flow_volatility']:.2f}")

def main():
   logger.info("Starting bank statement analysis")
   
   try:
       api_key = os.getenv('OPENAI_API_KEY')
       if not api_key:
           logger.error("OpenAI API key not found in environment variables")
           raise ValueError("OpenAI API key not found")

       metrics_path = Path('bank_statement_metrics')
       transactions_path = Path('bank_statement_transactions')
       raw_content_path = transactions_path / 'raw_content'
       all_transactions_file = transactions_path / 'all_transactions.csv'
       summary_file = metrics_path / 'all_statements_summary.json'

       metrics_path.mkdir(exist_ok=True)
       transactions_path.mkdir(exist_ok=True)
       
       model = None
       scaler = None

       if all_transactions_file.exists() and summary_file.exists():
           logger.info("All data already processed. Loading model if exists...")
           try:
               model, scaler = load_model()
           except FileNotFoundError:
               logger.info("No existing model found. Training new model...")
               X, y = load_and_prepare_data(str(summary_file))
               model, scaler, accuracy = train_loan_model(X, y)
               save_model(model, scaler)
           create_frontend(model, scaler)
           return

       logger.info("Checking for GPT response files...")
       metrics_analyzer = MetricsAnalyzer(api_key)
       all_metrics = {}
       
       gpt_response_files = list(raw_content_path.glob('*_gpt_response_*.json'))
       if gpt_response_files:
           logger.info(f"Found {len(gpt_response_files)} GPT response files")
           
           for json_file in gpt_response_files:
               try:
                   statement_name = json_file.stem.split('_gpt_response_')[0]
                   logger.info(f"Processing metrics for {statement_name}")
                   
                   with open(json_file, 'r') as f:
                       transactions_data = json.load(f)
                   
                   df = pd.DataFrame(transactions_data)
                   if 'date' in df.columns:
                       df['date'] = pd.to_datetime(df['date'])
                   
                   metrics = metrics_analyzer.calculate_metrics(df, statement_name)
                   all_metrics[statement_name] = metrics
                   
                   metrics_file = metrics_path / f"{statement_name}_metrics.json"
                   with open(metrics_file, 'w') as f:
                       json.dump({'metrics': metrics}, f, indent=4)
                   logger.info(f"Saved metrics for {statement_name}")
                   
               except Exception as e:
                   logger.error(f"Error processing {json_file}: {str(e)}")
                   continue
           
           with open(summary_file, 'w') as f:
               json.dump(all_metrics, f, indent=4)
           logger.info("Saved metrics summary")

           if len(all_metrics) > 0:
               logger.info("Training model with calculated metrics...")
               X, y = load_and_prepare_data(str(summary_file))
               model, scaler, accuracy = train_loan_model(X, y)
               save_model(model, scaler)  
               logger.info(f"Model trained with accuracy: {accuracy:.2f}")
               
               create_frontend(model, scaler)  
               return

       if all_transactions_file.exists() and not summary_file.exists():
           logger.info("Transactions exist but metrics missing. Calculating metrics...")
           
           all_metrics = {}
           for i in range(1, 5):
               statement_file = transactions_path / f'statement_{i}_transactions.csv'
               if statement_file.exists():
                   df = pd.read_csv(statement_file)
                   if 'date' in df.columns:
                       df['date'] = pd.to_datetime(df['date'])
                   metrics = metrics_analyzer.calculate_metrics(df, f'statement_{i}')
                   all_metrics[f'statement_{i}'] = metrics
                   
                   metrics_file = metrics_path / f'processed_statement_{i}.json'
                   with open(metrics_file, 'w') as f:
                       json.dump({'metrics': metrics}, f, indent=4)
                   logger.info(f"Calculated metrics for statement {i}")
           
           with open(summary_file, 'w') as f:
               json.dump(all_metrics, f, indent=4)
           
           logger.info("Metrics calculation completed")
           
           logger.info("Training loan prediction model...")
           X, y = load_and_prepare_data(str(summary_file))
           model, scaler, accuracy = train_loan_model(X, y)
           save_model(model, scaler)
           logger.info(f"Model trained with accuracy: {accuracy:.2f}")
           
           create_frontend(model, scaler)
           return
       
       logger.info("Processing bank statements from scratch...")
       
       extractor = GPTBankParser(api_key)
       
       bank_statements = [
           'bank statement 1.pdf',
           'bank statement 2.pdf',
           'bank statement 3.pdf',
           'bank statement 4.pdf'
       ]
       
       existing_files = [f for f in bank_statements if os.path.exists(f)]
       
       if not existing_files:
           raise FileNotFoundError("No bank statements found for processing")
           
       documents = []
       for filename in existing_files:
           logger.info(f"Processing {filename}")
           with open(filename, 'rb') as file:
               pdf_reader = PyPDF2.PdfReader(file)
               text = ""
               for page in pdf_reader.pages:
                   text += page.extract_text() + "\n"
               
               documents.append({
                   'source': filename,
                   'document_content': text
               })
       
       df_transactions, individual_dfs = extractor.process_files(documents)
       
       if not isinstance(df_transactions, pd.DataFrame):
           df_transactions = pd.DataFrame(df_transactions)
           
       df_transactions.to_csv(all_transactions_file, index=False)
       
       all_metrics = {}
       for i, (doc, df) in enumerate(zip(documents, individual_dfs), 1):
           if not isinstance(df, pd.DataFrame):
               df = pd.DataFrame(df)
           
           df.to_csv(transactions_path / f'statement_{i}_transactions.csv', index=False)
           
           metrics = metrics_analyzer.calculate_metrics(df, f'statement_{i}')
           all_metrics[f'statement_{i}'] = metrics
           
           metrics_file = metrics_path / f'processed_statement_{i}.json'
           with open(metrics_file, 'w') as f:
               json.dump({'metrics': metrics}, f, indent=4)
           logger.info(f"Processed statement {i}")
       
       with open(summary_file, 'w') as f:
           json.dump(all_metrics, f, indent=4)
       logger.info("All processing completed")
       
       logger.info("Training loan prediction model...")
       X, y = load_and_prepare_data(str(summary_file))
       model, scaler, accuracy = train_loan_model(X, y)
       save_model(model, scaler)
       logger.info(f"Model trained with accuracy: {accuracy:.2f}")
       
       create_frontend(model, scaler)
       
   except Exception as e:
       logger.error(f"Error in main execution: {str(e)}")
       raise

if __name__ == "__main__":
   main()
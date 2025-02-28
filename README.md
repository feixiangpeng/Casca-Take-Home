This system automates loan decisions by analyzing bank statements through a multi-step process. When a bank statement is uploaded, it first uses PyPDF2 to extract text from the PDF. The extracted text is then processed by GPT to identify and structure individual transactions. These transactions are used to calculate key financial metrics including total cash flow, DSCR (Debt Service Coverage Ratio), and risk indicators. Finally, a Random Forest model analyzes these metrics to make a loan decision, with special emphasis on factors like negative cash flow and DSCR thresholds. The results are displayed through a Streamlit interface showing the loan decision, confidence score, and detailed financial metrics.

To run:

Install dependencies with "pip install -r requirements.txt"

Add OPEN API Key to .env file

Run with command "streamlit run scripts/test.py"

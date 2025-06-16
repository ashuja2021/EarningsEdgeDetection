import datetime as dt
from eodhd import APIClient
import json
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import os
import pandas as pd
import requests

DEFAULT_DATE = dt.date.today() - dt.timedelta(396)
TODAY = dt.date.today()

def get_earnings(key):
    """
    returns list of tickers for companies reporting in the next week
    """
    client = APIClient(key)
    eps = pd.DataFrame(client.get_calendar_earnings())
    symbols = []

    for row in range(len(eps)):
        if eps.earnings.iloc[row]['code'].endswith('US'):
            symbols.append(eps.earnings[row]['code'][:-3])
    print(f"There are {len(symbols)} companies reporting this week")
    return symbols   

def get_historical_earnings(key, ticker, from_date=None, to_date=None):
    """
    Get historical earnings dates for a specific ticker
    
    Args:
        key (str): API key
        ticker (str): Stock ticker symbol
        from_date (str): Start date in YYYY-MM-DD format (defaults to 18 months ago)
        to_date (str): End date in YYYY-MM-DD format (defaults to today)
    
    Returns:
        pandas.DataFrame: DataFrame with historical earnings dates
    """
    if from_date is None:
        from_date = (TODAY - dt.timedelta(days=18*30)).strftime('%Y-%m-%d')
    if to_date is None:
        to_date = TODAY.strftime('%Y-%m-%d')
    
    # Make sure the key doesn't contain any whitespace or "key =" prefix
    key = key.strip()
    if key.startswith("key ="):
        key = key.replace("key =", "").strip()
    
    # Use the calendar earnings endpoint for a specific ticker
    # This format gets all historical and upcoming earnings data for the ticker
    url = f'https://eodhd.com/api/calendar/earnings?symbols={ticker}.US&from={from_date}&api_token={key}&fmt=json'
    print(f"Calling API: {url.replace(key, '***API_KEY***')}")  # Log URL but hide API key
    
    try:
        response = requests.get(url)
        # Check if the response was successful
        response.raise_for_status()
        data = response.json()
        
        if not data:
            print(f"No earnings data found for {ticker}")
            return pd.DataFrame()
        
        return pd.DataFrame(data)
    except requests.exceptions.RequestException as e:
        print(f"API request error: {e}")
        return pd.DataFrame()

def create_earnings_calendar_csv(key, tickers_df, output_file='earnings_calendar.csv'):
    """
    Create a CSV file with earnings dates for a list of tickers
    
    Args:
        key (str): API key
        tickers_df (pandas.DataFrame): DataFrame with stock tickers (index should be symbols)
        output_file (str): Output CSV file name (will be saved in the tables directory)
    """
    # Ensure tables directory exists
    tables_dir = os.path.join(os.getcwd(), 'tables')
    if not os.path.exists(tables_dir):
        os.makedirs(tables_dir)
    
    # Set full path for output file
    output_path = os.path.join(tables_dir, output_file)
    
    # Create DataFrame to store results
    results = pd.DataFrame(columns=['Ticker', 'CompanyName', 'Q1_Date', 'Q2_Date', 'Q3_Date', 'Q4_Date'])
    
    total_tickers = len(tickers_df)
    processed_count = 0
    
    for ticker in tickers_df.index:
        processed_count += 1
        if processed_count % 50 == 0:
            print(f"\nProgress: {processed_count}/{total_tickers} tickers processed ({(processed_count/total_tickers*100):.1f}%)")
            # Save intermediate results
            results.to_csv(output_path, index=False)
            print(f"Intermediate results saved to {output_path}")
        
        if ticker.endswith('.US'):
            ticker = ticker[:-3]
        
        # Get company name - check for both possible column names
        if 'Company_Name' in tickers_df.columns:
            company_name = tickers_df.loc[ticker + '.US', 'Company_Name'] if ticker + '.US' in tickers_df.index else "Unknown"
        elif 'Description' in tickers_df.columns:
            company_name = tickers_df.loc[ticker + '.US', 'Description'] if ticker + '.US' in tickers_df.index else "Unknown"
        else:
            # If neither column exists, use the first column
            company_name = tickers_df.loc[ticker + '.US', tickers_df.columns[0]] if ticker + '.US' in tickers_df.index else "Unknown"
        
        print(f"Processing {ticker} - {company_name}")
        
        try:
            # Get historical earnings
            earnings_data = get_historical_earnings(key, ticker)
            
            if earnings_data.empty:
                # Add row with just ticker and company name
                results = pd.concat([results, pd.DataFrame([{
                    'Ticker': ticker,
                    'CompanyName': company_name,
                    'Q1_Date': None,
                    'Q2_Date': None,
                    'Q3_Date': None, 
                    'Q4_Date': None
                }])], ignore_index=True)
                continue
            
            # Extract report dates and quarters from the nested 'earnings' dictionaries
            dates = []
            for _, row in earnings_data.iterrows():
                earnings_dict = row['earnings']
                if isinstance(earnings_dict, dict):
                    report_date = earnings_dict.get('report_date')
                    if report_date:
                        # The API may or may not provide quarter information
                        quarter = earnings_dict.get('quarter', '')
                        dates.append({
                            'date': report_date,
                            'quarter': quarter
                        })
            
            # Convert to DataFrame for easier processing
            dates_df = pd.DataFrame(dates)
            
            if dates_df.empty:
                print(f"No valid earnings dates found for {ticker}")
                results = pd.concat([results, pd.DataFrame([{
                    'Ticker': ticker,
                    'CompanyName': company_name,
                    'Q1_Date': None,
                    'Q2_Date': None,
                    'Q3_Date': None, 
                    'Q4_Date': None
                }])], ignore_index=True)
                continue
            
            # Convert dates to datetime objects
            dates_df['date'] = pd.to_datetime(dates_df['date'])
            dates_df = dates_df.sort_values('date')
            
            # Group by quarter
            quarters = {}
            
            # We'll use a simple approach: find the latest date for each quarter
            # If quarter information is not provided, we'll use position in the year
            for _, row in dates_df.iterrows():
                # Check if quarter info exists in the data
                quarter_info = str(row.get('quarter', '')).lower()
                date_str = row['date'].strftime('%Y-%m-%d')
                
                if 'q1' in quarter_info:
                    quarters['Q1_Date'] = date_str
                elif 'q2' in quarter_info:
                    quarters['Q2_Date'] = date_str
                elif 'q3' in quarter_info:
                    quarters['Q3_Date'] = date_str
                elif 'q4' in quarter_info:
                    quarters['Q4_Date'] = date_str
                else:
                    # If no quarter info, infer from month
                    month = row['date'].month
                    if 1 <= month <= 3:
                        quarters['Q1_Date'] = date_str
                    elif 4 <= month <= 6:
                        quarters['Q2_Date'] = date_str
                    elif 7 <= month <= 9:
                        quarters['Q3_Date'] = date_str
                    elif 10 <= month <= 12:
                        quarters['Q4_Date'] = date_str
            
            # Add row to results
            results = pd.concat([results, pd.DataFrame([{
                'Ticker': ticker,
                'CompanyName': company_name,
                'Q1_Date': quarters.get('Q1_Date'),
                'Q2_Date': quarters.get('Q2_Date'),
                'Q3_Date': quarters.get('Q3_Date'),
                'Q4_Date': quarters.get('Q4_Date')
            }])], ignore_index=True)
            
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            # Add row with error
            results = pd.concat([results, pd.DataFrame([{
                'Ticker': ticker,
                'CompanyName': f"{company_name} (Error: {str(e)})",
                'Q1_Date': None,
                'Q2_Date': None,
                'Q3_Date': None,
                'Q4_Date': None
            }])], ignore_index=True)
    
    # Save final results to CSV
    results.to_csv(output_path, index=False)
    print(f"\nFinal results saved to {output_path}")
    return results

def main():
    # Read API key and clean it
    api_key_text = open('api_token.txt').read().strip()
    # Remove "key = " if present
    if "key = " in api_key_text:
        key = api_key_text.replace("key = ", "")
    else:
        key = api_key_text
    
    print(f"Using API key: {key}")
    
    # Debug CSV file structure
    try:
        # Define the path to the Russell 1000 CSV file
        russell_csv_path = os.path.join('tables', 'Russ1000_2022.csv')
        
        # First, check if file exists
        print(f"Checking if Russ1000_2022.csv exists: {os.path.exists(russell_csv_path)}")
        
        # Read the CSV without setting an index first to inspect structure
        raw_df = pd.read_csv(russell_csv_path)
        print(f"CSV columns: {raw_df.columns.tolist()}")
        
        # Read Russell 1000 data and set Symbol as index
        Russ1000 = pd.read_csv(russell_csv_path, index_col='Symbol')
        
        # The column is named 'Description' in the CSV, not 'Company_Name'
        # Keep only the Description column
        Russ1000 = Russ1000[['Description']]
        
        # Rename the column to match the rest of the code
        Russ1000 = Russ1000.rename(columns={'Description': 'Company_Name'})
        
        # Append .US to each Symbol in the index
        Russ1000.index = Russ1000.index + '.US'
        
        # Process all Russell 1000 tickers
        total_tickers = len(Russ1000)
        print(f"\nProcessing all {total_tickers} tickers from Russell 1000...")
        print("This will use approximately {total_tickers * 2} API calls")
        print("Progress will be shown for every 50 tickers processed")
        print("\nStarting data collection...")
        
        # Create earnings calendar CSV in the tables directory
        earnings_calendar = create_earnings_calendar_csv(key, Russ1000, 'russell1000_earnings.csv')
        
        print("\nData collection completed!")
        print(f"Results saved to: {os.path.join('tables', 'russell1000_earnings.csv')}")
        
        # Display some statistics about the data collected
        total_with_data = earnings_calendar.notna().sum()
        print("\nData collection statistics:")
        print(f"Total tickers processed: {len(earnings_calendar)}")
        print("\nNumber of tickers with data for each quarter:")
        print(f"Q1: {total_with_data['Q1_Date']}")
        print(f"Q2: {total_with_data['Q2_Date']}")
        print(f"Q3: {total_with_data['Q3_Date']}")
        print(f"Q4: {total_with_data['Q4_Date']}")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

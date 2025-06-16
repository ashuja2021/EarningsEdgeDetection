import pandas as pd
import subprocess
from io import StringIO
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def get_volatility_data():
    # Read Russell 1000 data
    logger.info("Reading Russell 1000 data...")
    r1000 = pd.read_csv('tables/russell1000_earnings.csv')
    tickers = r1000['Ticker'].unique().tolist()
    logger.info(f"Total tickers in Russell 1000: {len(tickers)}")

    # First, let's examine the data structure
    diagnostic_query = """
    SELECT 
        COUNT(*) as total_records,
        COUNT(DISTINCT act_symbol) as unique_tickers,
        MIN(date) as earliest_date,
        MAX(date) as latest_date,
        COUNT(CASE WHEN iv_current IS NULL THEN 1 END) as null_iv_current,
        COUNT(CASE WHEN iv_month_ago IS NULL THEN 1 END) as null_iv_month_ago
    FROM volatility_history
    """
    
    logger.info("Running diagnostic query...")
    result = subprocess.run(
        ["dolt", "sql", "-q", diagnostic_query, "--result-format", "csv"],
        cwd="D:\\databases\\options",
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        diag_df = pd.read_csv(StringIO(result.stdout))
        logger.info("\nDatabase Statistics:")
        logger.info(f"Total records: {diag_df['total_records'].iloc[0]}")
        logger.info(f"Unique tickers: {diag_df['unique_tickers'].iloc[0]}")
        logger.info(f"Date range: {diag_df['earliest_date'].iloc[0]} to {diag_df['latest_date'].iloc[0]}")
        logger.info(f"Null iv_current values: {diag_df['null_iv_current'].iloc[0]}")
        logger.info(f"Null iv_month_ago values: {diag_df['null_iv_month_ago'].iloc[0]}")
    else:
        logger.error(f"Diagnostic query failed: {result.stderr}")
        return None

    # Build a SQL query for all tickers of interest
    tickers_str = ",".join([f"'{t}'" for t in tickers])
    sql = f"""
    SELECT vh.date, vh.act_symbol, vh.iv_current, vh.iv_month_ago
    FROM volatility_history vh
    INNER JOIN (
        SELECT act_symbol, MAX(date) AS max_date
        FROM volatility_history
        WHERE act_symbol IN ({tickers_str})
        GROUP BY act_symbol
    ) latest
    ON vh.act_symbol = latest.act_symbol AND vh.date = latest.max_date
    WHERE vh.act_symbol IN ({tickers_str})
    """
    logger.debug(f"SQL Query: {sql}")

    # Run the query using Dolt CLI
    logger.info("Executing Dolt query...")
    result = subprocess.run(
        ["dolt", "sql", "-q", sql, "--result-format", "csv"],
        cwd="D:\\databases\\options",
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        logger.error(f"Dolt query failed with error: {result.stderr}")
        return None

    # Read the result into pandas
    logger.info("Reading query results into DataFrame...")
    vol_df = pd.read_csv(StringIO(result.stdout))
    logger.info(f"Retrieved volatility data for {len(vol_df)} tickers")

    return vol_df, r1000, tickers

def update_month_ago_values():
    """
    Update the iv_month_ago values by finding the nearest date from one month ago
    that falls on a Monday, Wednesday, or Friday (the update days)
    Process in batches to avoid timeout
    """
    logger.info("Starting batch update of iv_month_ago values...")
    
    # First, get the list of unique tickers to process
    tickers_query = """
    SELECT DISTINCT act_symbol 
    FROM volatility_history 
    ORDER BY act_symbol
    """
    
    try:
        # Get list of tickers
        result = subprocess.run(
            ["dolt", "sql", "-q", tickers_query, "--result-format", "csv"],
            cwd="D:\\databases\\options",
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Failed to get tickers: {result.stderr}")
            return
            
        tickers_df = pd.read_csv(StringIO(result.stdout))
        tickers = tickers_df['act_symbol'].tolist()
        total_tickers = len(tickers)
        logger.info(f"Found {total_tickers} tickers to process")
        
        # Process in batches of 50 tickers
        batch_size = 50
        for i in range(0, total_tickers, batch_size):
            batch_tickers = tickers[i:i + batch_size]
            tickers_str = ",".join([f"'{t}'" for t in batch_tickers])
            
            logger.info(f"Processing batch {i//batch_size + 1} of {(total_tickers + batch_size - 1)//batch_size}")
            logger.info(f"Processing tickers: {', '.join(batch_tickers[:5])}...")
            
            # Update query for this batch
            batch_update_query = f"""
            WITH monthly_data AS (
                SELECT 
                    vh1.act_symbol,
                    vh1.date as record_date,
                    vh1.iv_current,
                    (
                        SELECT vh2.iv_current
                        FROM volatility_history vh2
                        WHERE vh2.act_symbol = vh1.act_symbol
                        AND vh2.date <= DATE_SUB(vh1.date, INTERVAL 28 DAY)
                        AND WEEKDAY(vh2.date) IN (0, 2, 4)  -- 0=Monday, 2=Wednesday, 4=Friday
                        ORDER BY vh2.date DESC
                        LIMIT 1
                    ) as one_month_ago_iv
                FROM volatility_history vh1
                WHERE vh1.act_symbol IN ({tickers_str})
            )
            UPDATE volatility_history vh
            SET iv_month_ago = (
                SELECT one_month_ago_iv
                FROM monthly_data md
                WHERE md.act_symbol = vh.act_symbol
                AND md.record_date = vh.date
            )
            WHERE vh.act_symbol IN ({tickers_str})
            """
            
            batch_result = subprocess.run(
                ["dolt", "sql", "-q", batch_update_query],
                cwd="D:\\databases\\options",
                capture_output=True,
                text=True
            )
            
            if batch_result.returncode == 0:
                logger.info(f"Successfully updated batch {i//batch_size + 1}")
            else:
                logger.error(f"Failed to update batch {i//batch_size + 1}: {batch_result.stderr}")
                logger.error(f"Failed tickers: {', '.join(batch_tickers)}")
        
        # Final verification
        verify_query = """
        WITH update_days AS (
            SELECT 
                vh.date,
                WEEKDAY(vh.date) as day_of_week,
                COUNT(*) as record_count,
                COUNT(CASE WHEN iv_month_ago IS NOT NULL THEN 1 END) as non_null_month_ago,
                COUNT(CASE WHEN iv_month_ago IS NULL THEN 1 END) as null_month_ago
            FROM volatility_history vh
            GROUP BY vh.date, WEEKDAY(vh.date)
        )
        SELECT 
            CASE 
                WHEN day_of_week = 0 THEN 'Monday'
                WHEN day_of_week = 2 THEN 'Wednesday'
                WHEN day_of_week = 4 THEN 'Friday'
                ELSE 'Other'
            END as update_day,
            COUNT(DISTINCT date) as number_of_dates,
            SUM(record_count) as total_records,
            SUM(non_null_month_ago) as records_with_month_ago,
            SUM(null_month_ago) as records_missing_month_ago
        FROM update_days
        GROUP BY 
            CASE 
                WHEN day_of_week = 0 THEN 'Monday'
                WHEN day_of_week = 2 THEN 'Wednesday'
                WHEN day_of_week = 4 THEN 'Friday'
                ELSE 'Other'
            END
        ORDER BY update_day
        """
        
        verify_result = subprocess.run(
            ["dolt", "sql", "-q", verify_query, "--result-format", "csv"],
            cwd="D:\\databases\\options",
            capture_output=True,
            text=True
        )
        
        if verify_result.returncode == 0:
            verify_df = pd.read_csv(StringIO(verify_result.stdout))
            logger.info("\nFinal Update Verification by Day:")
            for _, row in verify_df.iterrows():
                logger.info(f"\n{row['update_day']}:")
                logger.info(f"Number of dates: {row['number_of_dates']}")
                logger.info(f"Total records: {row['total_records']}")
                logger.info(f"Records with iv_month_ago: {row['records_with_month_ago']}")
                logger.info(f"Records missing iv_month_ago: {row['records_missing_month_ago']}")
        else:
            logger.error(f"Failed to verify update: {verify_result.stderr}")
            
    except Exception as e:
        logger.error(f"Exception occurred while updating iv_month_ago values: {str(e)}")
        logger.error("You may need to run the script again to complete the updates")

def get_latest_commit_date(dolt_db_path):
    result = subprocess.run(
        ["dolt", "log", "-n", "1", "--date=short"],
        cwd=dolt_db_path,
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            if line.startswith("Date:"):
                return line.split("Date:")[1].strip()
    return None

latest_date = get_latest_commit_date("D:/databases/options")
print("Latest commit date:", latest_date)

def get_commit_history(dolt_db_path, n=100):
    result = subprocess.run(
        ["dolt", "log", f"-n={n}", "--date=short", "--format=csv"],
        cwd=dolt_db_path,
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        df = pd.read_csv(StringIO(result.stdout))
        return df
    else:
        logger.error("Error fetching commit history: %s", result.stderr)
        return None

def get_closest_commit(commit_df, days_ago=28):
    latest_commit_date = pd.to_datetime(commit_df['Date'].iloc[0])
    target_date = latest_commit_date - pd.Timedelta(days=days_ago)
    commit_df['Date'] = pd.to_datetime(commit_df['Date'])
    commit_df['days_diff'] = (commit_df['Date'] - target_date).abs()
    closest_commit = commit_df.loc[commit_df['days_diff'].idxmin()]
    logger.info(f"Closest commit to {days_ago} days ago: {closest_commit['Date']} (hash: {closest_commit['commit_hash']})")
    return closest_commit

def get_iv_current_at_commit(dolt_db_path, commit_hash, ticker, date):
    query = f"""
    SELECT iv_current FROM volatility_history
    WHERE act_symbol = '{ticker}' AND date = '{date}'
    """
    result = subprocess.run(
        ["dolt", "sql", "-r", "csv", "-q", query, "--commit", commit_hash],
        cwd=dolt_db_path,
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        df = pd.read_csv(StringIO(result.stdout))
        return df
    else:
        logger.error("Error fetching iv_current: %s", result.stderr)
        return None

def main_commit_scanner(dolt_db_path, ticker, days_ago=28):
    logger.info("Pulling commit history...")
    commit_df = get_commit_history(dolt_db_path)
    if commit_df is None or commit_df.empty:
        logger.error("No commit history found.")
        return
    logger.info(f"Most recent commit date: {commit_df['Date'].iloc[0]}")
    closest_commit = get_closest_commit(commit_df, days_ago=days_ago)
    commit_hash = closest_commit['commit_hash']
    commit_date = closest_commit['Date'].strftime('%Y-%m-%d')
    logger.info(f"Fetching iv_current for ticker {ticker} on {commit_date} at commit {commit_hash}")
    iv_df = get_iv_current_at_commit(dolt_db_path, commit_hash, ticker, commit_date)
    if iv_df is not None and not iv_df.empty:
        logger.info(f"iv_current for {ticker} on {commit_date}: {iv_df['iv_current'].iloc[0]}")
    else:
        logger.warning(f"No iv_current found for {ticker} on {commit_date} at commit {commit_hash}")

if __name__ == "__main__":
    # First update the month ago values
    update_month_ago_values()
    
    # Then get and process the data
    result = get_volatility_data()
    if result is None:
        logger.error("Failed to get volatility data")
        exit(1)
        
    vol_df, r1000, tickers = result
    
    # Debug: Check for any NaN values in iv_current or iv_month_ago
    logger.debug(f"NaN values in iv_current: {vol_df['iv_current'].isna().sum()}")
    logger.debug(f"NaN values in iv_month_ago: {vol_df['iv_month_ago'].isna().sum()}")

    # Calculate TSS ratio and delta
    vol_df['tss_ratio'] = vol_df['iv_current'] / vol_df['iv_month_ago']
    vol_df['tss_delta'] = (vol_df['iv_month_ago'] - vol_df['iv_current']) / vol_df['iv_current']

    # Debug: Check for any NaN values in calculated columns
    logger.debug(f"NaN values in tss_ratio: {vol_df['tss_ratio'].isna().sum()}")
    logger.debug(f"NaN values in tss_delta: {vol_df['tss_delta'].isna().sum()}")

    logger.info("Merging with Russell 1000 data...")
    merged = pd.merge(r1000, vol_df, left_on='Ticker', right_on='act_symbol', how='inner')
    logger.info(f"After merge, we have {len(merged)} rows")

    # Drop act_symbol (it's the same as Ticker)
    merged = merged.drop(columns=['act_symbol'])

    # Set index to Ticker
    merged = merged.set_index('Ticker')

    # Select and order columns
    final_cols = [
        'CompanyName', 'Q1_Date', 'Q2_Date', 'Q3_Date', 'Q4_Date',
        'date', 'iv_current', 'iv_month_ago', 'tss_ratio', 'tss_delta'
    ]
    final_df = merged[final_cols]

    # Print summary statistics
    logger.info("\nSummary Statistics:")
    logger.info(f"Total tickers in Russell 1000: {len(tickers)}")
    logger.info(f"Tickers with volatility data: {len(vol_df)}")
    logger.info(f"Tickers in final merged dataset: {len(final_df)}")

    # Find missing tickers
    missing = set(tickers) - set(vol_df['act_symbol'])
    logger.info(f"\nMissing tickers count: {len(missing)}")
    logger.info("First 10 missing tickers: " + ", ".join(list(missing)[:10]))

    # Print sample of final data
    logger.info("\nSample of final data:")
    print(final_df.head())

    # Print rows with NaN values in tss_ratio or tss_delta
    nan_rows = final_df[final_df['tss_ratio'].isna() | final_df['tss_delta'].isna()]
    if not nan_rows.empty:
        logger.warning(f"\nFound {len(nan_rows)} rows with NaN values in tss_ratio or tss_delta")
        logger.warning("Sample of rows with NaN values:")
        print(nan_rows.head())

    # Example usage: scan for ticker 'AAPL' 28 days ago
    main_commit_scanner("D:/databases/options", ticker="AAPL", days_ago=28)
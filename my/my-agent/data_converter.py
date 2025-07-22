"""
Data Converter Utility for Trading Strategy

This utility helps convert various data formats to the format expected
by the trading strategy.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def convert_csv_data(input_file, output_file=None, date_col='Date', 
                    minute_col='Minute', open_col='Open', high_col='High', 
                    low_col='Low', close_col='Close', volume_col='Volume'):
    """
    Convert CSV data to the format expected by the trading strategy
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file (optional)
        date_col (str): Name of date column
        minute_col (str): Name of minute column  
        open_col (str): Name of open price column
        high_col (str): Name of high price column
        low_col (str): Name of low price column
        close_col (str): Name of close price column
        volume_col (str): Name of volume column
    
    Returns:
        pd.DataFrame: Converted data
    """
    
    print(f"Loading data from {input_file}...")
    
    try:
        # Load the data
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
        
        # Create mapping for column names
        column_mapping = {}
        
        # Try to auto-detect columns if not specified
        for col in df.columns:
            col_lower = col.lower()
            if 'date' in col_lower and date_col == 'Date':
                column_mapping[col] = 'Date'
            elif 'minute' in col_lower and minute_col == 'Minute':
                column_mapping[col] = 'Minute'
            elif 'open' in col_lower and open_col == 'Open':
                column_mapping[col] = 'Open'
            elif 'high' in col_lower and high_col == 'High':
                column_mapping[col] = 'High'
            elif 'low' in col_lower and low_col == 'Low':
                column_mapping[col] = 'Low'
            elif 'close' in col_lower and close_col == 'Close':
                column_mapping[col] = 'Close'
            elif 'volume' in col_lower and volume_col == 'Volume':
                column_mapping[col] = 'Volume'
        
        print(f"Auto-detected column mapping: {column_mapping}")
        
        # Apply column mapping
        if column_mapping:
            df = df.rename(columns=column_mapping)
        
        # Ensure we have required columns
        required_cols = ['Date', 'Minute', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"Warning: Missing columns: {missing_cols}")
            
            # Try to create missing columns
            if 'Minute' not in df.columns:
                print("Creating minute column from timestamp...")
                # Assume data is sorted by time and create minute numbers
                df['Minute'] = range(390, 390 + len(df))
            
            if 'Volume' not in df.columns:
                print("Creating dummy volume column...")
                df['Volume'] = np.random.lognormal(11, 0.5, len(df))
        
        # Validate data types
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Minute']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Validate date format
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        
        # Sort by date and minute
        if 'Date' in df.columns and 'Minute' in df.columns:
            df = df.sort_values(['Date', 'Minute'])
        
        # Remove any rows with NaN values
        before_dropna = len(df)
        df = df.dropna()
        after_dropna = len(df)
        
        if before_dropna != after_dropna:
            print(f"Removed {before_dropna - after_dropna} rows with missing data")
        
        print(f"Final data shape: {df.shape}")
        print("Sample of converted data:")
        print(df.head())
        
        # Save to output file if specified
        if output_file:
            df.to_csv(output_file, index=False)
            print(f"Converted data saved to {output_file}")
        
        return df
        
    except Exception as e:
        print(f"Error converting data: {e}")
        return None

def create_sample_data_file(filename='sample_data.csv', days=5):
    """
    Create a sample data file for testing
    
    Args:
        filename (str): Output filename
        days (int): Number of trading days to generate
    """
    
    print(f"Creating sample data file: {filename}")
    
    np.random.seed(42)
    
    # Generate dates for multiple trading days
    start_date = pd.Timestamp('2025-01-02')
    end_date = start_date + pd.Timedelta(days=days)
    trading_days = pd.bdate_range(start_date, end_date)
    
    # Minutes from market open
    minutes_per_day = range(390, 780)  # 6.5 hours trading day
    
    data_list = []
    base_price = 589.0
    current_price = base_price
    
    for date in trading_days:
        # Reset price with overnight gap
        overnight_gap = np.random.normal(0, 2)
        current_price += overnight_gap
        
        for minute in minutes_per_day:
            # Generate price movement
            price_change = np.random.normal(0, 0.5)
            
            open_price = current_price
            close_price = current_price + price_change
            
            # Generate high and low
            volatility = abs(np.random.normal(0, 0.3))
            high_price = max(open_price, close_price) + volatility
            low_price = min(open_price, close_price) - volatility
            
            # Generate volume
            volume = np.random.lognormal(11, 0.5)
            
            data_list.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Minute': minute,
                'Open': round(open_price, 4),
                'Low': round(low_price, 4),
                'High': round(high_price, 4),
                'Close': round(close_price, 4),
                'Volume': round(volume, 2)
            })
            
            current_price = close_price
    
    # Create DataFrame and save
    df = pd.DataFrame(data_list)
    df.to_csv(filename, index=False)
    
    print(f"Created sample data with {len(df)} rows")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print("Sample data:")
    print(df.head())
    
    return df

def main():
    """
    Main function for the data converter utility
    """
    print("=== Data Converter Utility ===")
    print("1. Convert existing CSV data")
    print("2. Create sample data file")
    print("3. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == '1':
                input_file = input("Enter path to input CSV file: ").strip()
                output_file = input("Enter path for output file (or press Enter to skip): ").strip()
                
                if not output_file:
                    output_file = None
                
                result = convert_csv_data(input_file, output_file)
                
                if result is not None:
                    print("Conversion completed successfully!")
                else:
                    print("Conversion failed!")
                    
            elif choice == '2':
                filename = input("Enter filename for sample data (default: sample_data.csv): ").strip()
                if not filename:
                    filename = 'sample_data.csv'
                
                days_str = input("Enter number of trading days (default: 5): ").strip()
                days = 5 if not days_str else int(days_str)
                
                create_sample_data_file(filename, days)
                print("Sample data created successfully!")
                
            elif choice == '3':
                print("Goodbye!")
                break
                
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()

# High/Low Breakout Trading Strategy

A comprehensive trading strategy implementation using VectorBT that follows these rules:
- **Buy Signal**: When the current price breaks above the highest high of the last 5 minutes
- **Sell Signal**: When the current price breaks below the lowest low of the last 5 minutes  
- **Position Management**: Close all positions at the end of each trading day

## Files Included

- `trading_strategy.ipynb` - Interactive Jupyter notebook with detailed analysis
- `trading_strategy.py` - Complete Python script version
- `requirements.txt` - Python package dependencies
- `setup.sh` - Setup script for macOS/Linux
- `README.md` - This documentation file

## Quick Start

### Option 1: Using the Setup Script (Recommended)

```bash
# Make the setup script executable
chmod +x setup.sh

# Run the setup script
./setup.sh
```

### Option 2: Manual Installation

```bash
# Install required packages
pip3 install pandas numpy matplotlib jupyter ipykernel vectorbt

# Run the Python script
python3 trading_strategy.py

# Or start Jupyter notebook
jupyter notebook trading_strategy.ipynb
```

## Data Format

Your CSV data should follow this format:

```
Date,Minute,Open,Low,High,Close,Volume
2025-01-02,390,589.3900,588.4600,589.4500,589.4500,1128698.00
2025-01-02,391,589.4500,589.0400,589.6000,589.5700,151034.00
...
```

Where:
- **Date**: Trading date (YYYY-MM-DD format)
- **Minute**: Minute number from market open (390 = 9:30 AM)
- **Open/High/Low/Close**: OHLC prices
- **Volume**: Trading volume

## Using Your Own Data

### In the Python Script

Edit `trading_strategy.py` and modify the main function:

```python
# Replace this line:
strategy.generate_sample_data(days=5)

# With this:
strategy.load_data_from_csv('path/to/your/data.csv')
```

### In the Jupyter Notebook

Replace the sample data generation cell with:

```python
# Load your actual data
df = pd.read_csv('path/to/your/data.csv')

# Process the data
df['DateTime'] = pd.to_datetime(df['Date']) + pd.to_timedelta((df['Minute'] - 390), unit='m')
df.set_index('DateTime', inplace=True)
df.sort_index(inplace=True)
```

## Strategy Parameters

You can customize the strategy by adjusting these parameters:

- **Lookback Periods**: Number of minutes to look back for high/low calculation (default: 5)
- **Initial Cash**: Starting capital (default: $100,000)
- **Transaction Fees**: Fee percentage per trade (default: 0.1%)

### Example Customization

```python
strategy = BreakoutTradingStrategy(
    lookback_periods=7,    # Look back 7 minutes instead of 5
    initial_cash=50000,    # Start with $50,000
    fees=0.002            # 0.2% transaction fees
)
```

## Strategy Features

### Core Logic
- Calculates rolling high/low over specified lookback period
- Generates buy signals on upward breakouts
- Generates sell signals on downward breakouts
- Automatically closes positions at end of trading day

### Analysis Tools
- Performance statistics (returns, Sharpe ratio, drawdown)
- Trade analysis (win rate, profit factor, trade distribution)
- Visualization (price charts, portfolio value, signals)
- Parameter optimization
- Risk metrics and drawdown analysis

### Optimization
The strategy includes automatic parameter optimization that tests different lookback periods to find the best performing configuration.

## Output and Analysis

The strategy provides comprehensive analysis including:

### Performance Metrics
- Total return percentage
- Sharpe ratio
- Maximum drawdown
- Win rate
- Total number of trades
- Profit factor

### Visualizations
- Price chart with buy/sell signals
- Portfolio value over time
- Cumulative returns
- Drawdown periods

### Trade Analysis
- Individual trade results
- Trade duration statistics
- Winning vs losing trade analysis
- Best and worst performing trades

## Example Results

When run with the sample data, you might see results like:

```
=== Performance Statistics ===
Total Return (%): 2.45%
Sharpe Ratio: 0.34
Maximum Drawdown (%): -1.23%
Total Trades: 45
Win Rate (%): 52.22%
Final Portfolio Value: $102,450.00
```

## Customization and Extensions

### Additional Filters
You can enhance the strategy by adding:
- Volume filters (only trade on high volume)
- Volatility filters (avoid trading in low volatility periods)
- Time-of-day filters (only trade during certain hours)

### Risk Management
Consider adding:
- Position sizing based on volatility
- Stop-loss orders
- Take-profit targets
- Maximum position limits

### Example Enhancement

```python
# Add volume filter
volume_filter = df['Volume'] > df['Volume'].rolling(20).mean()
df['Buy_Signal'] = df['Buy_Signal'] & volume_filter

# Add stop-loss
# Implement stop-loss at 2% below entry price
stop_loss_pct = 0.02
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all packages are installed
   ```bash
   pip3 install --upgrade pandas numpy matplotlib vectorbt
   ```

2. **Data Format Issues**: Ensure your CSV has the correct column names and data types

3. **No Trading Signals**: This might happen with certain market conditions. The strategy includes fallback logic for such cases.

4. **Memory Issues**: For large datasets, consider processing data in chunks or using a more powerful machine

### Performance Tips

- Use SSD storage for faster data loading
- Ensure you have at least 4GB RAM available
- Close other applications when running large backtests

## Next Steps

1. **Load Your Data**: Replace sample data with your actual historical data
2. **Optimize Parameters**: Use the built-in optimization to find best lookback period
3. **Add Filters**: Implement additional filters based on volume, volatility, etc.
4. **Risk Management**: Add stop-loss and position sizing rules
5. **Paper Trading**: Test the strategy with paper trading before live implementation

## Disclaimer

This is for educational and research purposes only. Past performance does not guarantee future results. Always test strategies thoroughly before using real money, and consider the risks involved in trading.

## Support

For questions or issues:
1. Check the troubleshooting section above
2. Review the code comments in both files
3. Consult the VectorBT documentation for advanced features

## License

This code is provided as-is for educational purposes. Use at your own risk.

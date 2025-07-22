#!/usr/bin/env python3
"""
High/Low Breakout Trading Strategy Implementation

This script implements a trading strategy based on price breakouts:
- Buy Signal: Price exceeds the highest high of the last 5 minutes
- Sell Signal: Price falls below the lowest low of the last 5 minutes  
- Position Management: Close all positions at the end of each trading day

Requirements:
- pandas
- numpy
- vectorbt
- matplotlib
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
import matplotlib.pyplot as plt
from datetime import datetime, time
import warnings
from io import StringIO

warnings.filterwarnings('ignore')

class BreakoutTradingStrategy:
    """
    High/Low Breakout Trading Strategy using VectorBT
    """
    
    def __init__(self, lookback_periods=5, initial_cash=100000, fees=0.001):
        """
        Initialize the trading strategy
        
        Args:
            lookback_periods (int): Number of periods to look back for high/low calculation
            initial_cash (float): Initial capital for the portfolio
            fees (float): Transaction fees as a percentage
        """
        self.lookback_periods = lookback_periods
        self.initial_cash = initial_cash
        self.fees = fees
        self.data = None
        self.portfolio = None
        
    def load_data_from_csv(self, file_path):
        """
        Load data from CSV file
        
        Args:
            file_path (str): Path to the CSV file
        """
        try:
            self.data = pd.read_csv(file_path)
            self._prepare_data()
            print(f"Data loaded successfully from {file_path}")
            print(f"Data shape: {self.data.shape}")
        except FileNotFoundError:
            print(f"File {file_path} not found. Using sample data instead.")
            self.generate_sample_data()
            
    def generate_sample_data(self, days=5):
        """
        Generate synthetic OHLC data for testing
        
        Args:
            days (int): Number of trading days to generate
        """
        print("Generating synthetic data for testing...")
        
        np.random.seed(42)  # For reproducible results
        
        # Generate dates for multiple trading days
        start_date = pd.Timestamp('2025-01-02')
        end_date = start_date + pd.Timedelta(days=days)
        trading_days = pd.bdate_range(start_date, end_date)
        
        # Minutes from market open (assuming 9:30 AM start, minute 390 = 9:30 AM)
        minutes_per_day = range(390, 780)  # 390 minutes = 6.5 hours trading day
        
        # Generate synthetic OHLC data
        data_list = []
        base_price = 589.0
        current_price = base_price
        
        for date in trading_days:
            # Reset price with some overnight gap
            overnight_gap = np.random.normal(0, 2)
            current_price += overnight_gap
            
            for minute in minutes_per_day:
                # Generate minute-level price movement
                price_change = np.random.normal(0, 0.5)
                
                open_price = current_price
                close_price = current_price + price_change
                
                # Generate high and low based on volatility
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
        
        self.data = pd.DataFrame(data_list)
        self._prepare_data()
        print(f"Generated {len(self.data)} data points over {days} trading days")
        
    def _prepare_data(self):
        """
        Prepare data with proper datetime index and sorting
        """
        # Create proper datetime index
        self.data['DateTime'] = (pd.to_datetime(self.data['Date']) + 
                                pd.to_timedelta((self.data['Minute'] - 390), unit='m'))
        self.data.set_index('DateTime', inplace=True)
        self.data.sort_index(inplace=True)
        
    def calculate_signals(self):
        """
        Calculate trading signals based on rolling high/low breakouts
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        # Calculate rolling high and low
        self.data['Rolling_High'] = (self.data['High']
                                   .rolling(window=self.lookback_periods, min_periods=self.lookback_periods)
                                   .max())
        
        self.data['Rolling_Low'] = (self.data['Low']
                                  .rolling(window=self.lookback_periods, min_periods=self.lookback_periods)
                                  .min())
        
        # Shift by 1 to avoid look-ahead bias
        self.data['Prev_Rolling_High'] = self.data['Rolling_High'].shift(1)
        self.data['Prev_Rolling_Low'] = self.data['Rolling_Low'].shift(1)
        
        # Generate trading signals
        self.data['Buy_Signal'] = self.data['High'] > self.data['Prev_Rolling_High']
        self.data['Sell_Signal'] = self.data['Low'] < self.data['Prev_Rolling_Low']
        
        # End of day signal (assuming market closes at minute 779)
        self.data['End_Of_Day'] = self.data['Minute'] == 779
        
        print(f"Signals calculated:")
        print(f"  Buy signals: {self.data['Buy_Signal'].sum()}")
        print(f"  Sell signals: {self.data['Sell_Signal'].sum()}")
        print(f"  End of day signals: {self.data['End_Of_Day'].sum()}")
        
    def run_backtest(self):
        """
        Run the backtest using VectorBT
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        # Ensure signals are calculated
        if 'Buy_Signal' not in self.data.columns:
            self.calculate_signals()
            
        # Prepare signals for VectorBT
        entries = self.data['Buy_Signal']
        exits = self.data['Sell_Signal'] | self.data['End_Of_Day']
        
        # Handle case where no signals are generated
        if entries.sum() == 0:
            print("Warning: No buy signals generated. Creating fallback signals...")
            # Fallback: simple moving average crossover
            self.data['SMA_Short'] = self.data['Close'].rolling(3).mean()
            self.data['SMA_Long'] = self.data['Close'].rolling(10).mean()
            entries = self.data['SMA_Short'] > self.data['SMA_Long']
            exits = self.data['SMA_Short'] < self.data['SMA_Long']
            print(f"Fallback signals - Entries: {entries.sum()}, Exits: {exits.sum()}")
        
        # Create portfolio
        self.portfolio = vbt.Portfolio.from_signals(
            close=self.data['Close'],
            entries=entries,
            exits=exits,
            init_cash=self.initial_cash,
            fees=self.fees,
            freq='1min'
        )
        
        print("Backtest completed successfully!")
        
    def get_performance_stats(self):
        """
        Get performance statistics
        """
        if self.portfolio is None:
            raise ValueError("No portfolio found. Please run backtest first.")
            
        stats = self.portfolio.stats()
        trades = self.portfolio.trades
        
        performance = {
            'Total Return (%)': self.portfolio.total_return() * 100,
            'Sharpe Ratio': self.portfolio.sharpe_ratio(),
            'Maximum Drawdown (%)': self.portfolio.max_drawdown() * 100,
            'Total Trades': trades.count(),
            'Win Rate (%)': trades.win_rate * 100 if trades.count() > 0 else 0,
            'Final Portfolio Value': self.portfolio.value().iloc[-1],
            'Profit Factor': trades.profit_factor if trades.count() > 0 else 0
        }
        
        return performance
        
    def plot_results(self, save_plots=False):
        """
        Plot the results
        
        Args:
            save_plots (bool): Whether to save plots to files
        """
        if self.portfolio is None:
            raise ValueError("No portfolio found. Please run backtest first.")
            
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Plot 1: Price chart with signals
        axes[0].plot(self.data.index, self.data['Close'], 
                    label='Close Price', linewidth=1, alpha=0.8)
        axes[0].plot(self.data.index, self.data['Prev_Rolling_High'], 
                    label=f'{self.lookback_periods}-Min Rolling High', 
                    linestyle='--', alpha=0.7)
        axes[0].plot(self.data.index, self.data['Prev_Rolling_Low'], 
                    label=f'{self.lookback_periods}-Min Rolling Low', 
                    linestyle='--', alpha=0.7)
        
        # Mark signals
        buy_points = self.data[self.data['Buy_Signal']]
        sell_points = self.data[self.data['Sell_Signal']]
        
        if len(buy_points) > 0:
            axes[0].scatter(buy_points.index, buy_points['Close'], 
                          color='green', marker='^', s=50, 
                          label='Buy Signal', alpha=0.8)
        if len(sell_points) > 0:
            axes[0].scatter(sell_points.index, sell_points['Close'], 
                          color='red', marker='v', s=50, 
                          label='Sell Signal', alpha=0.8)
        
        axes[0].set_title('Price Chart with Trading Signals')
        axes[0].set_ylabel('Price ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Portfolio value
        portfolio_value = self.portfolio.value()
        axes[1].plot(portfolio_value.index, portfolio_value.values, 
                    label='Portfolio Value', color='blue', linewidth=2)
        axes[1].axhline(y=self.initial_cash, color='gray', 
                       linestyle='--', alpha=0.7, label='Initial Capital')
        axes[1].set_title('Portfolio Value Over Time')
        axes[1].set_ylabel('Portfolio Value ($)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Cumulative returns
        cumulative_returns = (portfolio_value / self.initial_cash - 1) * 100
        axes[2].plot(portfolio_value.index, cumulative_returns, 
                    label='Cumulative Return (%)', color='purple', linewidth=2)
        axes[2].axhline(y=0, color='gray', linestyle='-', alpha=0.7)
        axes[2].set_title('Cumulative Returns')
        axes[2].set_ylabel('Return (%)')
        axes[2].set_xlabel('Date')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('trading_strategy_results.png', dpi=300, bbox_inches='tight')
            print("Plot saved as 'trading_strategy_results.png'")
        
        plt.show()
        
    def optimize_parameters(self, lookback_range=None):
        """
        Optimize strategy parameters
        
        Args:
            lookback_range (list): Range of lookback periods to test
        """
        if lookback_range is None:
            lookback_range = [3, 5, 7, 10, 15, 20]
            
        results = []
        original_lookback = self.lookback_periods
        
        for lookback in lookback_range:
            print(f"Testing lookback period: {lookback}")
            
            # Temporarily change lookback period
            self.lookback_periods = lookback
            
            try:
                # Recalculate signals
                self.calculate_signals()
                
                # Run backtest
                entries = self.data['Buy_Signal']
                exits = self.data['Sell_Signal'] | self.data['End_Of_Day']
                
                if entries.sum() == 0:
                    continue
                    
                pf = vbt.Portfolio.from_signals(
                    close=self.data['Close'],
                    entries=entries,
                    exits=exits,
                    init_cash=self.initial_cash,
                    fees=self.fees,
                    freq='1min'
                )
                
                results.append({
                    'Lookback': lookback,
                    'Total_Return': pf.total_return() * 100,
                    'Sharpe_Ratio': pf.sharpe_ratio(),
                    'Max_Drawdown': pf.max_drawdown() * 100,
                    'Total_Trades': pf.trades.count(),
                    'Win_Rate': pf.trades.win_rate * 100
                })
                
            except Exception as e:
                print(f"Error with lookback {lookback}: {e}")
                continue
        
        # Restore original lookback
        self.lookback_periods = original_lookback
        
        if results:
            optimization_results = pd.DataFrame(results)
            print("\n=== Optimization Results ===")
            print(optimization_results.round(2))
            
            # Find best parameters
            best_sharpe_idx = optimization_results['Sharpe_Ratio'].idxmax()
            best_return_idx = optimization_results['Total_Return'].idxmax()
            
            print(f"\nBest Sharpe Ratio: {optimization_results.loc[best_sharpe_idx, 'Lookback']} periods")
            print(f"Best Total Return: {optimization_results.loc[best_return_idx, 'Lookback']} periods")
            
            return optimization_results
        else:
            print("No valid optimization results found.")
            return None


def main():
    """
    Main function to run the trading strategy
    """
    print("=== High/Low Breakout Trading Strategy ===")
    print("Loading data and running backtest...\n")
    
    # Initialize strategy
    strategy = BreakoutTradingStrategy(
        lookback_periods=5,
        initial_cash=100000,
        fees=0.001
    )
    
    # Try to load actual data, fall back to sample data
    try:
        # Uncomment the line below and provide your data file path
        # strategy.load_data_from_csv('your_data_file.csv')
        strategy.generate_sample_data(days=5)
    except Exception as e:
        print(f"Error loading data: {e}")
        strategy.generate_sample_data(days=5)
    
    # Calculate signals and run backtest
    strategy.calculate_signals()
    strategy.run_backtest()
    
    # Get and display performance statistics
    performance = strategy.get_performance_stats()
    print("\n=== Performance Statistics ===")
    for key, value in performance.items():
        if isinstance(value, float):
            if 'Return' in key or 'Drawdown' in key or 'Rate' in key:
                print(f"{key}: {value:.2f}%")
            elif 'Value' in key:
                print(f"{key}: ${value:,.2f}")
            else:
                print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    # Plot results
    strategy.plot_results(save_plots=True)
    
    # Run optimization
    print("\n=== Running Parameter Optimization ===")
    optimization_results = strategy.optimize_parameters()
    
    print("\n=== Strategy Summary ===")
    print("This strategy implements a simple breakout system:")
    print("- Buys when price breaks above recent highs")
    print("- Sells when price breaks below recent lows")
    print("- Closes positions at end of day")
    print("\nTo use with your data:")
    print("1. Replace the sample data with your CSV file")
    print("2. Adjust the lookback period based on optimization results")
    print("3. Consider adding filters and risk management rules")


if __name__ == "__main__":
    main()

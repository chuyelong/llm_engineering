#!/bin/bash

# Setup script for the High/Low Breakout Trading Strategy

echo "=== Setting up High/Low Breakout Trading Strategy ==="
echo "This script will install the required Python packages"
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

echo "Python version:"
python3 --version
echo

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 is not installed or not in PATH"
    exit 1
fi

echo "Installing required packages..."
echo "This may take a few minutes..."
echo

# Install packages
pip3 install pandas numpy matplotlib jupyter ipykernel

# Install vectorbt (this might take longer)
echo "Installing vectorbt (this may take several minutes)..."
pip3 install vectorbt

echo
echo "=== Installation Complete ==="
echo
echo "You can now run the trading strategy:"
echo "1. For Jupyter notebook: jupyter notebook trading_strategy.ipynb"
echo "2. For Python script: python3 trading_strategy.py"
echo
echo "Make sure to replace the sample data with your actual stock data!"

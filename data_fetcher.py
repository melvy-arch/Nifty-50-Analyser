"""
Data Fetching Module for Nifty 50
Handles downloading historical price data from yfinance
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NiftyDataFetcher:
    """Fetch and preprocess Nifty 50 data"""
    
    def __init__(self, ticker="^NSEI", lookback_years=5):
        """
        Initialize data fetcher
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for Nifty 50 (default: ^NSEI)
        lookback_years : int
            Years of historical data to fetch
        """
        self.ticker = ticker
        self.lookback_years = lookback_years
        self.data = None
        self.returns = None
        
    def fetch_data(self):
        """
        Fetch historical OHLCV data from yfinance
        
        Returns:
        --------
        pd.DataFrame : Historical price data
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_years * 365)
            
            logger.info(f"Fetching {self.ticker} data from {start_date.date()} to {end_date.date()}")
            
            self.data = yf.download(
                self.ticker,
                start=start_date,
                end=end_date,
                progress=False
            )
            
            logger.info(f"Downloaded {len(self.data)} records")
            return self.data
            
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            raise
    
    def calculate_returns(self, method='log'):
        """
        Calculate returns from price data
        
        Parameters:
        -----------
        method : str
            'log' for log returns, 'simple' for simple returns
            
        Returns:
        --------
        pd.Series : Daily returns
        """
        if self.data is None:
            self.fetch_data()
        
        # Flatten columns if they are MultiIndex
        if isinstance(self.data.columns, pd.MultiIndex):
            self.data.columns = ['_'.join(col).strip() for col in self.data.columns]
        
        # Try different column names that yfinance might use
        adj_close = None
        possible_columns = ['Adj Close', 'Adj_Close', 'Close', 'close', 'Adjusted Close']
        
        for col in possible_columns:
            if col in self.data.columns:
                adj_close = self.data[col]
                logger.info(f"Using column: {col}")
                break
        
        # If still not found, use the last column (usually Close price)
        if adj_close is None:
            adj_close = self.data.iloc[:, -1]
            logger.warning(f"No standard column found. Using: {self.data.columns[-1]}")
        
        if method == 'log':
            self.returns = np.log(adj_close / adj_close.shift(1)).dropna()
        else:
            self.returns = (adj_close.pct_change()).dropna()
        
        logger.info(f"Calculated {len(self.returns)} daily returns")
        return self.returns
    
    def get_preprocessed_data(self):
        """
        Get returns scaled by 100 for better numerical stability
        
        Returns:
        --------
        np.ndarray : Daily returns scaled by 100
        """
        if self.returns is None:
            self.calculate_returns()
        
        # Scale by 100 for better numerical stability
        return (self.returns * 100).values
    
    def get_summary_stats(self):
        """
        Calculate summary statistics of returns
        
        Returns:
        --------
        dict : Summary statistics
        """
        if self.returns is None:
            self.calculate_returns()
        
        returns_pct = self.returns * 100
        
        stats = {
            'mean': returns_pct.mean(),
            'std': returns_pct.std(),
            'min': returns_pct.min(),
            'max': returns_pct.max(),
            'skewness': returns_pct.skew(),
            'kurtosis': returns_pct.kurtosis(),
            'observations': len(self.returns)
        }
        
        return stats

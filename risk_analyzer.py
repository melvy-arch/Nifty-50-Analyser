"""
Risk Analysis Module
Calculates Value at Risk (VaR) and other risk metrics
"""

import numpy as np
import pandas as pd
from scipy import stats
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class RiskAnalyzer:
    """Calculate and analyze risk metrics"""
    
    def __init__(self, conditional_volatility: np.ndarray, returns: np.ndarray):
        """
        Initialize risk analyzer
        
        Parameters:
        -----------
        conditional_volatility : np.ndarray
            Conditional volatility from GARCH model (in percentage)
        returns : np.ndarray
            Daily returns (in percentage)
        """
        self.conditional_volatility = conditional_volatility
        self.returns = returns
        self.mean_return = np.mean(returns)
        self.latest_vol = conditional_volatility[-1]
        
    def calculate_var(
        self,
        confidence_level: float = 0.95,
        portfolio_value: float = 100000,
        method: str = 'historical'
    ) -> Dict:
        """
        Calculate Value at Risk (VaR)
        
        Parameters:
        -----------
        confidence_level : float
            Confidence level (e.g., 0.95 for 95%)
        portfolio_value : float
            Portfolio value in base currency
        method : str
            'historical' or 'parametric'
            
        Returns:
        --------
        dict : VaR metrics
        """
        if method == 'historical':
            var_value = np.percentile(self.returns, (1 - confidence_level) * 100)
        else:  # parametric
            var_value = self.mean_return - 1.96 * self.latest_vol  # 95% CI
        
        var_amount = abs(var_value / 100 * portfolio_value)
        
        return {
            'var_percentage': abs(var_value),
            'var_amount': var_amount,
            'confidence_level': confidence_level * 100,
            'method': method
        }
    
    def calculate_expected_shortfall(
        self,
        confidence_level: float = 0.95,
        portfolio_value: float = 100000
    ) -> Dict:
        """
        Calculate Expected Shortfall (CVaR/ES)
        Measures average loss beyond VaR
        
        Parameters:
        -----------
        confidence_level : float
            Confidence level (e.g., 0.95 for 95%)
        portfolio_value : float
            Portfolio value in base currency
            
        Returns:
        --------
        dict : Expected Shortfall metrics
        """
        var_percentile = (1 - confidence_level) * 100
        var_value = np.percentile(self.returns, var_percentile)
        
        # Average of returns worse than VaR
        es_value = np.mean(self.returns[self.returns <= var_value])
        es_amount = abs(es_value / 100 * portfolio_value)
        
        return {
            'es_percentage': abs(es_value),
            'es_amount': es_amount,
            'confidence_level': confidence_level * 100,
            'var_percentage': abs(var_value)
        }
    
    def calculate_portfolio_metrics(
        self,
        portfolio_value: float = 100000,
        holding_period: int = 1
    ) -> Dict:
        """
        Calculate portfolio risk metrics
        
        Parameters:
        -----------
        portfolio_value : float
            Current portfolio value
        holding_period : int
            Holding period in days
            
        Returns:
        --------
        dict : Portfolio metrics
        """
        # Annualized metrics (assuming 252 trading days)
        annualized_vol = self.latest_vol * np.sqrt(252)
        annualized_return = self.mean_return * 252
        
        # Value at Risk
        var_95 = self.calculate_var(0.95, portfolio_value, 'historical')
        var_99 = self.calculate_var(0.99, portfolio_value, 'historical')
        
        # Expected Shortfall
        es_95 = self.calculate_expected_shortfall(0.95, portfolio_value)
        es_99 = self.calculate_expected_shortfall(0.99, portfolio_value)
        
        # Sharpe Ratio (assuming 0% risk-free rate for simplicity)
        sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
        
        # Maximum Drawdown
        cumulative_returns = np.cumsum(self.returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / np.abs(running_max)
        max_drawdown = np.min(drawdown)
        
        return {
            'portfolio_value': portfolio_value,
            'latest_volatility_pct': self.latest_vol,
            'annualized_volatility_pct': annualized_vol,
            'annualized_return_pct': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'maximum_drawdown_pct': max_drawdown * 100,
            'var_95': var_95,
            'var_99': var_99,
            'expected_shortfall_95': es_95,
            'expected_shortfall_99': es_99
        }
    
    def calculate_risk_decomposition(
        self,
        portfolio_value: float = 100000
    ) -> Dict:
        """
        Break down portfolio risk into components
        
        Parameters:
        -----------
        portfolio_value : float
            Portfolio value
            
        Returns:
        --------
        dict : Risk decomposition
        """
        # Today's risk
        todays_risk = (self.latest_vol / 100) * portfolio_value
        
        # 1-day, 5-day, 10-day risk
        var_1day = todays_risk
        var_5day = todays_risk * np.sqrt(5)
        var_10day = todays_risk * np.sqrt(10)
        
        return {
            '1_day_var': var_1day,
            '5_day_var': var_5day,
            '10_day_var': var_10day,
            'volatility_current_pct': self.latest_vol,
            'volatility_mean_pct': np.mean(self.conditional_volatility),
            'volatility_max_pct': np.max(self.conditional_volatility),
            'volatility_min_pct': np.min(self.conditional_volatility)
        }
    
    def get_risk_summary(
        self,
        portfolio_value: float = 100000
    ) -> pd.DataFrame:
        """
        Create comprehensive risk summary table
        
        Parameters:
        -----------
        portfolio_value : float
            Portfolio value
            
        Returns:
        --------
        pd.DataFrame : Risk summary
        """
        metrics = self.calculate_portfolio_metrics(portfolio_value)
        
        summary_data = {
            'Metric': [
                'Current Volatility (%)',
                'Annualized Volatility (%)',
                'Annualized Return (%)',
                'Sharpe Ratio',
                'Maximum Drawdown (%)',
                'VaR (95%) - Amount',
                'VaR (99%) - Amount',
                'Expected Shortfall (95%) - Amount',
                'Expected Shortfall (99%) - Amount'
            ],
            'Value': [
                f"{metrics['latest_volatility_pct']:.2f}",
                f"{metrics['annualized_volatility_pct']:.2f}",
                f"{metrics['annualized_return_pct']:.2f}",
                f"{metrics['sharpe_ratio']:.2f}",
                f"{metrics['maximum_drawdown_pct']:.2f}",
                f"₹{metrics['var_95']['var_amount']:,.2f}",
                f"₹{metrics['var_99']['var_amount']:,.2f}",
                f"₹{metrics['expected_shortfall_95']['es_amount']:,.2f}",
                f"₹{metrics['expected_shortfall_99']['es_amount']:,.2f}"
            ]
        }
        
        return pd.DataFrame(summary_data)
    
    @staticmethod
    def stress_test(
        conditional_volatility: np.ndarray,
        shocks: list = [1.5, 2.0, 2.5],
        portfolio_value: float = 100000
    ) -> Dict:
        """
        Stress test: Multiply volatility by shock factors
        
        Parameters:
        -----------
        conditional_volatility : np.ndarray
            Current conditional volatility
        shocks : list
            Shock multipliers (e.g., [1.5, 2.0, 2.5])
        portfolio_value : float
            Portfolio value
            
        Returns:
        --------
        dict : Stress test results
        """
        latest_vol = conditional_volatility[-1]
        
        stress_results = {}
        for shock in shocks:
            stressed_vol = latest_vol * shock
            stressed_loss = (stressed_vol / 100) * portfolio_value
            stressed_return = ((stressed_vol / 100) ** 2) * 250  # Rough annualization
            
            stress_results[f"{shock}x Shock"] = {
                'stressed_volatility_pct': stressed_vol,
                'estimated_daily_loss': stressed_loss,
                'annualized_volatility': stressed_vol * np.sqrt(252)
            }
        
        return stress_results

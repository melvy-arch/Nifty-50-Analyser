"""
Main Analysis Pipeline
Orchestrates data fetching, GARCH modeling, and risk analysis
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict

from data_fetcher import NiftyDataFetcher
from garch_model import GARCHModel, ModelSelector
from risk_analyzer import RiskAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NiftyVolatilityPredictor:
    """
    Complete pipeline for Nifty 50 volatility prediction and risk analysis
    """
    
    def __init__(self, lookback_years: int = 5, portfolio_value: float = 100000):
        """
        Initialize predictor
        
        Parameters:
        -----------
        lookback_years : int
            Years of historical data to use
        portfolio_value : float
            Portfolio value for VaR calculation
        """
        self.lookback_years = lookback_years
        self.portfolio_value = portfolio_value
        
        # Components
        self.data_fetcher = None
        self.garch_model = None
        self.risk_analyzer = None
        
        # Results storage
        self.returns = None
        self.data_summary = None
        self.model_order = None
        self.fitted_parameters = None
        self.volatility = None
        self.risk_metrics = None
        self.forecast_volatility = None
        
    def run_pipeline(
        self,
        max_p: int = 3,
        max_q: int = 3,
        forecast_days: int = 20,
        verbose: bool = True
    ) -> Dict:
        """
        Run complete analysis pipeline
        
        Parameters:
        -----------
        max_p : int
            Maximum ARCH order for model selection
        max_q : int
            Maximum GARCH order for model selection
        forecast_days : int
            Number of days to forecast volatility
        verbose : bool
            Print progress messages
            
        Returns:
        --------
        dict : Complete analysis results
        """
        logger.info("="*60)
        logger.info("NIFTY 50 VOLATILITY & RISK PREDICTION PIPELINE")
        logger.info("="*60)
        
        # Step 1: Fetch Data
        logger.info("\n[Step 1] Fetching historical data...")
        self._fetch_data()
        
        # Step 2: Data Summary
        logger.info("\n[Step 2] Calculating summary statistics...")
        self._calculate_data_summary()
        
        # Step 3: Select best GARCH model
        logger.info("\n[Step 3] Selecting best GARCH model...")
        self._select_and_fit_model(max_p, max_q)
        
        # Step 4: Risk Analysis
        logger.info("\n[Step 4] Performing risk analysis...")
        self._perform_risk_analysis()
        
        # Step 5: Forecasting
        logger.info("\n[Step 5] Forecasting volatility...")
        self._forecast_volatility(forecast_days)
        
        logger.info("\n" + "="*60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        
        return self._compile_results()
    
    def _fetch_data(self):
        """Fetch Nifty 50 historical data"""
        self.data_fetcher = NiftyDataFetcher(lookback_years=self.lookback_years)
        self.data_fetcher.fetch_data()
        self.returns = self.data_fetcher.calculate_returns()
        
        logger.info(f"Downloaded {len(self.returns)} trading days of data")
        logger.info(f"Date range: {self.returns.index[0].date()} to {self.returns.index[-1].date()}")
    
    def _calculate_data_summary(self):
        """Calculate summary statistics"""
        self.data_summary = self.data_fetcher.get_summary_stats()
        
        logger.info(f"Mean daily return: {self.data_summary['mean']:.4f}%")
        logger.info(f"Daily volatility: {self.data_summary['std']:.4f}%")
        logger.info(f"Skewness: {self.data_summary['skewness']:.4f}")
        logger.info(f"Kurtosis: {self.data_summary['kurtosis']:.4f}")
        logger.info(f"Min return: {self.data_summary['min']:.2f}%")
        logger.info(f"Max return: {self.data_summary['max']:.2f}%")
    
    def _select_and_fit_model(self, max_p: int, max_q: int):
        """Select and fit best GARCH model"""
        returns_scaled = self.data_fetcher.get_preprocessed_data()
        
        try:
            (p, q), model = ModelSelector.select_best_model(
                returns_scaled,
                max_p=max_p,
                max_q=max_q,
                criterion='aic'
            )
            
            self.model_order = (p, q)
            self.garch_model = model
            self.fitted_parameters = model.get_parameters_dict()
            self.volatility = model.conditional_volatility
            
            logger.info(f"\nBest model: GARCH({p},{q})")
            logger.info(f"Log-Likelihood: {model.log_likelihood:.2f}")
            logger.info(f"AIC: {model.aic:.2f}")
            logger.info(f"BIC: {model.bic:.2f}")
            
            # Print parameters
            logger.info("\nFitted Parameters:")
            logger.info(f"  μ (mean): {self.fitted_parameters['mu']:.6f}")
            logger.info(f"  ω (omega): {self.fitted_parameters['omega']:.6f}")
            logger.info(f"  α (alpha): {self.fitted_parameters['alpha']}")
            logger.info(f"  β (beta): {self.fitted_parameters['beta']}")
            
        except Exception as e:
            logger.error(f"Model selection failed: {str(e)}")
            raise
    
    def _perform_risk_analysis(self):
        """Perform comprehensive risk analysis"""
        self.risk_analyzer = RiskAnalyzer(
            self.volatility,
            self.data_fetcher.get_preprocessed_data()
        )
        
        self.risk_metrics = self.risk_analyzer.calculate_portfolio_metrics(
            self.portfolio_value
        )
        
        logger.info(f"\nPortfolio Value: ₹{self.portfolio_value:,.2f}")
        logger.info(f"Current Volatility: {self.risk_metrics['latest_volatility_pct']:.2f}%")
        logger.info(f"Annualized Volatility: {self.risk_metrics['annualized_volatility_pct']:.2f}%")
        logger.info(f"Annualized Return: {self.risk_metrics['annualized_return_pct']:.2f}%")
        logger.info(f"Sharpe Ratio: {self.risk_metrics['sharpe_ratio']:.2f}")
        logger.info(f"Maximum Drawdown: {self.risk_metrics['maximum_drawdown_pct']:.2f}%")
        
        logger.info(f"\nValue at Risk (95%): ₹{self.risk_metrics['var_95']['var_amount']:,.2f}")
        logger.info(f"Value at Risk (99%): ₹{self.risk_metrics['var_99']['var_amount']:,.2f}")
        logger.info(f"Expected Shortfall (95%): ₹{self.risk_metrics['expected_shortfall_95']['es_amount']:,.2f}")
        logger.info(f"Expected Shortfall (99%): ₹{self.risk_metrics['expected_shortfall_99']['es_amount']:,.2f}")
    
    def _forecast_volatility(self, forecast_days: int):
        """Forecast future volatility"""
        returns_scaled = self.data_fetcher.get_preprocessed_data()
        self.forecast_volatility = self.garch_model.forecast_volatility(
            returns_scaled,
            steps=forecast_days
        )
        
        logger.info(f"\nVolatility Forecast ({forecast_days} trading days):")
        logger.info(f"  Mean: {np.mean(self.forecast_volatility):.2f}%")
        logger.info(f"  Min: {np.min(self.forecast_volatility):.2f}%")
        logger.info(f"  Max: {np.max(self.forecast_volatility):.2f}%")
    
    def _compile_results(self) -> Dict:
        """Compile all results into a dictionary"""
        return {
            'data_summary': self.data_summary,
            'model_order': self.model_order,
            'fitted_parameters': self.fitted_parameters,
            'model_stats': {
                'log_likelihood': self.garch_model.log_likelihood,
                'aic': self.garch_model.aic,
                'bic': self.garch_model.bic
            },
            'risk_metrics': self.risk_metrics,
            'volatility_series': self.volatility,
            'forecast_volatility': self.forecast_volatility,
            'returns': self.returns,
            'data_fetcher': self.data_fetcher,
            'risk_analyzer': self.risk_analyzer
        }
    
    def get_summary_report(self) -> str:
        """Generate text summary report"""
        if self.risk_metrics is None:
            return "Pipeline not run yet. Call run_pipeline() first."
        
        report = []
        report.append("="*70)
        report.append("NIFTY 50 VOLATILITY & RISK ANALYSIS REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("="*70)
        
        # Data Summary
        report.append("\n1. DATA SUMMARY")
        report.append("-"*70)
        report.append(f"Mean Daily Return: {self.data_summary['mean']:.4f}%")
        report.append(f"Daily Volatility: {self.data_summary['std']:.4f}%")
        report.append(f"Observations: {self.data_summary['observations']:,}")
        report.append(f"Skewness: {self.data_summary['skewness']:.4f}")
        report.append(f"Kurtosis: {self.data_summary['kurtosis']:.4f}")
        report.append(f"Min Return: {self.data_summary['min']:.2f}%")
        report.append(f"Max Return: {self.data_summary['max']:.2f}%")
        
        # Model
        report.append("\n2. GARCH MODEL")
        report.append("-"*70)
        p, q = self.model_order
        report.append(f"Selected Model: GARCH({p},{q})")
        report.append(f"Log-Likelihood: {self.garch_model.log_likelihood:.2f}")
        report.append(f"AIC: {self.garch_model.aic:.2f}")
        report.append(f"BIC: {self.garch_model.bic:.2f}")
        
        report.append("\nFitted Parameters:")
        report.append(f"  μ (Mean): {self.fitted_parameters['mu']:.6f}")
        report.append(f"  ω (Intercept): {self.fitted_parameters['omega']:.6f}")
        report.append(f"  α (ARCH): {self.fitted_parameters['alpha']}")
        report.append(f"  β (GARCH): {self.fitted_parameters['beta']}")
        
        # Risk Metrics
        report.append("\n3. RISK METRICS")
        report.append("-"*70)
        report.append(f"Portfolio Value: ₹{self.portfolio_value:,.2f}")
        report.append(f"Current Volatility: {self.risk_metrics['latest_volatility_pct']:.2f}%")
        report.append(f"Annualized Volatility: {self.risk_metrics['annualized_volatility_pct']:.2f}%")
        report.append(f"Annualized Return: {self.risk_metrics['annualized_return_pct']:.2f}%")
        report.append(f"Sharpe Ratio: {self.risk_metrics['sharpe_ratio']:.2f}")
        report.append(f"Maximum Drawdown: {self.risk_metrics['maximum_drawdown_pct']:.2f}%")
        
        # Value at Risk
        report.append("\n4. VALUE AT RISK (VaR)")
        report.append("-"*70)
        var95 = self.risk_metrics['var_95']
        var99 = self.risk_metrics['var_99']
        report.append(f"VaR (95%): {var95['var_percentage']:.2f}% or ₹{var95['var_amount']:,.2f}")
        report.append(f"VaR (99%): {var99['var_percentage']:.2f}% or ₹{var99['var_amount']:,.2f}")
        
        # Expected Shortfall
        report.append("\n5. EXPECTED SHORTFALL (CVaR)")
        report.append("-"*70)
        es95 = self.risk_metrics['expected_shortfall_95']
        es99 = self.risk_metrics['expected_shortfall_99']
        report.append(f"Expected Shortfall (95%): {es95['es_percentage']:.2f}% or ₹{es95['es_amount']:,.2f}")
        report.append(f"Expected Shortfall (99%): {es99['es_percentage']:.2f}% or ₹{es99['es_amount']:,.2f}")
        
        # Forecast
        report.append("\n6. VOLATILITY FORECAST (20 Trading Days)")
        report.append("-"*70)
        report.append(f"Mean Forecasted Volatility: {np.mean(self.forecast_volatility):.2f}%")
        report.append(f"Min Forecasted Volatility: {np.min(self.forecast_volatility):.2f}%")
        report.append(f"Max Forecasted Volatility: {np.max(self.forecast_volatility):.2f}%")
        
        report.append("\n" + "="*70)
        
        return "\n".join(report)


if __name__ == "__main__":
    # Run pipeline
    predictor = NiftyVolatilityPredictor(
        lookback_years=5,
        portfolio_value=1000000  # 10 lakh portfolio
    )
    
    results = predictor.run_pipeline(
        max_p=3,
        max_q=3,
        forecast_days=20
    )
    
    # Print report
    print(predictor.get_summary_report())

"""
Nifty 50 Volatility & Risk Predictor
Complete package for GARCH-based volatility forecasting and risk analysis

Main Components:
- data_fetcher: Download and preprocess market data
- garch_model: GARCH model implementation with MLE
- risk_analyzer: Risk metrics and VaR calculations
- main_pipeline: Complete analysis orchestration
- app: Interactive Streamlit dashboard
"""

__version__ = "1.0"
__author__ = "Financial Analytics Team"

from .main_pipeline import NiftyVolatilityPredictor
from .data_fetcher import NiftyDataFetcher
from .garch_model import GARCHModel, ModelSelector
from .risk_analyzer import RiskAnalyzer

__all__ = [
    'NiftyVolatilityPredictor',
    'NiftyDataFetcher',
    'GARCHModel',
    'ModelSelector',
    'RiskAnalyzer'
]

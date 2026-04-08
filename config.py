"""
Configuration Module
Central place for all project configuration parameters
"""

# Data Configuration
DATA_CONFIG = {
    'ticker': '^NSEI',  # Nifty 50 ticker on yfinance
    'lookback_years': 5,
    'data_source': 'yfinance'
}

# GARCH Model Configuration
GARCH_CONFIG = {
    'max_p': 3,  # Maximum ARCH order
    'max_q': 3,  # Maximum GARCH order
    'selection_criterion': 'aic',  # 'aic' or 'bic'
    'optimization_method': 'Nelder-Mead',
    'optimizer_options': {
        'maxiter': 3000,
        'xatol': 1e-8,
        'fatol': 1e-8
    }
}

# Risk Analysis Configuration
RISK_CONFIG = {
    'var_confidence_levels': [0.95, 0.99],
    'portfolio_value': 1000000,  # 10 lakh
    'holding_period': 1,  # days
    'trading_days_per_year': 252
}

# Volatility Forecasting
FORECAST_CONFIG = {
    'forecast_horizon': 20,  # trading days
    'method': 'garch'
}

# Stress Testing
STRESS_CONFIG = {
    'shock_multipliers': [1.5, 2.0, 2.5],
    'scenario_names': {
        1.5: 'Moderate Stress',
        2.0: 'Significant Stress',
        2.5: 'Extreme Stress'
    }
}

# Logging Configuration
LOG_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
}

# Streamlit App Configuration
APP_CONFIG = {
    'page_title': 'Nifty 50 Volatility & Risk Predictor',
    'layout': 'wide',
    'theme': 'light'
}

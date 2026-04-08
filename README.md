# Nifty 50 Volatility & Risk Predictor

## Overview

A comprehensive Python project for predicting market volatility and analyzing financial risk using **GARCH (Generalized Autoregressive Conditional Heteroskedasticity)** models. This project implements advanced time-series analysis on the Nifty 50 index to forecast volatility and compute Value at Risk (VaR) metrics for portfolio management.

## Key Features

### 1. **Time-Series Analysis**
- Downloads 5 years of historical Nifty 50 data using yfinance
- Calculates log returns with proper statistical preprocessing
- Analyzes return distribution (mean, std, skewness, kurtosis)

### 2. **GARCH Modeling with MLE**
- Implements GARCH(p,q) models from scratch
- Maximum Likelihood Estimation (MLE) for parameter optimization
- Supports multiple ARCH and GARCH orders
- Recursive calculation of conditional variance/volatility

### 3. **Model Selection (AIC)**
- Compares multiple GARCH configurations (p, q combinations)
- Selects best model using Akaike Information Criterion (AIC)
- Also provides BIC scores for comparison
- Systematic grid search for optimal parameters

### 4. **Risk Analysis**
- **Value at Risk (VaR)**: Estimates maximum expected loss at 95% and 99% confidence levels
- **Expected Shortfall (CVaR)**: Average loss beyond VaR threshold
- **Volatility Forecasting**: 20-day ahead volatility predictions
- **Portfolio Metrics**: Sharpe ratio, maximum drawdown, annualized returns

### 5. **Interactive Dashboard**
- Streamlit-based web application
- Real-time parameter adjustment
- Multiple visualization tabs
- Stress testing scenarios

## Project Structure

```
stat project/
├── requirements.txt           # Python dependencies
├── data_fetcher.py           # Data downloading & preprocessing
├── garch_model.py            # GARCH modeling & MLE implementation
├── risk_analyzer.py          # Risk metrics & VaR calculations
├── main_pipeline.py          # End-to-end analysis pipeline
├── app.py                    # Streamlit dashboard
├── run_analysis.py           # Quick-start script
└── README.md                 # This file
```

## Installation

### 1. Clone/Setup the Project
```bash
cd "stat project"
```

### 2. Create Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # macOS/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Quick Analysis (CLI)
Run the complete analysis pipeline from command line:

```bash
python main_pipeline.py
```

This will:
- Download Nifty 50 data
- Fit optimal GARCH model
- Calculate risk metrics
- Generate summary report
- Print results to console

### Option 2: Interactive Dashboard
Launch the Streamlit dashboard for interactive exploration:

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

### Option 3: Custom Script
Use the modules programmatically:

```python
from main_pipeline import NiftyVolatilityPredictor

# Create predictor
predictor = NiftyVolatilityPredictor(
    lookback_years=5,
    portfolio_value=1000000  # 10 lakh
)

# Run analysis
results = predictor.run_pipeline(
    max_p=3,
    max_q=3,
    forecast_days=20
)

# Print report
print(predictor.get_summary_report())

# Access individual components
print("Risk Metrics:", results['risk_metrics'])
print("Volatility Forecast:", results['forecast_volatility'])
```

## Module Documentation

### `data_fetcher.py`
**Class: `NiftyDataFetcher`**
- `fetch_data()`: Download historical OHLCV data
- `calculate_returns()`: Compute log or simple returns
- `get_preprocessed_data()`: Returns scaled for numerical stability
- `get_summary_stats()`: Return distribution statistics

### `garch_model.py`
**Class: `GARCHModel`**
- `__init__(p, q)`: Initialize GARCH(p,q) model
- `fit(returns)`: Fit using MLE optimization
- `forecast_volatility(returns, steps)`: Forecast future volatility
- `get_parameters_dict()`: Extract fitted parameters

**Class: `ModelSelector`**
- `select_best_model()`: Grid search for optimal (p,q)

### `risk_analyzer.py`
**Class: `RiskAnalyzer`**
- `calculate_var()`: Compute Value at Risk
- `calculate_expected_shortfall()`: Compute CVaR
- `calculate_portfolio_metrics()`: Comprehensive risk stats
- `stress_test()`: Scenario analysis with volatility shocks

### `main_pipeline.py`
**Class: `NiftyVolatilityPredictor`**
- `run_pipeline()`: Execute complete analysis
- `get_summary_report()`: Generate text report

## Output Examples

### Console Output
```
============================================================
NIFTY 50 VOLATILITY & RISK PREDICTION PIPELINE
============================================================

[Step 1] Fetching historical data...
Downloaded 1252 records
Date range: 2019-12-31 to 2024-12-31

[Step 2] Calculating summary statistics...
Mean daily return: 0.0512%
Daily volatility: 1.2456%
Skewness: -0.1234
Kurtosis: 3.4567

[Step 3] Selecting best GARCH model...
GARCH(1,1): AIC=2156.34
GARCH(1,2): AIC=2158.12
...
Best model: GARCH(1,1) with AIC=2156.34

[Step 4] Performing risk analysis...
Portfolio Value: ₹1,000,000
Current Volatility: 1.24%
VaR (95%): ₹18,652
VaR (99%): ₹28,901
```

### Dashboard Features
1. **Overview Tab**: Returns distribution, historical volatility
2. **Risk Analysis Tab**: VaR/CVaR metrics, risk decomposition
3. **Volatility Tab**: Forecasts with trend indicators
4. **Model Details Tab**: Parameters, AIC/BIC scores
5. **Stress Testing Tab**: Sensitivity to volatility shocks

## Key Insights

### GARCH Model
The GARCH(p,q) model captures:
- **Volatility clustering**: High volatility tends to be followed by high volatility
- **Mean reversion**: Volatility tends to revert to long-run average
- **Conditional heteroskedasticity**: Variance changes over time

### Value at Risk
- **VaR 95%**: There's a 5% chance of losing more than this amount in 1 day
- **VaR 99%**: There's a 1% chance of losing more than this amount in 1 day
- **CVaR/ES**: Average loss when VaR threshold is breached

### Volatility Forecasting
- Next 20 trading days' volatility projections
- Useful for options pricing, hedging strategies
- Helps identify market stress periods

## Technical Details

### Statistical Methods
1. **Maximum Likelihood Estimation**: 
   - Scipy's Nelder-Mead optimizer
   - Negative log-likelihood minimization
   - Parameter constraints (positivity, stationarity)

2. **Model Selection**:
   - Akaike Information Criterion (AIC) = -2*LL + 2k
   - Bayesian Information Criterion (BIC) = -2*LL + k*ln(n)
   - Grid search: p,q ∈ {1,2,3}

3. **Risk Metrics**:
   - Historical VaR: Percentile-based
   - Parametric VaR: Normal distribution assumption
   - Expected Shortfall: Mean of tail losses

## Sample Results

For a typical 5-year analysis on ₹10 lakh portfolio:
- Best model typically: **GARCH(1,1)** (simplest, most stable)
- Current volatility: **1.2-1.5%** daily (~19-24% annualized)
- VaR 95%: **₹15,000-25,000** per day
- VaR 99%: **₹25,000-40,000** per day
- Sharpe Ratio: **0.4-0.8** (market dependent)

## Dependencies
- **numpy**: Numerical computations
- **pandas**: Data manipulation
- **scipy**: Statistical functions & optimization
- **matplotlib/seaborn**: Static plotting
- **plotly**: Interactive visualizations
- **yfinance**: Market data fetching
- **arch**: Alternative GARCH implementation
- **statsmodels**: Additional time-series tools
- **streamlit**: Web dashboard

## Advanced Usage

### Custom GARCH Fitting
```python
from garch_model import GARCHModel
from data_fetcher import NiftyDataFetcher

# Get data
fetcher = NiftyDataFetcher()
fetcher.fetch_data()
returns = fetcher.get_preprocessed_data()

# Fit custom model
model = GARCHModel(p=2, q=1)
model.fit(returns, verbose=True)

# Forecast
forecast = model.forecast_volatility(returns, steps=30)
```

### Risk Scenario Analysis
```python
from risk_analyzer import RiskAnalyzer

analyzer = RiskAnalyzer(volatility, returns)

# Stress test
stress = RiskAnalyzer.stress_test(
    volatility, 
    shocks=[1.5, 2.0, 3.0],
    portfolio_value=5000000
)
```

## Limitations & Considerations

1. **Data Quality**: Relies on yfinance data accuracy
2. **Model Assumptions**: GARCH assumes normal residuals (may not hold for extreme events)
3. **Parameter Stability**: Past volatility may not predict future perfectly
4. **Computational**: Large datasets or high-order models may take time to fit
5. **Market Regimes**: Model may perform differently in different market conditions

## Future Enhancements

- [ ] Alternative distributions (Student-t, skewed distributions)
- [ ] Multivariate GARCH for portfolio analysis
- [ ] Machine learning volatility predictions
- [ ] Real-time data streaming
- [ ] Database storage for historical results
- [ ] Email alerts for risk threshold breaches
- [ ] Support for other indices (Sensex, Nifty 500, etc.)

## References

### Academic Papers
- Bollerslev, T. (1986). "Generalized Autoregressive Conditional Heteroskedasticity"
- Engle, R. F. (1982). "Autoregressive Conditional Heteroscedasticity"
- Jorion, P. (2007). "Value at Risk: The New Benchmark"

### Libraries
- [statsmodels GARCH](https://www.statsmodels.org/)
- [ARCH Package Documentation](https://arch.readthedocs.io/)
- [yfinance](https://github.com/ranaroussi/yfinance)

## License
Free to use for educational and commercial purposes.

## Author Notes
This project demonstrates:
- Statistical modeling in Python
- Time-series analysis techniques
- Risk management principles
- Interactive data visualization
- Software engineering best practices

## Support & Troubleshooting

### Issue: "No models converged"
- Try reducing max_p and max_q
- Ensure data quality
- Check stability constraints

### Issue: Streamlit app slow
- Reduce lookback period
- Clear cache (sidebar button)
- Use smaller portfolio values

### Issue: Data fetch fails
- Check internet connection
- Verify NSE data availability
- Try different date ranges

---

**Version**: 1.0  
**Last Updated**: 2024  
**Status**: Production Ready

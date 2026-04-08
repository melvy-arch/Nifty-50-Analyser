streamlit run app.pyCtrl + C
streamlit run app.py --logger.level=warning# Technical Architecture & Implementation Details

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Data Flow](#data-flow)
3. [Module Specifications](#module-specifications)
4. [GARCH Implementation](#garch-implementation)
5. [Risk Calculations](#risk-calculations)
6. [Extension Points](#extension-points)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Streamlit Dashboard                   │
│                      (app.py)                          │
└────────────────┬────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────┐
│              Main Pipeline Orchestrator                  │
│            (main_pipeline.py - Main class)              │
└────────┬──────────────┬──────────────┬──────────────────┘
         │              │              │
    ┌────▼────┐  ┌─────▼────┐  ┌─────▼────┐
    │   Data  │  │  GARCH   │  │   Risk   │
    │ Fetcher │  │  Model   │  │ Analyzer │
    │         │  │          │  │          │
    │Nifty    │  │Model     │  │Portfolio │
    │DataFetch│  │Selector  │  │Metrics   │
    └────┬────┘  └─────┬────┘  └─────┬────┘
         │              │              │
         └──────────────┼──────────────┘
                        │
             ┌──────────▼──────────┐
             │  Configuration      │
             │  (config.py)        │
             └─────────────────────┘
```

### Layer Responsibilities

1. **Presentation Layer** (Streamlit App)
   - User interface and interaction
   - Visualization and reporting
   - Parameter management

2. **Orchestration Layer** (Main Pipeline)
   - Workflow coordination
   - Result compilation
   - End-to-end pipeline management

3. **Analysis Layer** (GARCH + Risk)
   - Model fitting and forecasting
   - Statistical calculations
   - Risk metric computation

4. **Data Layer** (Data Fetcher)
   - External data acquisition
   - Data preprocessing
   - Cache management

---

## Data Flow

### Complete Pipeline Flow

```
START
  │
  ├─→ [DATA FETCHING]
  │   ├─ Download Nifty 50 data (5 years)
  │   ├─ Validate data quality
  │   └─ Calculate log returns
  │
  ├─→ [SUMMARY STATISTICS]
  │   ├─ Mean, Std, Min, Max
  │   ├─ Skewness, Kurtosis
  │   └─ Store in data_summary
  │
  ├─→ [MODEL SELECTION]
  │   ├─ Loop p=1 to max_p
  │   │   └─ Loop q=1 to max_q
  │   │       ├─ Fit GARCH(p,q)
  │   │       ├─ Calculate AIC
  │   │       └─ Store result
  │   ├─ Select model with min AIC
  │   └─ Extract fitted parameters
  │
  ├─→ [VOLATILITY ESTIMATION]
  │   ├─ Calculate conditional variance with fitted params
  │   ├─ Extract conditional volatility
  │   └─ Store time-series
  │
  ├─→ [RISK ANALYSIS]
  │   ├─ Calculate VaR (95%, 99%)
  │   ├─ Calculate CVaR/ES
  │   ├─ Compute Sharpe Ratio
  │   ├─ Calculate Maximum Drawdown
  │   └─ Store all metrics
  │
  ├─→ [FORECASTING]
  │   ├─ Forecast next 20 trading days
  │   └─ Store forecast
  │
  └─→ [COMPILE & RETURN RESULTS]
      └─ Package all results as dict

END
```

### Data Transformations

```
Raw Price Data     →  [Log Returns]  →  [Scaled by 100]
(OHLCV)               (Daily)            (for MLE stability)
                       ↓
                    [Mean=0]
                    [Std~1-2]
                       ↓
                    [GARCH Model]
                       ↓
    ┌───────────────────┼───────────────────┐
    ↓                   ↓                   ↓
[Conditional      [Conditional         [Fitted
 Volatility]      Residuals]           Parameters]
    ↓                   ↓                   ↓
[Risk Metrics] [Diagnostics]         [Forecasts]
```

---

## Module Specifications

### 1. data_fetcher.py

**Class: `NiftyDataFetcher`**

```python
class NiftyDataFetcher:
    def __init__(self, ticker="^NSEI", lookback_years=5)
    def fetch_data() -> pd.DataFrame      # Downloads OHLCV
    def calculate_returns(method='log') -> pd.Series  # Log or simple returns
    def get_preprocessed_data() -> np.ndarray  # Scaled for MLE
    def get_summary_stats() -> dict  # Statistical summary
```

**Key Methods:**

| Method | Input | Output | Purpose |
|--------|-------|--------|---------|
| `fetch_data()` | ticker | DataFrame | Download historical prices |
| `calculate_returns()` | method | Series | Convert prices to returns |
| `get_preprocessed_data()` | - | ndarray | Prepare for GARCH |
| `get_summary_stats()` | - | dict | Basic statistics |

**Data Validation:**
- Check for missing values
- Verify date continuity
- Validate price ranges

---

### 2. garch_model.py

**Class: `GARCHModel`**

```python
class GARCHModel:
    def __init__(self, p: int, q: int)
    def fit(returns, verbose=False) -> dict  # MLE optimization
    def forecast_volatility(returns, steps=20) -> np.ndarray
    def get_parameters_dict() -> dict
```

**Mathematical Model:**

```
Mean Equation:
    r_t = μ + ε_t

Variance Equation (GARCH(p,q)):
    h_t = ω + Σ(α_i * ε²_(t-i)) + Σ(β_j * h_(t-j))
          i=1 to p           j=1 to q

Where:
    r_t = Daily return at time t
    μ = Mean return
    ε_t = Error term (assumed normal)
    h_t = Conditional variance
    ω = Intercept
    α_i = ARCH coefficients (capture shock effects)
    β_j = GARCH coefficients (capture volatility persistence)
```

**Likelihood Function:**

```python
L = -0.5 * Σ(log(h_t) + ε²_t / h_t)
    
# MLE minimizes: -L (negative log-likelihood)
```

**Constraints:**
- ω > 0 (positive variance)
- α_i ≥ 0 (non-negative coefficients)
- β_j ≥ 0
- Σα_i + Σβ_j < 1 (stationarity)

**Class: `ModelSelector`**

```python
class ModelSelector:
    @staticmethod
    def select_best_model(
        returns, max_p=3, max_q=3, criterion='aic'
    ) -> Tuple[(p,q), GARCHModel]
```

---

### 3. risk_analyzer.py

**Class: `RiskAnalyzer`**

```python
class RiskAnalyzer:
    def __init__(self, conditional_volatility, returns)
    def calculate_var(confidence_level=0.95, 
                     portfolio_value=100000) -> dict
    def calculate_expected_shortfall(confidence_level=0.95) -> dict
    def calculate_portfolio_metrics(portfolio_value) -> dict
    def stress_test(volatility, shocks, portfolio_value) -> dict
```

**VaR Calculation (Historical Method):**

```
VaR(95%) = 5th percentile of returns
VaR(99%) = 1st percentile of returns

Loss Amount = |VaR%| × Portfolio Value
```

**Expected Shortfall (CVaR):**

```
ES(95%) = Mean of returns ≤ VaR(95%)
        = Mean of worst 5% of returns

CVaR ≥ VaR (always)
```

**Sharpe Ratio:**

```
Sharpe = (Annual Return - Risk-Free Rate) / Annual Volatility
       ≈ (Annual Return) / Annual Volatility  (assuming Rf=0)
```

**Maximum Drawdown:**

```
Drawdown_t = (Cumulative Return_t - Running Max) / Running Max
MaxDD = Minimum Drawdown
      = Largest peak-to-trough decline
```

---

## GARCH Implementation

### Optimization Algorithm

**Method: Nelder-Mead Simplex**

1. Start with initial parameter guess
2. Evaluate objective function (negative log-likelihood)
3. Create simplex (n+1 points for n parameters)
4. Reflect, expand, contract until convergence
5. Return best parameters

**Implementation Details:**

```python
from scipy.optimize import minimize

result = minimize(
    self._calculate_likelihood,      # Objective: -LL
    x0=initial_params,               # Starting point
    args=(returns,),                 # Data
    method='Nelder-Mead',           # Algorithm
    options={
        'maxiter': 3000,            # Max iterations
        'xatol': 1e-8,              # Parameter tolerance
        'fatol': 1e-8               # Function tolerance
    }
)
```

### Variance Recursion Algorithm

```python
def recursive_variance(params, returns, p, q):
    mu = params[0]
    omega = params[1]
    alpha = params[2:2+p]
    beta = params[2+p:2+p+q]
    
    epsilon = returns - mu
    h = np.zeros(len(returns))
    h[0] = np.var(epsilon)  # Initial variance
    
    for t in range(1, len(returns)):
        h[t] = omega                        # Intercept
        
        # ARCH: Sum of squared past shocks
        for i in range(p):
            if t-i-1 >= 0:
                h[t] += alpha[i] * (epsilon[t-i-1]**2)
        
        # GARCH: Sum of past variances
        for j in range(q):
            if t-j-1 >= 0:
                h[t] += beta[j] * h[t-j-1]
    
    return h  # Conditional variances
```

### Information Criteria

**Akaike Information Criterion (AIC):**
```
AIC = -2 * Log-Likelihood + 2 * num_params
    = -2LL + 2k

Penalizes model complexity
Lower AIC = Better fit
```

**Bayesian Information Criterion (BIC):**
```
BIC = -2 * Log-Likelihood + k * ln(n)
    = -2LL + k*ln(n)

Penalizes complexity more than AIC
Lower BIC = Better fit
```

---

## Risk Calculations

### Value at Risk Methodologies

**1. Historical Method**
```python
var_95 = np.percentile(returns, 5)  # 5th percentile

Advantages:
- Non-parametric (no distribution assumption)
- Simple interpretation
- Captures actual market behavior

Disadvantages:
- Requires sufficient historical data
- May miss recent regime changes
```

**2. Parametric Method**
```python
# Assuming normal distribution
var_95 = mean - 1.645 * std_dev    # 95% confidence
var_99 = mean - 2.326 * std_dev    # 99% confidence

Advantages:
- Works with less data
- Fast computation

Disadvantages:
- Assumes normality (may not hold for fat tails)
- Underestimates tail risk
```

## Extension Points

### 1. Add Alternative Distributions

**Current:**
```python
# In _calculate_likelihood
# Assumes normal distribution for residuals
```

**Extension - Student-t Distribution:**
```python
def _calculate_likelihood_student_t(self, params, returns):
    # Use scipy.stats.t.logpdf instead of normal
    # Helps capture fat tails
    pass
```

### 2. Add Multivariate GARCH

**Extension - DCC-GARCH:**
```python
class DCC_GARCH:
    # Dynamic Conditional Correlation
    # For multiple indices simultaneously
    def fit_multiple_indices(self, returns_dict):
        pass
```

### 3. Add Machine Learning

**Extension - GARCH + ML:**
```python
class HybridPredictor:
    # Combine GARCH with LSTM for forecasts
    # Use ensemble methods
    pass
```

### 4. Add Real-time Streaming

**Extension - Live Updates:**
```python
class StreamingAnalyzer:
    # Connect to live data feeds
    # Update models in real-time
    # Generate alerts
    pass
```

### 5. Add Custom Constraints

**Extension - Global Optimization:**
```python
from scipy.optimize import differential_evolution

# Use global optimization instead of local
# Better for non-convex problems
```

---

## Performance Optimization

### Current Performance

| Component | Time | Optimization Level |
|-----------|------|-------------------|
| Data Download | 20-30s | High (yfinance) |
| GARCH Fit | 30-60s | Medium |
| Risk Calc | 5-10s | High |
| Dashboard | 2-5s | Medium |

### Optimization Opportunities

1. **Vectorization**
   - Use NumPy operations instead of loops
   - Apply broadcasting for batch operations

2. **Caching**
   - Cache download data
   - Store fitted models
   - Streamlit @st.cache_resource

3. **Parallel Processing**
   - Grid search in parallel
   - Multiple model fits simultaneously

4. **Algorithmic Improvements**
   - Use filtered likelihood for better convergence
   - Implement Kalman filter for state-space GARCH

---

## Testing & Quality Assurance

### Unit Test Examples

```python
import pytest
from garch_model import GARCHModel

def test_garch_fit():
    # Generate synthetic returns
    returns = np.random.normal(0, 1, 1000)
    
    # Fit model
    model = GARCHModel(1, 1)
    result = model.fit(returns)
    
    # Assertions
    assert result['converged']
    assert result['log_likelihood'] < 0
    assert model.params is not None

def test_var_calculation():
    analyzer = RiskAnalyzer(vol, returns)
    var = analyzer.calculate_var(0.95, 100000)
    
    assert var['var_amount'] > 0
    assert var['var_percentage'] < abs(returns).max()
```

---

## Deployment Considerations

### Production Checklist

- [ ] Error handling for network failures
- [ ] Data validation and sanitization
- [ ] Logging and monitoring
- [ ] Documentation of assumptions
- [ ] Regular model retraining schedule
- [ ] Backup and disaster recovery
- [ ] Security (if exposing over web)
- [ ] Performance monitoring

---

## Bibliography & References

**GARCH Papers:**
- Bollerslev, T. (1986). "Generalized autoregressive conditional heteroskedasticity"
- Engle, R. F. (1982). "Autoregressive conditional heteroscedasticity"
- Nelson, D. B. (1991). "Conditional heteroskedasticity in asset returns"

**Risk Management:**
- Jorion, P. (2007). "Value at Risk: The New Benchmark"
- Dowd, K. (2007). "Measuring Market Risk"

**Implementation:**
- scipy.optimize documentation
- numpy documentation
- pandas time-series guide

---

**Version**: 1.0  
**Technical Level**: Advanced  
**Last Updated**: 2024

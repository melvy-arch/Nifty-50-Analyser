# Getting Started Guide

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation Steps](#installation-steps)
3. [Running the Analysis](#running-the-analysis)
4. [Dashboard Usage](#dashboard-usage)
5. [Troubleshooting](#troubleshooting)
6. [Next Steps](#next-steps)

## System Requirements

- **Python**: 3.8 or higher
- **OS**: Windows, macOS, or Linux
- **RAM**: Minimum 4GB (8GB recommended)
- **Disk Space**: 500MB (including dependencies)
- **Internet**: Required for data download (yfinance)

## Installation Steps

### Step 1: Navigate to Project Directory
```bash
cd "stat project"
```

### Step 2: Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Verification:**
```bash
python -c "import pandas, numpy, arch; print('✓ Dependencies installed')"
```

## Running the Analysis

### Option 1: Quick CLI Analysis (Fastest)
```bash
python run_analysis.py
```

**Output:**
- Prints comprehensive report to console
- Saves report to timestamped text file
- Takes: ~2-3 minutes

### Option 2: Interactive Streamlit Dashboard (Recommended)
```bash
streamlit run app.py
```

**Features:**
- Real-time parameter adjustment
- Interactive visualizations
- Multiple analysis tabs
- Risk scenario exploration

**Access:**
- Opens automatically at `http://localhost:8501`
- Or manually visit in browser

### Option 3: Programmatic Usage
```python
from main_pipeline import NiftyVolatilityPredictor

predictor = NiftyVolatilityPredictor(
    lookback_years=5,
    portfolio_value=1000000
)

results = predictor.run_pipeline()
print(predictor.get_summary_report())
```

### Option 4: Run Examples
```bash
# Run all examples
python examples.py

# Run specific example
python examples.py 1  # Example 1: Basic Analysis
python examples.py 5  # Example 5: Stress Testing
```

## Dashboard Usage

### Main Interface
1. **Sidebar Configuration:**
   - Adjust portfolio value
   - Change lookback period
   - Click "Run Analysis" button

2. **Tabs:**
   - **Overview**: Returns distribution, volatility history
   - **Risk Analysis**: VaR, CVaR, risk decomposition
   - **Volatility**: 20-day forecast with trends
   - **Model Details**: GARCH parameters, AIC/BIC
   - **Stress Testing**: Scenario analysis

### Key Visualizations
- **Returns Distribution**: Histogram with normal overlay
- **Volatility Series**: Time-series of conditional volatility
- **Forecast**: 20-day ahead predictions
- **Risk Bars**: VaR comparison at different confidence levels

### Dashboard Tips
- Use "Run Analysis" button to refresh with new parameters
- Hover over metrics for additional information
- Download plots using plotly's save icon
- Dashboard auto-caches for faster interactions

## Common Use Cases

### Use Case 1: Daily Risk Monitoring
```bash
streamlit run app.py
# Check VaR and volatility daily
# Adjust portfolio size as needed
```

### Use Case 2: Batch Analysis
```bash
python run_analysis.py
# Gets text report for archiving
# Can schedule as cron job (Linux/macOS)
```

### Use Case 3: Model Comparison
```python
python examples.py 2  # Compares GARCH(1,1), (1,2), (2,1), (2,2)
```

### Use Case 4: Portfolio Stress Testing
```python
python examples.py 5  # Tests multiple shock scenarios
```

## Troubleshooting

### Issue 1: "No module named 'arch'"
**Solution:**
```bash
pip install arch --upgrade
# or
pip install -r requirements.txt --force-reinstall
```

### Issue 2: Data Download Fails
**Causes:** Internet issues, yfinance service down
**Solution:**
```python
# Test data connection
from data_fetcher import NiftyDataFetcher
fetcher = NiftyDataFetcher()
data = fetcher.fetch_data()
print("✓ Data fetched successfully")
```

### Issue 3: "No models converged"
**Causes:** Poor initial parameters, unstable data
**Solution:**
```python
from main_pipeline import NiftyVolatilityPredictor
predictor = NiftyVolatilityPredictor(lookback_years=2)  # Try less data
results = predictor.run_pipeline(max_p=2, max_q=2)  # Try lower orders
```

### Issue 4: Streamlit App Slow
**Solution:**
- Reduce lookback period (3 years instead of 5)
- Clear cache: Click "Run Analysis" with sidebar button
- Use smaller portfolio values initially
- Close other intensive applications

### Issue 5: Port Already in Use
**Error:** `Address already in use`
**Solution:**
```bash
streamlit run app.py --server.port 8502
```

### Issue 6: OutOfMemory Error
**Causes:** Too much data with high GARCH orders
**Solution:**
```bash
# Reduce data
python examples.py 1  # Uses 3 years instead of 5
# Or reduce model complexity
python main_pipeline.py  # Manually set lower max_p, max_q
```

## File Structure Reference

```
stat project/
├── data_fetcher.py          # ← Data downloading
├── garch_model.py           # ← GARCH implementation
├── risk_analyzer.py         # ← Risk calculations
├── main_pipeline.py         # ← Main orchestrator
├── app.py                   # ← Streamlit dashboard
├── examples.py              # ← Usage examples
├── run_analysis.py          # ← Quick start
├── config.py                # ← Configuration
├── requirements.txt         # ← Dependencies
├── README.md                # ← Full documentation
└── GETTING_STARTED.md       # ← This file
```

## Performance Expectations

| Operation | Time | RAM Used |
|-----------|------|----------|
| Data download | 20-30s | 50-100MB |
| GARCH fitting | 30-60s | 100-200MB |
| Risk calculation | 5-10s | 50MB |
| Dashboard load | 2-5s | 150-300MB |

**Total end-to-end: 1-3 minutes**

## Environment Variables (Optional)

For advanced users, set these to customize behavior:

```bash
# Windows PowerShell
$env:NIFTY_CACHE_DIR = "C:\Temp\nifty_cache"
$env:NIFTY_DATA_DIR = "C:\Data\nifty"

# Linux/macOS
export NIFTY_CACHE_DIR="/tmp/nifty"
export NIFTY_DATA_DIR="~/Data/nifty"
```

## Next Steps

1. **Run Quick Analysis**
   ```bash
   python run_analysis.py
   ```

2. **Explore Dashboard**
   ```bash
   streamlit run app.py
   ```

3. **Run Examples**
   ```bash
   python examples.py
   ```

4. **Customize for Your Portfolio**
   - Edit `config.py` for your parameters
   - Modify `main_pipeline.py` for custom analysis

5. **Integrate with Other Tools**
   - Export results to Excel/CSV
   - Use results in trading strategies
   - Build on top with additional metrics

## Support Resources

- **Data Issues**: Check [yfinance documentation](https://github.com/ranaroussi/yfinance)
- **GARCH Theory**: See README.md references section
- **Streamlit Help**: [streamlit.io/docs](https://docs.streamlit.io)
- **Python Issues**: Use standard debugging + print statements

## Quick Reference Commands

```bash
# Setup
python -m venv venv
venv\Scripts\activate  # or: source venv/bin/activate
pip install -r requirements.txt

# Run
python run_analysis.py          # CLI analysis
streamlit run app.py            # Interactive dashboard
python examples.py              # Run all examples
python examples.py 1            # Run example 1

# Verify
python -c "from main_pipeline import NiftyVolatilityPredictor; print('✓')"
```

---

**Version**: 1.0  
**Status**: Production Ready  
**Last Updated**: 2024

**Ready to start? Run:**
```bash
python run_analysis.py
```

"""
Quick Start Script
Run this to perform a complete analysis without dashboard
"""

import sys
from main_pipeline import NiftyVolatilityPredictor


def main():
    """Run complete analysis"""
    
    print("\n" + "="*70)
    print("NIFTY 50 VOLATILITY & RISK PREDICTOR - Quick Start")
    print("="*70)
    
    # Parameters
    lookback_years = 5
    portfolio_value = 1000000  # 10 lakh
    max_p = 3
    max_q = 3
    forecast_days = 20
    
    print(f"\nConfiguration:")
    print(f"  Lookback Period: {lookback_years} years")
    print(f"  Portfolio Value: ₹{portfolio_value:,}")
    print(f"  Max ARCH Order (p): {max_p}")
    print(f"  Max GARCH Order (q): {max_q}")
    print(f"  Forecast Horizon: {forecast_days} days")
    
    # Initialize predictor
    predictor = NiftyVolatilityPredictor(
        lookback_years=lookback_years,
        portfolio_value=portfolio_value
    )
    
    # Run pipeline
    print("\nRunning analysis pipeline...")
    results = predictor.run_pipeline(
        max_p=max_p,
        max_q=max_q,
        forecast_days=forecast_days,
        verbose=True
    )
    
    # Print comprehensive report
    print("\n" + predictor.get_summary_report())
    
    # Save report to file
    report_filename = f"nifty_risk_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_filename, 'w') as f:
        f.write(predictor.get_summary_report())
    print(f"\n✅ Report saved to: {report_filename}")
    
    # Launch dashboard prompt
    print("\n" + "="*70)
    print("Want to explore the results interactively?")
    print("Run: streamlit run app.py")
    print("="*70)


if __name__ == "__main__":
    import pandas as pd
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Check internet connection (for data download)")
        print("2. Verify all dependencies installed: pip install -r requirements.txt")
        print("3. Check if NSE data is available")
        sys.exit(1)

"""
Advanced Examples and Use Cases
Demonstrates different ways to use the Nifty Volatility Predictor
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from data_fetcher import NiftyDataFetcher
from garch_model import GARCHModel, ModelSelector
from risk_analyzer import RiskAnalyzer
from main_pipeline import NiftyVolatilityPredictor


def example_1_basic_analysis():
    """
    Example 1: Basic end-to-end analysis with default parameters
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic End-to-End Analysis")
    print("="*70)
    
    predictor = NiftyVolatilityPredictor(
        lookback_years=3,
        portfolio_value=500000
    )
    
    results = predictor.run_pipeline()
    print(predictor.get_summary_report())


def example_2_custom_garch():
    """
    Example 2: Fit custom GARCH models and compare
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Custom GARCH Model Fitting & Comparison")
    print("="*70)
    
    # Fetch data
    fetcher = NiftyDataFetcher(lookback_years=3)
    fetcher.fetch_data()
    returns = fetcher.get_preprocessed_data()
    
    print(f"\nTesting different GARCH configurations:")
    print(f"Data points: {len(returns)}")
    print(f"Mean return: {fetcher.get_summary_stats()['mean']:.4f}%\n")
    
    # Test different configurations
    configurations = [(1,1), (1,2), (2,1), (2,2)]
    results = []
    
    for p, q in configurations:
        model = GARCHModel(p, q)
        fit_result = model.fit(returns, verbose=False)
        
        results.append({
            'Model': f'GARCH({p},{q})',
            'Converged': fit_result['converged'],
            'LL': f"{fit_result['log_likelihood']:.2f}",
            'AIC': f"{fit_result['aic']:.2f}",
            'BIC': f"{fit_result['bic']:.2f}"
        })
        
        print(f"GARCH({p},{q}): AIC={fit_result['aic']:.2f}, "
              f"BIC={fit_result['bic']:.2f}, Converged={fit_result['converged']}")
    
    df_results = pd.DataFrame(results)
    print("\n" + df_results.to_string(index=False))


def example_3_volatility_forecast():
    """
    Example 3: Detailed volatility forecasting
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Detailed Volatility Forecasting")
    print("="*70)
    
    # Setup
    predictor = NiftyVolatilityPredictor(lookback_years=5, portfolio_value=1000000)
    results = predictor.run_pipeline(forecast_days=60)
    
    forecast_vol = predictor.forecast_volatility
    
    # Print forecast
    print(f"\n60-Day Volatility Forecast:")
    print(f"Current Volatility: {predictor.risk_metrics['latest_volatility_pct']:.2f}%")
    print(f"Mean Forecast: {np.mean(forecast_vol):.2f}%")
    print(f"Std Dev: {np.std(forecast_vol):.2f}%")
    print(f"Min: {np.min(forecast_vol):.2f}%")
    print(f"Max: {np.max(forecast_vol):.2f}%")
    
    # Forecast by period
    print("\nForecast by Period:")
    print(f"Week 1 (5 days):  {np.mean(forecast_vol[:5]):.2f}%")
    print(f"Week 2-4 (15 days): {np.mean(forecast_vol[5:20]):.2f}%")
    print(f"Week 5-12 (40 days): {np.mean(forecast_vol[20:]):.2f}%")


def example_4_risk_decomposition():
    """
    Example 4: Detailed risk decomposition analysis
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Risk Decomposition Analysis")
    print("="*70)
    
    portfolio_value = 5000000  # 50 lakh
    
    predictor = NiftyVolatilityPredictor(
        lookback_years=5,
        portfolio_value=portfolio_value
    )
    results = predictor.run_pipeline(verbose=False)
    
    # Risk decomposition
    risk_decomp = predictor.risk_analyzer.calculate_risk_decomposition(portfolio_value)
    
    print(f"\nPortfolio Value: ₹{portfolio_value:,}")
    print(f"Current Volatility: {risk_decomp['volatility_current_pct']:.2f}%\n")
    
    print("Risk Over Different Holding Periods:")
    print(f"1-Day Risk (VaR): ₹{risk_decomp['1_day_var']:,.0f}")
    print(f"5-Day Risk (Var): ₹{risk_decomp['5_day_var']:,.0f}")
    print(f"10-Day Risk (VaR): ₹{risk_decomp['10_day_var']:,.0f}")
    
    print("\nVolatility Statistics:")
    print(f"Current: {risk_decomp['volatility_current_pct']:.2f}%")
    print(f"Mean: {risk_decomp['volatility_mean_pct']:.2f}%")
    print(f"Max: {risk_decomp['volatility_max_pct']:.2f}%")
    print(f"Min: {risk_decomp['volatility_min_pct']:.2f}%")


def example_5_stress_testing():
    """
    Example 5: Comprehensive stress testing
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Stress Testing & Scenario Analysis")
    print("="*70)
    
    portfolio_value = 2000000
    
    predictor = NiftyVolatilityPredictor(
        lookback_years=5,
        portfolio_value=portfolio_value
    )
    results = predictor.run_pipeline(verbose=False)
    
    print(f"\nPortfolio Value: ₹{portfolio_value:,}")
    print(f"Current Volatility: {predictor.risk_metrics['latest_volatility_pct']:.2f}%\n")
    
    # Stress test with various shocks
    shocks = [1.2, 1.5, 2.0, 2.5, 3.0]
    stress_results = RiskAnalyzer.stress_test(
        predictor.volatility,
        shocks=shocks,
        portfolio_value=portfolio_value
    )
    
    print("Shock Scenario Analysis:")
    print("-" * 70)
    print(f"{'Shock Level':<15} | {'Stressed Vol':<15} | {'Daily Loss':<20} | {'Ann. Vol':<15}")
    print("-" * 70)
    
    for shock in shocks:
        shock_label = f"{shock:.1f}x"
        metrics = RiskAnalyzer.stress_test(
            predictor.volatility,
            shocks=[shock],
            portfolio_value=portfolio_value
        )[f"{shock_label} Shock"]
        
        print(f"{shock_label:<15} | {metrics['stressed_volatility_pct']:>6.2f}% | "
              f"₹{metrics['estimated_daily_loss']:>17,.0f} | {metrics['annualized_volatility']:>6.2f}%")


def example_6_portfolio_comparison():
    """
    Example 6: Compare risk metrics for different portfolio sizes
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Portfolio Size Comparison")
    print("="*70)
    
    predictor = NiftyVolatilityPredictor(lookback_years=5)
    results = predictor.run_pipeline(verbose=False)
    
    # Different portfolio sizes
    portfolio_sizes = [100000, 500000, 1000000, 5000000, 10000000]
    
    print("\nRisk Metrics for Different Portfolio Sizes:")
    print("-" * 100)
    print(f"{'Portfolio':<15} | {'VaR 95%':<15} | {'VaR 99%':<15} | {'ES 95%':<15} | {'ES 99%':<15}")
    print("-" * 100)
    
    for pv in portfolio_sizes:
        metrics = predictor.risk_analyzer.calculate_portfolio_metrics(pv)
        print(f"₹{pv/100000:>6.0f} Lakh      | ₹{metrics['var_95']['var_amount']:>12,.0f} | "
              f"₹{metrics['var_99']['var_amount']:>12,.0f} | ₹{metrics['expected_shortfall_95']['es_amount']:>12,.0f} | "
              f"₹{metrics['expected_shortfall_99']['es_amount']:>12,.0f}")


def example_7_extended_analysis():
    """
    Example 7: Extended analysis with multiple metrics
    """
    print("\n" + "="*70)
    print("EXAMPLE 7: Extended Analysis Dashboard")
    print("="*70)
    
    predictor = NiftyVolatilityPredictor(
        lookback_years=5,
        portfolio_value=1000000
    )
    results = predictor.run_pipeline(verbose=False)
    
    # Comprehensive metrics table
    metrics_table = predictor.risk_analyzer.get_risk_summary(1000000)
    print("\n" + metrics_table.to_string(index=False))
    
    # Model coefficients
    print("\n\nGARCH Model Coefficients:")
    print("-" * 50)
    params = predictor.fitted_parameters
    print(f"Mean (μ):        {params['mu']:.8f}")
    print(f"Intercept (ω):   {params['omega']:.8f}")
    print(f"ARCH (α):        {params['alpha']}")
    print(f"GARCH (β):       {params['beta']}")
    
    # Persistence
    persistence = np.sum(params['alpha']) + np.sum(params['beta'])
    print(f"\nModel Persistence: {persistence:.4f}")
    if persistence < 1:
        print("✓ Model is stationary (persistence < 1)")
    else:
        print("✗ Warning: Model may be non-stationary")


def run_all_examples():
    """Run all examples"""
    try:
        example_1_basic_analysis()
        example_2_custom_garch()
        example_3_volatility_forecast()
        example_4_risk_decomposition()
        example_5_stress_testing()
        example_6_portfolio_comparison()
        example_7_extended_analysis()
        
        print("\n" + "="*70)
        print("✅ All examples completed successfully!")
        print("="*70)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        examples = {
            '1': example_1_basic_analysis,
            '2': example_2_custom_garch,
            '3': example_3_volatility_forecast,
            '4': example_4_risk_decomposition,
            '5': example_5_stress_testing,
            '6': example_6_portfolio_comparison,
            '7': example_7_extended_analysis,
        }
        
        if example_num in examples:
            examples[example_num]()
        else:
            print(f"Example {example_num} not found. Available: 1-7")
    else:
        # Run all
        run_all_examples()

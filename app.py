"""
Streamlit Dashboard for Nifty 50 Volatility & Risk Analysis
Interactive visualization and risk metrics dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Import modules
from data_fetcher import NiftyDataFetcher
from garch_model import ModelSelector
from risk_analyzer import RiskAnalyzer
from main_pipeline import NiftyVolatilityPredictor

# Page configuration
st.set_page_config(
    page_title="Nifty 50 Volatility & Risk Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_data_cached(portfolio_value, lookback_years):
    """Cache the data loading and analysis"""
    predictor = NiftyVolatilityPredictor(
        lookback_years=lookback_years,
        portfolio_value=portfolio_value
    )
    # Use lower complexity for faster results
    results = predictor.run_pipeline(max_p=2, max_q=2, forecast_days=20)
    return predictor, results


def plot_returns_distribution(returns):
    """Plot returns distribution"""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=returns * 100,
        nbinsx=50,
        name='Daily Returns',
        marker_color='#1f77b4',
        opacity=0.7
    ))
    
    # Add normal distribution overlay
    mean = np.mean(returns * 100)
    std = np.std(returns * 100)
    x_range = np.linspace(mean - 4*std, mean + 4*std, 100)
    normal_dist = len(returns) * 0.05 * (1/(std * np.sqrt(2*np.pi))) * np.exp(-0.5*((x_range-mean)/std)**2)
    
    fig.add_trace(go.Scatter(
        x=x_range, y=normal_dist,
        mode='lines',
        name='Normal Distribution',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title="Distribution of Daily Returns",
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
        hovermode='x unified',
        height=400
    )
    
    return fig


def plot_volatility_series(volatility, returns_dates):
    """Plot conditional volatility over time"""
    fig = go.Figure()
    
    # Conditional volatility
    fig.add_trace(go.Scatter(
        x=returns_dates,
        y=volatility,
        mode='lines',
        name='Conditional Volatility (GARCH)',
        line=dict(color='#2ca02c', width=2)
    ))
    
    fig.update_layout(
        title="Conditional Volatility Over Time",
        xaxis_title="Date",
        yaxis_title="Volatility (%)",
        hovermode='x unified',
        height=400,
        template='plotly_white'
    )
    
    return fig


def plot_volatility_forecast(forecast, forecast_dates):
    """Plot volatility forecast"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast,
        mode='lines+markers',
        name='Forecasted Volatility',
        line=dict(color='#ff7f0e', width=2),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="20-Day Volatility Forecast",
        xaxis_title="Trading Days Ahead",
        yaxis_title="Volatility (%)",
        hovermode='x unified',
        height=400,
        template='plotly_white'
    )
    
    return fig


def plot_var_chart(portfolio_value, risk_metrics):
    """Plot VaR distribution"""
    var_data = {
        'Confidence Level': ['95%', '99%'],
        'Value at Risk': [
            risk_metrics['var_95']['var_amount'],
            risk_metrics['var_99']['var_amount']
        ],
        'Expected Shortfall': [
            risk_metrics['expected_shortfall_95']['es_amount'],
            risk_metrics['expected_shortfall_99']['es_amount']
        ]
    }
    
    df_var = pd.DataFrame(var_data)
    
    fig = go.Figure(data=[
        go.Bar(x=df_var['Confidence Level'], y=df_var['Value at Risk'], 
               name='Value at Risk', marker_color='#d62728'),
        go.Bar(x=df_var['Confidence Level'], y=df_var['Expected Shortfall'], 
               name='Expected Shortfall', marker_color='#9467bd')
    ])
    
    fig.update_layout(
        title="Risk Metrics Comparison",
        xaxis_title="Confidence Level",
        yaxis_title="Amount (₹)",
        barmode='group',
        height=400,
        template='plotly_white'
    )
    
    return fig


def main():
    # Header
    st.markdown("""
        <h1 style='text-align: center; color: #1f77b4;'>
        📊 Nifty 50 Volatility & Risk Predictor
        </h1>
        <p style='text-align: center; color: #666;'>
        GARCH-based Time-Series Analysis with Value at Risk Dashboard
        </p>
    """, unsafe_allow_html=True)
    
    # Sidebar for parameters
    st.sidebar.header("⚙️ Configuration")
    
    portfolio_value = st.sidebar.number_input(
        "Portfolio Value (₹)",
        min_value=10000,
        max_value=100000000,
        value=1000000,
        step=100000
    )
    
    lookback_months = st.sidebar.slider(
        "Lookback Period (Months)",
        min_value=1,
        max_value=120,
        value=1,
        step=1
    )
    
    # Convert months to years
    lookback_years = lookback_months / 12
    
    st.sidebar.caption(f"📅 Loading ~{lookback_months} months of data (~{int(lookback_months * 20)} trading days)")
    
    refresh_button = st.sidebar.button("🔄 Run Analysis", use_container_width=True)
    
    # Run analysis
    with st.spinner("Running volatility and risk analysis..."):
        if refresh_button:
            st.cache_resource.clear()
        
        predictor, results = load_data_cached(portfolio_value, lookback_years)
    
    # Extract data
    returns = predictor.returns.values * 100
    returns_dates = predictor.returns.index
    volatility = predictor.volatility
    forecast_vol = predictor.forecast_volatility
    risk_metrics = predictor.risk_metrics
    fitted_params = predictor.fitted_parameters
    model_order = predictor.model_order
    
    # Create forecast dates
    last_date = returns_dates[-1]
    forecast_dates = pd.date_range(
        start=last_date + timedelta(days=1),
        periods=len(forecast_vol),
        freq='B'
    )
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", "🎲 Risk Analysis", "📉 Volatility", 
        "🔬 Model Details", "⚠️ Stress Testing"
    ])
    
    with tab1:
        st.header("Executive Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Volatility", 
                     f"{risk_metrics['latest_volatility_pct']:.2f}%")
        with col2:
            st.metric("Annualized Volatility", 
                     f"{risk_metrics['annualized_volatility_pct']:.2f}%")
        with col3:
            st.metric("Sharpe Ratio", 
                     f"{risk_metrics['sharpe_ratio']:.2f}")
        with col4:
            st.metric("Max Drawdown", 
                     f"{risk_metrics['maximum_drawdown_pct']:.2f}%")
        
        st.divider()
        
        # Distribution
        st.subheader("Returns Distribution")
        fig_dist = plot_returns_distribution(returns/100)
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Volatility series
        st.subheader("Historical Conditional Volatility")
        fig_vol = plot_volatility_series(volatility, returns_dates)
        st.plotly_chart(fig_vol, use_container_width=True)
    
    with tab2:
        st.header("Risk Metrics")
        
        # Key risk metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("VaR (95%)", 
                     f"₹{risk_metrics['var_95']['var_amount']:,.0f}",
                     help="Maximum expected loss at 95% confidence")
        with col2:
            st.metric("VaR (99%)", 
                     f"₹{risk_metrics['var_99']['var_amount']:,.0f}",
                     help="Maximum expected loss at 99% confidence")
        with col3:
            st.metric("ES (95%)", 
                     f"₹{risk_metrics['expected_shortfall_95']['es_amount']:,.0f}",
                     help="Average loss beyond VaR at 95%")
        with col4:
            st.metric("ES (99%)", 
                     f"₹{risk_metrics['expected_shortfall_99']['es_amount']:,.0f}",
                     help="Average loss beyond VaR at 99%")
        
        st.divider()
        
        # VaR chart
        fig_var = plot_var_chart(portfolio_value, risk_metrics)
        st.plotly_chart(fig_var, use_container_width=True)
        
        # Risk decomposition
        st.subheader("Risk Decomposition")
        risk_decomp = predictor.risk_analyzer.calculate_risk_decomposition(portfolio_value)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("1-Day VaR", 
                     f"₹{risk_decomp['1_day_var']:,.0f}")
        with col2:
            st.metric("5-Day VaR", 
                     f"₹{risk_decomp['5_day_var']:,.0f}")
        with col3:
            st.metric("10-Day VaR", 
                     f"₹{risk_decomp['10_day_var']:,.0f}")
    
    with tab3:
        st.header("Volatility Analysis & Forecast")
        
        # Forecast chart
        st.subheader("20-Day Volatility Forecast")
        fig_forecast = plot_volatility_forecast(forecast_vol, forecast_dates)
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Forecast statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean Forecast", f"{np.mean(forecast_vol):.2f}%")
        with col2:
            st.metric("Min Forecast", f"{np.min(forecast_vol):.2f}%")
        with col3:
            st.metric("Max Forecast", f"{np.max(forecast_vol):.2f}%")
        with col4:
            st.metric("Trend", 
                     "↑ Increasing" if forecast_vol[-1] > forecast_vol[0] else "↓ Decreasing")
    
    with tab4:
        st.header("GARCH Model Details")
        
        # Model info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Type", f"GARCH({model_order[0]},{model_order[1]})")
        with col2:
            st.metric("Log-Likelihood", f"{predictor.garch_model.log_likelihood:.2f}")
        with col3:
            st.metric("AIC Score", f"{predictor.garch_model.aic:.2f}")
        
        st.divider()
        
        # Parameters
        st.subheader("Fitted Parameters")
        
        param_data = {
            'Parameter': ['μ (Mean)', 'ω (Intercept)', 'α (ARCH)', 'β (GARCH)'],
            'Value': [
                f"{fitted_params['mu']:.6f}",
                f"{fitted_params['omega']:.6f}",
                str(np.round(fitted_params['alpha'], 6)),
                str(np.round(fitted_params['beta'], 6))
            ]
        }
        
        st.dataframe(pd.DataFrame(param_data), use_container_width=True)
        
        # Data summary
        st.subheader("Data Summary Statistics")
        summary_stats = {
            'Statistic': ['Mean', 'Std Dev', 'Min', 'Max', 'Skewness', 'Kurtosis'],
            'Value': [
                f"{predictor.data_summary['mean']:.4f}%",
                f"{predictor.data_summary['std']:.4f}%",
                f"{predictor.data_summary['min']:.2f}%",
                f"{predictor.data_summary['max']:.2f}%",
                f"{predictor.data_summary['skewness']:.4f}",
                f"{predictor.data_summary['kurtosis']:.4f}"
            ]
        }
        st.dataframe(pd.DataFrame(summary_stats), use_container_width=True)
    
    with tab5:
        st.header("Stress Testing")
        
        st.info("Stress test scenario: Multiply current volatility by shock factors")
        
        # Stress test
        stress_results = RiskAnalyzer.stress_test(
            volatility,
            shocks=[1.5, 2.0, 2.5],
            portfolio_value=portfolio_value
        )
        
        stress_data = []
        for shock_label, metrics in stress_results.items():
            stress_data.append({
                'Shock': shock_label,
                'Stressed Vol (%)': f"{metrics['stressed_volatility_pct']:.2f}%",
                'Daily Loss (₹)': f"₹{metrics['estimated_daily_loss']:,.0f}",
                'Ann. Vol (%)': f"{metrics['annualized_volatility']:.2f}%"
            })
        
        st.dataframe(pd.DataFrame(stress_data), use_container_width=True)
        
        # Interpretation
        st.subheader("Interpretation")
        st.markdown("""
        - **1.5x Shock**: Moderate market stress scenario
        - **2.0x Shock**: Significant market volatility increase
        - **2.5x Shock**: Extreme market stress scenario (e.g., market crash)
        
        These scenarios help estimate potential portfolio losses under different 
        volatility regimes and assist in developing risk management strategies.
        """)


if __name__ == "__main__":
    main()

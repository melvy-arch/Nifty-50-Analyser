"""
GARCH Modeling Module
Implements GARCH models with Maximum Likelihood Estimation (MLE)
and model selection using Akaike Information Criterion (AIC)
"""

import numpy as np
from scipy import stats
import pandas as pd
import warnings
import logging
from typing import Tuple, Dict

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class GARCHModel:
    """
    GARCH(p, q) model with Maximum Likelihood Estimation
    
    Implements:
    - Mean equation: r_t = mu + epsilon_t
    - Variance equation: h_t = omega + sum(alpha_i * epsilon_t-i^2) + sum(beta_j * h_t-j)
    """
    
    def __init__(self, p: int, q: int):
        """
        Initialize GARCH(p, q) model
        
        Parameters:
        -----------
        p : int
            Order of ARCH component (q)
        q : int
            Order of GARCH component (p)
        """
        self.p = p
        self.q = q
        self.params = None
        self.log_likelihood = None
        self.aic = None
        self.bic = None
        self.converged = False
        self.conditional_volatility = None
        self.residuals = None
        
    def initialize_params(self, returns: np.ndarray) -> np.ndarray:
        """
        Initialize parameters for optimization
        
        Parameters:
        -----------
        returns : np.ndarray
            Daily returns (in percentage)
            
        Returns:
        --------
        np.ndarray : Initial parameter guess
        """
        # Mean
        mu = np.mean(returns)
        
        # Variance components
        long_run_var = np.var(returns)
        
        # Initial omega (intercept)
        omega = 0.1 * long_run_var
        
        # Initial ARCH coefficients (alpha)
        alpha = [0.05 / self.p] * self.p
        
        # Initial GARCH coefficients (beta)
        beta = [0.85 / self.q] * self.q
        
        params = [mu] + [omega] + alpha + beta
        return np.array(params)
    
    def _calculate_likelihood(self, params: np.ndarray, returns: np.ndarray) -> float:
        """
        Calculate negative log-likelihood (for minimization)
        
        Parameters:
        -----------
        params : np.ndarray
            Model parameters [mu, omega, alpha1...alphap, beta1...betaq]
        returns : np.ndarray
            Daily returns
            
        Returns:
        --------
        float : Negative log-likelihood
        """
        mu = params[0]
        omega = params[1]
        alpha = params[2:2+self.p]
        beta = params[2+self.p:2+self.p+self.q]
        
        # Residuals
        epsilon = returns - mu
        
        # Initialize conditional variance
        n_obs = len(returns)
        h = np.zeros(n_obs)
        h[0] = np.var(epsilon)
        
        # Validate parameters
        if omega <= 0:
            return 1e10
        if np.any(alpha < 0) or np.any(beta < 0):
            return 1e10
        if np.sum(alpha) + np.sum(beta) >= 1:
            return 1e10
        
        # Recursive calculation of conditional variance
        try:
            for t in range(1, n_obs):
                h[t] = omega
                
                # ARCH component
                for i in range(self.p):
                    if t - i - 1 >= 0:
                        h[t] += alpha[i] * (epsilon[t - i - 1] ** 2)
                
                # GARCH component
                for j in range(self.q):
                    if t - j - 1 >= 0:
                        h[t] += beta[j] * h[t - j - 1]
                
                if h[t] <= 0:
                    return 1e10
        
        except:
            return 1e10
        
        # Log-likelihood
        ll = -0.5 * np.sum(np.log(h) + (epsilon ** 2) / h)
        
        return -ll  # Negative because we minimize
    
    def fit(self, returns: np.ndarray, verbose: bool = False) -> Dict:
        """
        Fit GARCH model using Maximum Likelihood Estimation
        
        Parameters:
        -----------
        returns : np.ndarray
            Daily returns (in percentage)
        verbose : bool
            Print optimization details
            
        Returns:
        --------
        dict : Fitted parameters and statistics
        """
        from scipy.optimize import minimize
        
        # Initial parameter guess
        x0 = self.initialize_params(returns)
        
        # Optimize
        result = minimize(
            self._calculate_likelihood,
            x0,
            args=(returns,),
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        
        self.params = result.x
        self.converged = result.success
        
        if self.converged:
            # Calculate final statistics
            self.log_likelihood = -result.fun
            n_params = len(self.params)
            n_obs = len(returns)
            
            # AIC and BIC for model selection
            self.aic = -2 * self.log_likelihood + 2 * n_params
            self.bic = -2 * self.log_likelihood + n_params * np.log(n_obs)
            
            # Calculate conditional volatility
            self._calculate_conditional_volatility(returns)
            
            if verbose:
                logger.info(f"GARCH({self.p},{self.q}) fitted successfully")
                logger.info(f"Log-Likelihood: {self.log_likelihood:.2f}")
                logger.info(f"AIC: {self.aic:.2f}")
                logger.info(f"BIC: {self.bic:.2f}")
        
        return {
            'params': self.params,
            'converged': self.converged,
            'log_likelihood': self.log_likelihood,
            'aic': self.aic,
            'bic': self.bic
        }
    
    def _calculate_conditional_volatility(self, returns: np.ndarray):
        """
        Calculate conditional volatility from fitted model
        
        Parameters:
        -----------
        returns : np.ndarray
            Daily returns
        """
        mu = self.params[0]
        omega = self.params[1]
        alpha = self.params[2:2+self.p]
        beta = self.params[2+self.p:2+self.p+self.q]
        
        epsilon = returns - mu
        n_obs = len(returns)
        h = np.zeros(n_obs)
        h[0] = np.var(epsilon)
        
        for t in range(1, n_obs):
            h[t] = omega
            for i in range(self.p):
                if t - i - 1 >= 0:
                    h[t] += alpha[i] * (epsilon[t - i - 1] ** 2)
            for j in range(self.q):
                if t - j - 1 >= 0:
                    h[t] += beta[j] * h[t - j - 1]
        
        self.conditional_volatility = np.sqrt(h)
        self.residuals = epsilon
    
    def forecast_volatility(self, returns: np.ndarray, steps: int = 20) -> np.ndarray:
        """
        Forecast future volatility
        
        Parameters:
        -----------
        returns : np.ndarray
            Historical daily returns
        steps : int
            Number of steps to forecast (default: 20 trading days)
            
        Returns:
        --------
        np.ndarray : Forecasted conditional volatility
        """
        if self.params is None:
            raise ValueError("Model must be fitted first")
        
        mu = self.params[0]
        omega = self.params[1]
        alpha = self.params[2:2+self.p]
        beta = self.params[2+self.p:2+self.p+self.q]
        
        # Use last observations
        epsilon = returns - mu
        n_obs = len(returns)
        
        # Initialize with final conditional variance
        h_last = np.zeros(max(self.p, self.q))
        
        # Fill with recent values
        last_eps_sq = epsilon[-self.p:][::-1] if self.p > 0 else np.array([])
        last_h = self.conditional_volatility[-self.q:][::-1] if self.q > 0 else np.array([])
        
        # Forecast
        forecast = np.zeros(steps)
        h = np.var(epsilon)
        
        for step in range(steps):
            h = omega
            
            for i in range(self.p):
                if i < len(last_eps_sq):
                    h += alpha[i] * (last_eps_sq[i] ** 2)
            
            for j in range(self.q):
                if j < len(last_h):
                    h += beta[j] * last_h[j]
                else:
                    h += beta[j] * h
            
            forecast[step] = np.sqrt(h)
            last_eps_sq = np.append([0], last_eps_sq[:-1]) if len(last_eps_sq) > 0 else np.array([])
            last_h = np.append([h], last_h[:-1]) if len(last_h) > 0 else np.array([h])
        
        return forecast
    
    def get_parameters_dict(self) -> Dict:
        """Get fitted parameters as dictionary"""
        if self.params is None:
            return None
        
        mu = self.params[0]
        omega = self.params[1]
        alpha = self.params[2:2+self.p]
        beta = self.params[2+self.p:2+self.p+self.q]
        
        return {
            'mu': mu,
            'omega': omega,
            'alpha': alpha,
            'beta': beta
        }


class ModelSelector:
    """Select best GARCH model using AIC criterion"""
    
    @staticmethod
    def select_best_model(
        returns: np.ndarray,
        max_p: int = 3,
        max_q: int = 3,
        criterion: str = 'aic'
    ) -> Tuple[Tuple[int, int], GARCHModel]:
        """
        Select best GARCH(p,q) model
        
        Parameters:
        -----------
        returns : np.ndarray
            Daily returns
        max_p : int
            Maximum ARCH order
        max_q : int
            Maximum GARCH order
        criterion : str
            'aic' or 'bic' for model selection
            
        Returns:
        --------
        tuple : (p, q) and fitted model
        """
        results = []
        
        logger.info(f"Selecting best GARCH model (max p={max_p}, max q={max_q})...")
        
        for p in range(1, max_p + 1):
            for q in range(1, max_q + 1):
                try:
                    model = GARCHModel(p, q)
                    model.fit(returns, verbose=False)
                    
                    if model.converged:
                        crit_value = model.aic if criterion == 'aic' else model.bic
                        results.append({
                            'p': p,
                            'q': q,
                            'criterion': crit_value,
                            'aic': model.aic,
                            'bic': model.bic,
                            'model': model
                        })
                        logger.info(f"GARCH({p},{q}): {criterion.upper()}={crit_value:.2f}")
                except:
                    logger.warning(f"GARCH({p},{q}) fitting failed")
        
        if len(results) == 0:
            raise ValueError("No models converged")
        
        # Select best
        best = min(results, key=lambda x: x['criterion'])
        logger.info(f"Best model: GARCH({best['p']},{best['q']}) with {criterion.upper()}={best['criterion']:.2f}")
        
        return (best['p'], best['q']), best['model']

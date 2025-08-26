"""
Monte-carlo pricer for European options.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

class MonteCarloPricing():
    """
    Monte Carlo pricer for European options.

    This class uses Monte Carlo simulations to estimate the price and Greeks 
    of European call or put options.

    Parameters
    ----------
    payoff : str
        Option type, either 'call' or 'put'.
    S0 : float
        Initial underlying asset price.
    K : float
        Strike price of the option.
    T : float
        Time to maturity (in years).
    r : float
        Risk-free interest rate (annualized).
    sigma : float
        Volatility of the underlying asset (annualized).
    N : int
        Number of Monte Carlo simulations.

    Methods
    -------
    get_option_price()
        Estimate the option price using Monte Carlo simulation.
    delta()
        Estimate the option Delta using finite differences.
    theta()
        Estimate the option Theta using finite differences.
    gamma()
        Estimate the option Gamma using finite differences.
    vega()
        Estimate the option Vega using finite differences.
    rho()
        Estimate the option Rho using finite differences.
    greeks()
        Compute all the Greeks and return them in a dictionary.
    plot_greek(greek, diff)
        Plot the evolution of a Greek with respect to the underlying asset price.
    """
    def __init__(self, payoff, S0, K, T, r, sigma, N):
        self.payoff = payoff
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.N = N

    def get_option_price(self):
        """
        Estimate the option price using Monte Carlo simulation.

        Returns
        -------
        tuple of float
            Estimated option price and standard error.

        Raises
        ------
        ValueError
            If payoff type is not 'call' or 'put'.
        """
        np.random.seed(0)
        W = scipy.stats.norm.rvs(loc=(self.r - (self.sigma**2/2))*self.T, scale=np.sqrt(self.T)*self.sigma, size=self.N)
        S_T = self.S0 * np.exp(W)

        if self.payoff == 'call':
            return (np.mean(np.exp(-self.r * self.T) * np.maximum(S_T - self.K, 0)), 
                    scipy.stats.sem(np.exp(-self.r * self.T) * np.maximum(S_T - self.K, 0)))

        elif self.payoff == 'put':
            return (np.mean(np.exp(-self.r * self.T) * np.maximum(self.K - S_T, 0)),
                    scipy.stats.sem(np.exp(-self.r * self.T) * np.maximum(self.K - S_T, 0)))

        else:
            raise ValueError("Invalid payoff type. Set 'call' or 'put'.")

    def delta(self):
        """
        Estimate the option Delta using finite differences.

        Returns
        -------
        float
            Sensitivity of the option price with respect to the underlying asset price.
        """
        initial_S0 = self.S0
        dx = 0.01 * initial_S0

        V = self.get_option_price()[0]
        self.S0 = initial_S0 + dx
        V_prime = self.get_option_price()[0]

        self.S0 = initial_S0
        return (V_prime - V) / dx

    def theta(self):
        """
        Estimate the option Theta using finite differences.

        Returns
        -------
        float
            Sensitivity of the option price with respect to the passage of time.

        Raises
        ------
        ValueError
            If time step reduction results in negative maturity.
        """
        initial_T = self.T
        dt = 0.01 * initial_T

        V = self.get_option_price()[0]

        self.T = initial_T - dt
        if self.T <= 0:
            raise ValueError("T - dt must remain positive")
        V_prime = self.get_option_price()[0]

        self.T = initial_T
        return (V_prime - V) / dt

    def gamma(self):
        """
        Estimate the option Gamma using finite differences.

        Returns
        -------
        float
            Sensitivity of Delta with respect to the underlying asset price.
        """
        initial_S0 = self.S0
        h = 0.01 * initial_S0

        V = self.get_option_price()[0]

        self.S0 = initial_S0 + h
        V_plus = self.get_option_price()[0]

        self.S0 = initial_S0 - h
        V_minus = self.get_option_price()[0]

        self.S0 = initial_S0
        return (V_plus - 2 * V + V_minus) / (h ** 2)

    def vega(self):
        """
        Estimate the option Vega using finite differences.

        Returns
        -------
        float
            Sensitivity of the option price with respect to volatility.
        """
        initial_sigma = self.sigma
        dx = 0.01 * initial_sigma

        V = self.get_option_price()[0]
        self.sigma = initial_sigma + dx
        V_prime = self.get_option_price()[0]

        self.sigma = initial_sigma
        return (V_prime - V) / dx

    def rho(self):
        """
        Estimate the option Rho using finite differences.

        Returns
        -------
        float
            Sensitivity of the option price with respect to the risk-free rate.
        """
        initial_r = self.r
        dx = 0.01 * initial_r

        V = self.get_option_price()[0]
        self.r = initial_r + dx
        V_prime = self.get_option_price()[0]

        self.r = initial_r 
        return (V_prime - V) / dx

    def greeks(self):
        """
        Compute all the Greeks and return them in a dictionary.

        Returns
        -------
        dict
            Dictionary containing 'delta', 'theta', 'gamma', 'vega', and 'rho'.
        """
        delta = self.delta()
        theta = self.theta()
        gamma = self.gamma()
        vega = self.vega()
        rho = self.rho()
        greeks = {
            'delta': delta,
            'theta': theta,
            'gamma': gamma,
            'vega': vega,
            'rho': rho
        }
        return greeks

    def plot_greek(self, greek, diff):
        """
        Plot the evolution of a Greek with respect to the underlying asset price.

        Parameters
        ----------
        greek : str
            Name of the Greek to plot ('delta', 'theta', 'gamma', 'vega', 'rho').
        diff : float
            Range of variation around the initial asset price S0.

        Raises
        ------
        ValueError
            If the provided Greek name is invalid.
        """
        try:
            s0values = np.linspace(self.S0 - diff, self.S0 + diff, 100)

            plt.plot(s0values, [getattr(MonteCarloPricing(self.payoff, s, self.K, self.T, self.r, self.sigma,N=10000), greek)() for s in s0values])
            plt.xlabel('Underlying asset price')
            plt.ylabel(greek)
            plt.title(f'{greek} for a European {self.payoff} option with maturity {self.T}')
            plt.show()

        except Exception as exc:
            raise ValueError(f"Invalid Greek name: '{greek}'. Expected one of: 'Delta', 'Theta', 'Gamma', 'Vega', 'Rho'.") from exc

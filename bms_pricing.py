"""
Black-Merton-Scholes pricer for European options.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

class BMSPricer():
    """
    Black-Merton-Scholes pricer for European options.

    This class allows to compute the price and Greeks of a European call or put option
    using the Black-Merton-Scholes model.

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

    Methods
    -------
    get_option_price()
        Compute the option price using the Black-Scholes formula.
    delta()
        Compute the option Delta.
    theta()
        Compute the option Theta.
    gamma()
        Compute the option Gamma.
    vega()
        Compute the option Vega.
    rho()
        Compute the option Rho.
    greeks()
        Compute all the Greeks and return them in a dictionary.
    plot_greeks(greek, diff)
        Plot the evolution of a Greek with respect to the underlying asset price.
    """
    def __init__(self, payoff, S0, K, T, r, sigma):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.payoff = payoff

    def _d1(self):
        """
        Compute the d1 term of the Black-Scholes formula.

        Returns
        -------
        float
            The value of d1.
        """
        return (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))

    def _d2(self):
        """
        Compute the d2 term of the Black-Scholes formula.

        Returns
        -------
        float
            The value of d2.
        """
        return self._d1() - self.sigma * np.sqrt(self.T)

    def get_option_price(self):
        """
        Compute the option price using the Black-Scholes formula.

        Returns
        -------
        float
            Price of the European option.

        Raises
        ------
        ValueError
            If payoff type is not 'call' or 'put'.
        """
        if self.payoff == 'call':
            return self.S0 * scipy.stats.norm.cdf(self._d1()) - self.K * np.exp(-self.r * self.T) * scipy.stats.norm.cdf(self._d2())

        if self.payoff == 'put':
            return self.K * np.exp(-self.r * self.T) * scipy.stats.norm.cdf(-self._d2()) - self.S0 * scipy.stats.norm.cdf(-self._d1())

        else:
            raise ValueError("Invalid payoff type. Set 'call' or 'put'.")

    def delta(self):
        """
        Compute the option Delta.

        Returns
        -------
        float
            Sensitivity of the option price with respect to the underlying asset price.
        """
        if self.payoff == 'call':
            return scipy.stats.norm.cdf(self._d1())

        if self.payoff == 'put':
            return (- scipy.stats.norm.cdf(-self._d1()))

        else:
            raise ValueError("Invalid payoff type. Set 'call' or 'put'.")

    def theta(self):
        """
        Compute the option Theta.

        Returns
        -------
        float
            Sensitivity of the option price with respect to the passage of time.
        """
        if self.payoff == 'call':
            return (- ((self.S0 * scipy.stats.norm.pdf(self._d1()) * self.sigma) / (2*np.sqrt(self.T))) - self.r * self.K * np.exp(-self.r * self.T) * scipy.stats.norm.cdf(self._d2()))

        if self.payoff == 'put':
            return (- ((self.S0 * scipy.stats.norm.pdf(self._d1()) * self.sigma) / (2*np.sqrt(self.T))) + self.r * self.K * np.exp(-self.r * self.T) * scipy.stats.norm.cdf(-self._d2()))

        else:
            raise ValueError("Invalid payoff type. Set 'call' or 'put'.")

    def gamma(self):
        """
        Compute the option Gamma.

        Returns
        -------
        float
            Sensitivity of Delta with respect to the underlying asset price.
        """
        return (scipy.stats.norm.pdf(self._d1()) / (self.S0 * self.sigma * np.sqrt(self.T)))

    def vega(self):
        """
        Compute the option Vega.

        Returns
        -------
        float
            Sensitivity of the option price with respect to volatility.
        """
        return self.S0 * scipy.stats.norm.pdf(self._d1()) * np.sqrt(self.T)

    def rho(self):
        """
        Compute the option Rho.

        Returns
        -------
        float
            Sensitivity of the option price with respect to the risk-free rate.

        Raises
        ------
        ValueError
            If payoff type is not 'call' or 'put'.
        """
        if self.payoff == 'call':
            return self.K * self.T * np.exp(-self.r*self.T) * scipy.stats.norm.cdf(self._d2())

        if self.payoff == 'put':
            return - self.K * self.T * np.exp(-self.r*self.T) * scipy.stats.norm.cdf(-self._d2())

        else:
            raise ValueError("Invalid payoff type. Set 'call' or 'put'.")

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

    def plot_greeks(self, greek, diff):
        '''
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
        '''
        greek = greek.lower()
        try:
            s0values = np.linspace(self.S0 - diff, self.S0 + diff, 100)

            plt.plot(s0values, [getattr(BMSPricer(self.payoff, s, self.K, self.T, self.r, self.sigma), greek)() for s in s0values])
            plt.xlabel('Underlying asset price')
            plt.ylabel(greek)
            plt.title(f'{greek} for a European {self.payoff} option with maturity {self.T}')
            plt.show()

        except Exception as exc:
            raise ValueError(f"Invalid Greek name: '{greek}'. Expected one of: 'delta', 'theta', 'gamma', 'vega', 'rho'.") from exc

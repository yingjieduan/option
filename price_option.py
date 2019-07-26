__author__ = 'Yingjie'

from math import log, sqrt, exp
from scipy import stats
import numpy.random as random
import numpy as np
import functools

class CallOption(object):

    def __init__(self, s0, k, t, r, sigma, timeStep = 0.01):
        self._s0 = s0        # ini price
        self._k = k          # strike price
        self._t = t          # life time
        self._r = r          # risk free rate
        self._sigma = sigma  # volatility
        self._timeStep = timeStep


    def bsmPrice(self, s0, k, t, r, sigma):
        ''' Valuation of European call option in BSM model.
        Analytical formula.
        Parameters
        ==========
        S0 : float
        initial stock/index level
        K : float
        strike price
        T : float
        maturity date (in year fractions)
        r : float
        constant risk-free short rate
        sigma : float
        volatility factor in diffusion term
        Returns
        =======
        value : float
        present value of the Digital call option
        '''
        if s0 is None :
            s0 = self._s0
        if k is None :
            k = self._k
        if t is None :
            t = self._t
        if r is None :
            r = self._r
        if sigma is None :
            sigma = self._sigma

        s0 = float(s0)
        d2 = (log(s0 / k) + (r - 0.5 * sigma ** 2) * t) / (sigma * sqrt(t))
        self.value = exp(-r * t) * stats.norm.cdf(d2, 0.0, 1.0)
        return self.value

    def riskNeutralPrice(self, s0, k, t, r, sigma, stepNo, pathNo):
        if s0 is None :
            s0 = self._s0
        if k is None :
            k = self._k
        if t is None :
            t = self._t
        if r is None :
            r = self._r
        if sigma is None :
            sigma = self._sigma
        if stepNo is None:
            stepNo = 250
        self._timeStep = t / stepNo
        if pathNo is None:
            pathNo = 1000

        #randomMatrix = np.zeros(shape = ( pathNo, stepNo))
        payOffT = np.zeros(pathNo)

        for i in range(0, pathNo):
            random_arr = random.standard_normal(stepNo)
            sT = functools.reduce(self._sForStep, random_arr, self._s0)
            payOffT[i] = 1 if(sT > self._k) else 0

        avePayOff = payOffT.sum()/pathNo
        self.riskNeutralValue = avePayOff * exp(-self._r * self._t)
        return self.riskNeutralValue


    def _sForStep(self, s, randomNo):
        return s*(1 + self._r * self._timeStep + self._sigma * sqrt(self._timeStep) * randomNo)


    def __str__(self):
        value = str(self.value) if (self.value is not None) else 'NaN'
        return "s0- " + self._s0 \
        + "\n k- " + self._k \
        + "\n t- " + self._t \
        + "\n r- " + self._r \
        + "\n sigma- " + self._sigma \
        + "\n Option Value-" + value

    def __repr__(self):
        value = str(self.value) if (self.value is not None) else 'NaN'
        return "s0- " + self._s0 \
        + "\n k- " + self._k \
        + "\n t- " + self._t \
        + "\n r- " + self._r \
        + "\n sigma- " + self._sigma \
        + "\n Option Value-" + value
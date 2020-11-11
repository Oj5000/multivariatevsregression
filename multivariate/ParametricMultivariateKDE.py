from multivariate.MultivariateBase import MultivariateBase

from scipy.stats import multivariate_normal
import numpy as np

class ParametricMultivariateKDE (MultivariateBase):

    def __init__(self):
        MultivariateBase.__init__(self)

        self.type = 'Parametric Multivariate KDE'
        self.chartName = 'parametric_multivariate_ranked'

    def fitpredict(self, data):
        sample_mean = np.mean(data).values
        cov_m = np.cov(np.transpose(data))
        x = data.values

        predictions = multivariate_normal.pdf(x, mean=sample_mean, cov=cov_m, allow_singular=True)

        return predictions
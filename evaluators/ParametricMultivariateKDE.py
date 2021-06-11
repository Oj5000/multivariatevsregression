from evaluators.EvaluatorBase import EvaluatorBase

from scipy.stats import multivariate_normal
import numpy as np

class ParametricMultivariateKDE (EvaluatorBase):

    def __init__(self):
        EvaluatorBase.__init__(self)

        self.type = 'Parametric Multivariate KDE'
        self.chartName = 'parametric_multivariate'
        self.marker = 's'

    def fitpredict(self, data, target=None):
        sample_mean = np.mean(data).values
        cov_m = np.cov(np.transpose(data))
        x = data.values

        predictions = multivariate_normal.pdf(x, mean=sample_mean, cov=cov_m, allow_singular=True)

        return predictions
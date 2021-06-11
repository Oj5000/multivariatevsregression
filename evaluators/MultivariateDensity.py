import numpy as np
from evaluators.EvaluatorBase import EvaluatorBase

class MultivariateDensity(EvaluatorBase):

    def __init__(self):
        EvaluatorBase.__init__(self)

        self.type = 'Multivariate Density'
        self.chartName = 'multivariate_density'
        self.marker = '+'

    def fitpredict(self, data, target=None):
        X = data.to_numpy()

        lHood = np.zeros((len(data),1))
        C = X.T@data
        detC = np.linalg.det(C)
        Cinv = np.linalg.inv(C)
        den = ((2*np.pi)**(len(data.columns)/2)) * (detC**(0.5))

        for n in range(len(data)):
            x = X[n,:].T
            lHood[n,:] = (1/den)*np.exp(-0.5*(x.T@Cinv@x))

        return lHood
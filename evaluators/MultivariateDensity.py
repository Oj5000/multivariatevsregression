import numpy as np
import pandas as pd
from evaluators.EvaluatorBase import EvaluatorBase

class MultivariateDensity(EvaluatorBase):

    def __init__(self):
        EvaluatorBase.__init__(self)

        self.type = 'Multivariate Density'
        self.chartName = 'multivariate_density'
        self.marker = '+'

        self.auc = []
        self.all_tp = []

    def fitpredict(self, data):
        X = data.to_numpy()

        lHood = np.zeros((len(data),1))
        C = X.T@X
        detC = np.linalg.det(C)
        Cinv = np.linalg.inv(C)
        den = ((2*np.pi)**(len(data.columns)/2)) * (detC**(0.5))

        for n in range(len(data)):
            x = X[n,:].T
            lHood[n,:] = np.log((1/den)*np.exp(-0.5*(x.T@Cinv@x)))

        return pd.DataFrame(lHood, columns=['0'])

    def addResults(self, auc, all_tp):
        self.auc.append(auc)

        if all_tp is not None:
            self.all_tp.append(all_tp)

    def get_auc_result(self):
        return "%.4f +- %.4f" % (np.mean(self.auc), np.std(self.auc))
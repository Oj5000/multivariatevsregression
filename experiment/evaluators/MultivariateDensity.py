import numpy as np
import pandas as pd
import math
from evaluators.EvaluatorBase import EvaluatorBase

class MultivariateDensity(EvaluatorBase):

    def __init__(self):
        EvaluatorBase.__init__(self)

        self.type = 'Multivariate Density'
        self.chartName = 'multivariate_density'
        self.marker = '+'

        self.auc = []
        self.all_tp = []

        self.results_all_data = {}
        self.results_blind = {}
        self.d_inv = 0
        self.Cinv = 0

    def predict(self, data):
        f = (self.get_lhood(x, self.d_inv, self.Cinv) for x in data.to_numpy())
        lHood = list(f)
        lHood = np.array(lHood).reshape(len(lHood), 1)

        return pd.DataFrame(lHood, columns=['0'], index=data.index)

    def fitpredict(self, data):
        X = data.to_numpy()

        lHood = np.zeros((len(data),1))
        C = X.T@X

        # check for inf
        if math.isinf(np.linalg.det(C)):
            sign, logdet = np.linalg.slogdet(C)
            detC = sign * np.exp(logdet)
        else:
            detC = np.linalg.det(C)

        Cinv = np.linalg.inv(C)
        den = ((2*np.pi)**(len(data.columns)/2)) * (detC**(0.5))
        d_inv = (1/den)
        #print("detC %s" % str(detC))

        f = (self.get_lhood(x, d_inv, Cinv) for x in X)
        lHood = list(f)
        lHood = np.array(lHood).reshape(len(lHood), 1)

        self.d_inv = d_inv
        self.Cinv = Cinv

        return pd.DataFrame(lHood, columns=['0'], index=data.index)

    def get_lhood(self, x, d_inv, Cinv):
        return np.log(d_inv * np.exp(-0.5*(x.T@Cinv@x)) )

    def addResults_all_data(self, mut_amp, auc, all_tp):
        
        if mut_amp in self.results_all_data:
            self.results_all_data[mut_amp]['auc'].append(auc)
            self.results_all_data[mut_amp]['all_tp'].append(all_tp)
        else:
            self.results_all_data[mut_amp] = {'auc': [auc], 'all_tp': [all_tp]}

    def get_auc_result_all_data(self, mut_amp):
        return "%.4f +- %.4f" % (np.mean(self.results_all_data[mut_amp]['auc']), np.std(self.results_all_data[mut_amp]['auc']))

    def addResults_blind(self, mut_amp, auc, all_tp):
        
        if mut_amp in self.results_blind:
            self.results_blind[mut_amp]['auc'].append(auc)
            self.results_blind[mut_amp]['all_tp'].append(all_tp)
        else:
            self.results_blind[mut_amp] = {'auc': [auc], 'all_tp': [all_tp]}

    def get_auc_result_blind(self, mut_amp):
        return "%.4f +- %.4f" % (np.mean(self.results_blind[mut_amp]['auc']), np.std(self.results_blind[mut_amp]['auc']))
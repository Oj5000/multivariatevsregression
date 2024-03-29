import numpy as np
import pandas as pd
from scipy import stats
import math

class MultivariateDensity():

    def __init__(self):
        self.type = 'Multivariate Density'
        self.chartName = 'multivariate_density'
        self.marker = '+'

        self.f1 = []
        self.all_tp = []

        self.results_all_data = {}
        self.results_blind = {}
        self.d_inv = 0
        self.Cinv = 0

        self.kde = None

    def predict(self, data):
        f = (self.get_lhood(x, self.d_inv, self.Cinv) for x in data.to_numpy())
        lHood = list(f)
        lHood = np.array(lHood).reshape(len(lHood), 1)

        return pd.DataFrame(lHood, columns=['0'], index=data.index)

    def fitpredict(self, data):
        try:
            X = data.to_numpy()

            lHood = np.zeros((len(data),1))
            C = X.T@X

            # check for inf - what do we do if this is 0?
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
        except:
            return self.fitpredict(data.sample(frac=1)) # If we run into a matrix problem, just reshuffle the rows
        
    def get_lhood(self, x, d_inv, Cinv):
        return np.log(d_inv * np.exp(-0.5*(x.T@Cinv@x)) )

    def addResults_all_data(self, mut_amp, f1, all_tp):
        
        if mut_amp in self.results_all_data:
            self.results_all_data[mut_amp]['f1'].append(f1)
            self.results_all_data[mut_amp]['all_tp'].append(all_tp)
        else:
            self.results_all_data[mut_amp] = {'f1': [f1], 'all_tp': [all_tp]}

    def addResults_blind(self, mut_amp, f1, all_tp):
        
        if mut_amp in self.results_blind:
            self.results_blind[mut_amp]['f1'].append(f1)
            self.results_blind[mut_amp]['all_tp'].append(all_tp)
        else:
            self.results_blind[mut_amp] = {'f1': [f1], 'all_tp': [all_tp]}

    def get_f1_result_all_data(self, mut_amp):
        return "%.4f +- %.4f" % (np.mean(self.results_all_data[mut_amp]['f1']), np.std(self.results_all_data[mut_amp]['f1']))

    def get_f1_result_blind(self, mut_amp):
        return "%.4f +- %.4f" % (np.mean(self.results_blind[mut_amp]['f1']), np.std(self.results_blind[mut_amp]['f1']))

    def get_all_tp_result_all_data(self, mut_amp):
        return "%.4f +- %.4f" % (np.mean(self.results_all_data[mut_amp]['all_tp']), np.std(self.results_all_data[mut_amp]['all_tp']))

    def get_all_tp_result_blind(self, mut_amp):
        return "%.4f +- %.4f" % (np.mean(self.results_blind[mut_amp]['all_tp']), np.std(self.results_blind[mut_amp]['all_tp']))
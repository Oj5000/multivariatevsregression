import numpy as np
import pandas as pd
from scipy import stats
import math

class MultivariateDensity2():

    def __init__(self):
        self.type = 'Multivariate Density2'
        self.chartName = 'multivariate_density2'
        self.marker = '+'

        self.f1 = []
        self.all_tp = []

        self.results_all_data = {}
        self.results_blind = {}
        self.d_inv = 0
        self.Cinv = 0

    def predict(self, data):
        return pd.DataFrame(self.kde(data.T), columns=['0'], index=data.index)

    def fitpredict(self, data):
        try:
            self.kde = stats.gaussian_kde(data.T)
            return self.predict(data)
        except:
            return self.fitpredict(data.sample(frac=1)) # If we run into a matrix problem, just reshuffle the rows

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
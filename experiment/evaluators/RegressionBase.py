import pandas as pd
import numpy as np

class RegressionBase():

    def __init__(self, columns, polynomials, regularisation_level):
        self.polynomials = polynomials
        self.regularisation_level = regularisation_level
        self.lr_models = {}

        self.type = 'Regression, polys=%i, norm=%i' % (polynomials, regularisation_level)
        self.chartName = 'linear_regression'
        self.marker = ['o', 'v', '^', '<', '>', 'x'][polynomials]

        self.columns = columns
        self.results_all_data = {}
        self.results_blind = {}

    def addResults_all_data(self, mut_amp, auc, all_tp):

        if mut_amp in self.results_all_data:
            auc_tmp = self.results_all_data[mut_amp]['auc']
            auc_tmp = auc_tmp.append(pd.DataFrame([auc], columns=self.columns))
            self.results_all_data[mut_amp]['auc'] = auc_tmp
            self.results_all_data[mut_amp]['all_tp'].append(all_tp)
        else:
            self.results_all_data[mut_amp] = {'auc': pd.DataFrame([auc], columns=self.columns), 'all_tp': [all_tp]}

    def get_auc_result_all_data(self, mut_amp):
        auc = self.results_all_data[mut_amp]['auc']

        best_auc = 0
        pm = 0
        col = ""

        for c in auc.columns:
            mu = np.mean(auc[c])
            std = np.std(auc[c])

            if mu > best_auc:
                best_auc = mu
                pm = std
                col = c

        return "%s: %.4f +- %.4f" % (col, mu, std)

    def addResults_blind(self, mut_amp, auc, all_tp):

        if mut_amp in self.results_blind:
            auc_tmp = self.results_blind[mut_amp]['auc']
            auc_tmp = auc_tmp.append(pd.DataFrame([auc], columns=self.columns))
            self.results_blind[mut_amp]['auc'] = auc_tmp
            self.results_blind[mut_amp]['all_tp'].append(all_tp)
        else:
            self.results_blind[mut_amp] = {'auc': pd.DataFrame([auc], columns=self.columns), 'all_tp': [all_tp]}

    def get_auc_result_blind(self, mut_amp):
        auc = self.results_blind[mut_amp]['auc']

        best_auc = 0
        pm = 0
        col = ""

        for c in auc.columns:
            mu = np.mean(auc[c])
            std = np.std(auc[c])

            if mu > best_auc:
                best_auc = mu
                pm = std
                col = c

        return "%s: %.4f +- %.4f" % (col, mu, std)
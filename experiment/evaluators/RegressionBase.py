import pandas as pd
import numpy as np

class RegressionBase():

    def __init__(self, columns, polynomials):
        self.polynomials = polynomials
        self.lr_models = {}

        self.type = 'Linear Regression, %i degrees' % polynomials
        self.chartName = 'linear_regression'
        self.marker = ['o', 'v', '^', '<', '>', 'x'][polynomials]

        self.columns = columns
        self.results = {}

    def addResults(self, mut_amp, auc, all_tp):

        if mut_amp in self.results:
            auc_tmp = self.results[mut_amp]['auc']
            auc_tmp = auc_tmp.append(pd.DataFrame([auc], columns=self.columns))
            self.results[mut_amp]['auc'] = auc_tmp
            self.results[mut_amp]['all_tp'].append(all_tp)
        else:
            self.results[mut_amp] = {'auc': pd.DataFrame([auc], columns=self.columns), 'all_tp': [all_tp]}

    def get_auc_result(self, mut_amp):
        auc = self.results[mut_amp]['auc']

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
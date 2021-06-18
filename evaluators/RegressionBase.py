import pandas as pd
import numpy as np

class RegressionBase():

    def __init__(self, columns, polynomials):
        self.polynomials = polynomials
        self.lr_models = {}

        self.type = 'Linear Regression, %i degrees' % polynomials
        self.chartName = 'linear_regression'
        self.marker = ['o', 'v', '^', '<', '>', 'x'][polynomials]

        self.auc = pd.DataFrame(columns=columns)
        self.all_tp = []

    def addResults(self, auc, all_tp):
        self.auc = self.auc.append(pd.DataFrame([auc], columns=self.auc.columns))

        if all_tp is not None:
            self.all_tp.append(all_tp)

    def get_auc_result(self):
        best_auc = 0
        pm = 0
        col = ""

        for c in self.auc.columns:
            mu = np.mean(self.auc[c])
            std = np.std(self.auc[c])

            if mu > best_auc:
                best_auc = mu
                pm = std
                col = c

        return "%s: %.4f +- %.4f" % (col, mu, std)
import pandas as pd
import numpy as np
import copy

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

    def addResults_all_data(self, mut_amp, f1, all_tp):

        if mut_amp in self.results_all_data:
            f1_tmp = copy.deepcopy(self.results_all_data[mut_amp]['f1'])
            f1_tmp = pd.concat([f1_tmp,pd.DataFrame([f1], columns=self.columns)])
            #f1_tmp = f1_tmp.append(pd.DataFrame([f1], columns=self.columns))
            self.results_all_data[mut_amp]['f1'] = f1_tmp
            #self.results_all_data[mut_amp]['all_tp'].append(all_tp)

            all_tp_tmp = copy.deepcopy(self.results_all_data[mut_amp]['all_tp'])
            all_tp_tmp = pd.concat([all_tp_tmp, all_tp])
            self.results_all_data[mut_amp]['all_tp'] = all_tp_tmp
        else:
            self.results_all_data[mut_amp] = {'f1': pd.DataFrame([f1], columns=self.columns), 'all_tp': all_tp}

    def addResults_blind(self, mut_amp, f1, all_tp):

        if mut_amp in self.results_blind:
            f1_tmp = copy.deepcopy(self.results_blind[mut_amp]['f1'])
            f1_tmp = pd.concat([f1_tmp, pd.DataFrame([f1], columns=self.columns)])
            #f1_tmp = f1_tmp.append(pd.DataFrame([f1], columns=self.columns))
            self.results_blind[mut_amp]['f1'] = f1_tmp

            all_tp_tmp = copy.deepcopy(self.results_blind[mut_amp]['all_tp'])
            all_tp_tmp = pd.concat([all_tp_tmp, all_tp])

            #self.results_blind[mut_amp]['all_tp'].append(all_tp)
            self.results_blind[mut_amp]['all_tp'] = all_tp_tmp
        else:
            self.results_blind[mut_amp] = {'f1': pd.DataFrame([f1], columns=self.columns), 'all_tp': all_tp}

    def get_f1_result_all_data(self, mut_amp):
        f1 = self.results_all_data[mut_amp]['f1']

        best_f1 = 0
        pm = 0
        col = ""

        for c in f1.columns:
            mu = np.mean(f1[c])
            std = np.std(f1[c])

            if mu > best_f1:
                best_f1 = mu
                pm = std
                col = c

        return "%s: %.4f +- %.4f" % (col, best_f1, pm)

    def get_f1_result_blind(self, mut_amp):
        f1 = self.results_blind[mut_amp]['f1']

        best_f1 = 0
        pm = 0
        col = ""

        for c in f1.columns:
            mu = np.mean(f1[c])
            std = np.std(f1[c])

            if mu > best_f1:
                best_f1 = mu
                pm = std
                col = c

        return "%s: %.4f +- %.4f" % (col, best_f1, pm)

    def get_all_tp_result_all_data(self, mut_amp):
        all_tp = self.results_all_data[mut_amp]['all_tp']

        best_all_tp = 0
        pm = 0
        col = ""

        for c in all_tp.columns:
            mu = np.mean(all_tp[c])
            std = np.std(all_tp[c])

            if mu > best_all_tp:
                best_all_tp = mu
                pm = std
                col = c

        return "%s: %.4f +- %.4f" % (col, best_all_tp, pm)

    def get_all_tp_result_blind(self, mut_amp):
        all_tp = self.results_blind[mut_amp]['all_tp']

        best_all_tp = 0
        pm = 0
        col = ""

        for c in all_tp.columns:
            mu = np.mean(all_tp[c])
            std = np.std(all_tp[c])

            if mu > best_all_tp:
                best_all_tp = mu
                pm = std
                col = c

        return "%s: %.4f +- %.4f" % (col, best_all_tp, pm)
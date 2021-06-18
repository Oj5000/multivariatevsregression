import pandas as pd
import numpy as np
import math

from sklearn import metrics

from evaluators.NonParametricMultivariateKDE import NonParametricMultivariateKDE
from evaluators.ParametricMultivariateKDE import ParametricMultivariateKDE

from evaluators.MultivariateDensity import MultivariateDensity
from evaluators.LinearEvaluator import LinearEvaluator

def evaluate(dataset, runs, mutation):
    dataset.preprocess()
    evaluators = [MultivariateDensity(), LinearEvaluator(dataset.columns, 1)]

    for run in range(0, runs):
        print("Run %s of %s" % (run+1, runs))

        sample_idx, data = dataset.mutate(mutation)

        for evaluator in evaluators:
            print("Running " + evaluator.type)
            errors = evaluator.fitpredict(data)

            auc, all_tp = get_results(errors, sample_idx)
            evaluator.addResults(auc, all_tp)

        dataset.restore()

    return evaluators

def get_results(errors, sample_idx):
    Npoints = 50
    thRange = np.linspace(np.min(errors), np.max(errors), Npoints)

    npos = len(sample_idx)
    nneg = len(errors) - npos

    all_tp = None

    if errors.shape[1] > 1:
        all_tp = np.zeros(errors.shape[1])

        for i in range(len(all_tp)):
            all_tp[i] = 1e9

        tpr = pd.DataFrame(columns=errors.columns)
        fpr = pd.DataFrame(columns=errors.columns)

        # Regression
        for i in range(Npoints):
            results = (errors >= thRange[i])

            tp_r = results.iloc[sample_idx].sum() / npos
            fp_r = results.iloc[~results.index.isin(sample_idx)].sum() / nneg

            tpr = tpr.append(pd.DataFrame([tp_r], columns=errors.columns))
            fpr = fpr.append(pd.DataFrame([fp_r], columns=errors.columns))

            for i in range(len(tp_r.values)):
                if tp_r.values[i] == 1:
                    fp = fp_r.values[i] * nneg
                    if fp < all_tp[i]:
                        all_tp[i] = fp_r.values[i] * nneg

        auc = []

        for i in range(len(errors.columns)):
            auc.append(metrics.auc(fpr.iloc[:,i].values, tpr.iloc[:,i].values))
    else:
        all_tp = 1e9
        tpr = []
        fpr = []

        # density
        for i in range(Npoints):
            results = (errors <= thRange[i])

            tp_r = (results.iloc[sample_idx].sum() / npos).values[0]
            fp_r = (results.iloc[~results.index.isin(sample_idx)].sum() / nneg).values[0]

            tpr.append(tp_r)
            fpr.append(fp_r)

            if tp_r == 1:
                fp = fp_r * nneg
                if fp < all_tp:
                    all_tp = fp

        auc = metrics.auc(fpr, tpr)

    return auc, all_tp
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
            predictions = evaluator.fitpredict(data)

            auc, all_tp = get_results(predictions, sample_idx)
            evaluator.addResults(auc, all_tp)

    return evaluators

def get_results(predictions, sample_idx):
    Npoints = 50
    thRange = np.linspace(np.min(predictions), np.max(predictions), Npoints)

    npos = len(sample_idx)
    nneg = len(predictions) - npos

    all_tp = None

    if predictions.shape[1] > 1:
        tpr = pd.DataFrame(columns=predictions.columns)
        fpr = pd.DataFrame(columns=predictions.columns)

        # Regression
        for i in range(Npoints):
            results = (predictions >= thRange[i])

            tp_r = results.iloc[sample_idx].sum() / npos
            fp_r = results.iloc[~results.index.isin(sample_idx)].sum() / nneg

            tpr = tpr.append(pd.DataFrame([tp_r], columns=predictions.columns))
            fpr = fpr.append(pd.DataFrame([fp_r], columns=predictions.columns))

            if all_tp is None:
                if tp_r.values[0] == 1:
                    all_tp = fp_r.values[0] * nneg

        auc = []

        for i in range(len(predictions.columns)):
            auc.append(metrics.auc(fpr.iloc[:,i].values, tpr.iloc[:,i].values))
    else:
        tpr = []
        fpr = []

        # density
        for i in range(Npoints):
            results = (predictions <= thRange[i])

            tp_r = (results.iloc[sample_idx].sum() / npos).values[0]
            fp_r = (results.iloc[~results.index.isin(sample_idx)].sum() / nneg).values[0]

            tpr.append(tp_r)
            fpr.append(fp_r)

            if all_tp is None:
                if tp_r == 1:
                    all_tp = fp_r * nneg

        auc = metrics.auc(fpr, tpr)

    return auc, all_tp
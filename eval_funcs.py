import pandas as pd
import numpy as np
import math

from sklearn import metrics

from evaluators.NonParametricMultivariateKDE import NonParametricMultivariateKDE
from evaluators.ParametricMultivariateKDE import ParametricMultivariateKDE

from evaluators.MultivariateDensity import MultivariateDensity
from evaluators.LinearEvaluator import LinearEvaluator

def evaluate(dataset, name, runs):
    dataset.preprocess()
    evaluators = [MultivariateDensity()] #LinearEvaluator(1),

    for run in range(0, runs):
        print("Run %s of %s" % (run+1, runs))

        sample_idx, data = dataset.mutate(0.2)

        for evaluator in evaluators:
            print("Running " + evaluator.type)
            predictions = evaluator.fitpredict(data)
            results = get_results(data, sample_idx)

    return evaluators

# Improve this by finding all thresholds, then doing an np.where - should only take 1 pass and be way faster
def get_results(data, sample_idx):
    Npoints = 50
    res = []
    thRange = np.linspace(np.min(data), np.max(data), Npoints)

    npos = len(sample_idx)
    nneg = len(data) - npos

    tpr = pd.DataFrame(columns=data.columns)
    fpr = pd.DataFrame(columns=data.columns)

    for i in range(Npoints):
        results = (data >= thRange[i])

        tpr = tpr.append(pd.DataFrame([results.iloc[sample_idx].sum() / npos], columns=data.columns))
        fpr = fpr.append(pd.DataFrame([results.iloc[~results.index.isin(sample_idx)].sum() / nneg], columns=data.columns))

    auc = []

    for i in range(len(data.columns)):
        auc.append(metrics.auc(fpr.iloc[:,i].values, tpr.iloc[:,i].values))

    return auc
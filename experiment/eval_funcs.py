import pandas as pd
import numpy as np
import math

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)

from sklearn import metrics

from evaluators.MultivariateDensity2 import MultivariateDensity2
from evaluators.MultivariateDensity import MultivariateDensity
from evaluators.LinearEvaluator import LinearEvaluator

def evaluate(dataset, runs, mutation, mutation_min, mutation_max):
    evaluators = [
        MultivariateDensity2(),
        LinearEvaluator(dataset.columns, 1, 0),
        LinearEvaluator(dataset.columns, 1, 1),
        LinearEvaluator(dataset.columns, 1, 2),
    ]

    print("Mutation percentage: %s" % mutation)

    for mutation_amplitude in np.linspace(mutation_min, mutation_max, 10):
        print("Mutation amplitude: %s" % mutation_amplitude)
        for run in range(runs):
            print("   Run %s of %s" % (run+1, runs))

            sample_idx, data = dataset.mutate(mutation, mutation_amplitude)

            for evaluator in evaluators:
                #print("      Running " + evaluator.type)

                # All results
                errors = evaluator.fitpredict(data)
                f1, all_tp = get_results(errors, sample_idx)
                #print("Results mutation_amplitude: %s, f1: %s, all_tp: %s" % (str(mutation_amplitude), str(f1.max()), str(all_tp.max())))
                evaluator.addResults_all_data(mutation_amplitude, f1, all_tp)

                # Blind
                # split data into train and test
                raw_size = len(data) - len(sample_idx)
                training = data[~data.index.isin(sample_idx)].sample(raw_size-len(sample_idx))
                testing  = data[~data.index.isin(training.index)]

                errors_gen = evaluator.fitpredict(training)
                errors = evaluator.predict(testing)

                f1, all_tp = get_results(errors, sample_idx)
                #print("Results mutation_amplitude: %s, f1: %s, all_tp: %s" % (str(mutation_amplitude), str(f1.max()), str(all_tp.max())))
                evaluator.addResults_blind(mutation_amplitude, f1, all_tp)

            dataset.restore()

    return evaluators

def get_results(errors, sample_idx):
    Npoints = 100

    npos = len(sample_idx)

    if errors.shape[1] > 1:
        thRange = np.linspace(np.min(np.abs(errors),axis=0), np.max(np.abs(errors),axis=0), Npoints)
        all_tp = pd.DataFrame([np.full(len(errors.columns),1e9)], columns=errors.columns)
        f1 = pd.DataFrame(columns=errors.columns)

        # Regression
        for i in range(Npoints):
            results = (np.abs(errors) >= thRange[i])

            tp = results.loc[sample_idx].sum()
            fp = np.sum(results) - tp
            fn = npos - tp

            f = get_f1(tp, fp, fn)
            f1 = pd.concat([f1, pd.DataFrame([f], columns=errors.columns)], ignore_index=True)

            for c in errors.columns:
                if fp[c] < all_tp[c].values[0] and tp[c] == len(sample_idx):
                    all_tp[c].iloc[0] = fp[c]

        best_f = f1.max()
    else:
        thRange = np.linspace(np.min(errors,axis=0), np.max(errors,axis=0), Npoints)

        # density
        best_f = 0
        all_tp = 1e9

        for i in range(Npoints):
            results = (errors <= thRange[i]) # High likelihood = Highly likely to be part of the training data & NOT an outlier = True negative

            tp = results.loc[sample_idx].sum().values[0]
            fp = np.sum(results).values[0] - tp
            fn = npos - tp

            f1 = get_f1(tp, fp, fn)

            if f1 > best_f:
                best_f = f1

            if fp < all_tp and tp == len(sample_idx):
                all_tp = fp

    return best_f, all_tp

def get_f1(tp, fp, fn):
    return tp / (tp + 0.5*(fp + fn))
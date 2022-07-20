import pandas as pd
import numpy as np
import math

from sklearn import metrics

from evaluators.MultivariateDensity import MultivariateDensity
from evaluators.LinearEvaluator import LinearEvaluator

def evaluate(dataset, runs, mutation, mutation_min, mutation_max):
    evaluators = [
        MultivariateDensity(),
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
                print("      Running " + evaluator.type)

                # All results
                errors = evaluator.fitpredict(data)
                auc, all_tp = get_results(errors, sample_idx)
                print("Results mutation_amplitude: %s, auc: %s, all_tp: %s" % (str(mutation_amplitude), str(np.max(auc)), str(np.max(all_tp))))
                evaluator.addResults_all_data(mutation_amplitude, auc, all_tp)

                # Blind
                # split data into train and test
                raw_size = len(data) - len(sample_idx)
                training = data[~data.index.isin(sample_idx)].sample(raw_size-len(sample_idx))
                testing  = data[~data.index.isin(training.index)]

                errors_gen = evaluator.fitpredict(training)
                errors = evaluator.predict(testing)

                auc, all_tp = get_results(errors, sample_idx)
                print("Results mutation_amplitude: %s, auc: %s, all_tp: %s" % (str(mutation_amplitude), str(np.max(auc)), str(np.max(all_tp))))
                evaluator.addResults_blind(mutation_amplitude, auc, all_tp)

            dataset.restore()

    return evaluators

def get_results(errors, sample_idx):
    Npoints = 100
    thRange = np.linspace(np.min(np.abs(errors)), np.max(np.abs(errors)), Npoints)

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
            results = (np.abs(errors) >= thRange[i])

            tp = results.loc[sample_idx].sum()
            fp = results.sum() - tp

            tp_r = tp / npos
            fp_r = fp / nneg

            #tpr = tpr.append(pd.DataFrame([tp_r], columns=errors.columns))
            #fpr = fpr.append(pd.DataFrame([fp_r], columns=errors.columns))
            tpr = pd.concat([tpr, pd.DataFrame([tp_r], columns=errors.columns)], ignore_index=True)
            fpr = pd.concat([fpr, pd.DataFrame([fp_r], columns=errors.columns)], ignore_index=True)

            for i in range(len(tp_r.values)):
                if tp_r.values[i] == 1:
                    if fp.values[i] < all_tp[i]:
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
            results = (np.abs(errors) >= thRange[i])

            tp = results.loc[sample_idx].sum().values[0]
            fp = results.sum().values[0] - tp

            tp_r = (tp / npos)
            fp_r = (fp / nneg)

            if tp_r == 1:
                if fp < all_tp:
                    all_tp = fp

            tpr.append(tp_r)
            fpr.append(fp_r)

        auc = metrics.auc(fpr, tpr)

    return auc, all_tp
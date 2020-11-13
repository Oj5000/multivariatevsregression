import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

from evaluators.NonParametricMultivariateKDE import NonParametricMultivariateKDE
from evaluators.ParametricMultivariateKDE import ParametricMultivariateKDE
from evaluators.LinearEvaluator import LinearEvaluator

def preprocess(data, columns):
    # Simple pre-processing
    columns2 = columns.copy()
    
    # del singular valued cols
    del_fields = []

    for col in columns:
        if len(pd.unique(data[col])) == 1:
            del_fields.append(col)

    if len(del_fields) > 0:
        print("Removing fields %s - values all the same" % del_fields)

    data2 = data.drop(columns = del_fields)

    for d in del_fields:
        columns2.remove(d)

    # We need to impute values for the N/A's - use median value
    for col in columns2:
        i_val = np.median(data2[col].dropna())
        data2[col].fillna(i_val, inplace=True)

    # check again for singular valued cols
    del_fields = []

    for col in columns2:
        if len(pd.unique(data2[col])) == 1:
            del_fields.append(col)

    if len(del_fields) > 0:
        print("Removing fields %s - values all the same" % del_fields)

    data2 = data2.drop(columns = del_fields)

    for d in del_fields:
        columns2.remove(d)

    return data2, columns2

def evaluate(data, columns, name, runs):
    data, columns2 = preprocess(data, columns)

    target = data[data['class'] == 1.0]
    evaluators = [ParametricMultivariateKDE(), NonParametricMultivariateKDE(), LinearEvaluator(1)]

    for run in range(0, runs):
        # design training data using a random sample of outliers - use same data for each evaluator
        data2 = data[data['class'] == 0.0]
        target_sample = target.sample(n= int(len(target) * 0.75), replace=False, random_state=np.random.RandomState(seed=None))
        data2 = data2.append(target_sample)

        for evaluator in evaluators:
            predictions = evaluator.fitpredict(data2[columns2], data2['class'])

            res = pd.DataFrame.from_dict({"Density": predictions, 'class' : data2['class']})
            res.set_index(data2.index, inplace=True)

            results = {}
            found = False
            for percentile in range(0, 101, 1):
                perc = np.percentile(res['Density'], percentile)
                idxs = res[res['Density'] <= perc].index

                tp = sum(data2['class'].loc[idxs])
                fp = len(idxs) - tp
                fn = sum(data2['class']) - tp

                if tp == len(target_sample):
                    if found == False:
                        found = True
                        evaluator.fps.append(fp)
                
                if percentile == 5:
                    evaluator.fith_p_tp.append(tp)
                    evaluator.fith_p_fp.append(fp)
                    evaluator.fith_p_fn.append(fn)

                    f1 = tp/(tp + ((fp+fn)*0.5))

                    if f1 >= evaluator.best_f1:
                        evaluator.best_f1 = f1
                        evaluator.best_results = res

    # Create results
    for evaluator in evaluators:
        print("    Plotting results and saving as %s_%s.pdf" % (name,evaluator.chartName))

        fig = sns.displot(evaluator.best_results, bins=10, x='Density', hue='class', element="step", fill=True)
        plt.title("%s: %s PDF" % (name, evaluator.type))
        fig.savefig("Results/" + name+"_%s.pdf" % evaluator.chartName)
        plt.close()

        print("    TP: " + str(np.mean(evaluator.fith_p_tp)) + "+-" + str(np.std(evaluator.fith_p_tp)) + " FP: " + str(np.mean(evaluator.fith_p_fp)) + "+-" + str(np.std(evaluator.fith_p_fp)) + " FN: " + str(np.mean(evaluator.fith_p_fn)) + "+-" + str(np.std(evaluator.fith_p_fn)))
        print("    Mean fp for all tp: " + str(np.mean(evaluator.fps)) + " +- " + str(np.std(evaluator.fps)))

        evaluator.setResults({'tp' : "%.2f +- %.2f" % (np.mean(evaluator.fith_p_tp), np.std(evaluator.fith_p_tp)), 'fp' : "%.2f +- %.2f" % (np.mean(evaluator.fith_p_fp), np.std(evaluator.fith_p_fp)), 'fn' : "%.2f +- %.2f" % (np.mean(evaluator.fith_p_fn), np.std(evaluator.fith_p_fn)) },
                             "%.2f +- %.2f" % (np.mean(evaluator.fps), np.std(evaluator.fps)))

    return evaluators
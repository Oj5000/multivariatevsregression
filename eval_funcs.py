import pandas as pd
import numpy as np
from numpy import inf
import math
from time import strptime
import matplotlib.pyplot as plt
from pandas import DataFrame as df
from matplotlib.backends.backend_pdf import PdfPages
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import copy
import seaborn as sns

def preprocess(data, columns):
    # Simple pre-processing
    columns2 = columns.copy()
    
    # del singular valued cols
    del_fields = []

    for col in columns:
        if len(pd.unique(data[col])) == 1:
            print("Removing field %s - values all the same" % col)
            del_fields.append(col)

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
            print("Removing field %s - values all the same" % col)
            del_fields.append(col)

    data2 = data2.drop(columns = del_fields)

    for d in del_fields:
        columns2.remove(d)

    return data2, columns2

def eval_multivariate(data, columns, name, runs):
    print("Multivariate analysis")

    data, columns2 = preprocess(data, columns)

    target = data[data['class'] == 1.0]
    fps = []

    fith_p_tp = []
    fith_p_fp = []
    fith_p_fn = []

    best_results = None
    best_f1 = 0

    for run in range(0, runs):
        # design training data using a random sample of outliers
        data2 = data[data['class'] == 0.0]
        target_sample = target.sample(n= int(len(target) * 0.75), replace=False, random_state=np.random.RandomState(seed=None))
        data2 = data2.append(target_sample)

        v_type = ""
        for i in range(0, len(columns2)):
            v_type += 'c'
        try:
            dens_u = sm.nonparametric.KDEMultivariate(data=data2[columns2], var_type=v_type, bw='normal_reference')
            predictions = dens_u.pdf(data2[columns2])
        except:
            print("statsmodels multivariate did not work, using method 2")
            predictions = multivariate_2(data2, columns2)    

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
                    fps.append(fp)
            
            if percentile == 5:
                fith_p_tp.append(tp)
                fith_p_fp.append(fp)
                fith_p_fn.append(fn)

                f1 = tp/(tp + ((fp+fn)*0.5))

                if f1 >= best_f1:
                    best_f1 = f1
                    best_results = res

    print("Plotting results and saving as multivariate_ranked.pdf")

    fig = sns.displot(best_results, x='Density', hue='class', element="step", fill=True)
    plt.title(name+": Multivariate PDF")
    fig.savefig("Results/" + name+"_multivariate_ranked.pdf")

    print("TP: " + str(np.mean(fith_p_tp)) + "+-" + str(np.std(fith_p_tp)) + " FP: " + str(np.mean(fith_p_fp)) + "+-" + str(np.std(fith_p_fp)) + " FN: " + str(np.mean(fith_p_fn)) + "+-" + str(np.std(fith_p_fn)))
    print("Mean fp for all tp: " + str(np.mean(fps)) + " +- " + str(np.std(fps)))

    t1_results = {'tp' : "%.2f +- %.2f" % (np.mean(fith_p_tp), np.std(fith_p_tp)), 'fp' : "%.2f +- %.2f" % (np.mean(fith_p_fp), np.std(fith_p_fp)), 'fn' : "%.2f +- %.2f" % (np.mean(fith_p_fn), np.std(fith_p_fn)) }
    t2_results = "%.2f +- %.2f" % (np.mean(fps), np.std(fps))

    return t1_results, t2_results

def multivariate_2(data, columns):
    # Second try using NM idea
    t = np.zeros(np.shape(data), dtype=float)

    idx = 0
    for col in columns:
        model = stats.gaussian_kde(data[col])
        t[:,idx] = model.pdf(data[col])
        idx += 1

    a = np.sum(t, axis=1)

    return a

def regression_err(data, columns, name, runs):
    print("Regression error analysis")

    # Preprocess data
    data, columns2 = preprocess(data, columns)

    target = data[data['class'] == 1.0]

    fps = []
    fith_p_tp = np.zeros(runs)
    fith_p_fp = np.zeros(runs)
    fith_p_fn = np.zeros(runs)
    fith_p = {"TP" : fith_p_tp, "FP": fith_p_fp, "FN": fith_p_fn, "PDF": []}

    for run in range(0, runs):
        # design training data using a random sample of outliers
        d_train = data[data['class'] == 0.0]

        target_sample = target.sample(n= int(len(target) * 0.75), replace=False, random_state=np.random.RandomState(seed=None))
        d_train = d_train.append(target_sample)

        e_df = {}
        lr_models = {}

        # Learn a regression focused on each column as a target
        for col in columns2:
            # Assign learning and target features
            target_f = col
            features = columns2.copy()
            features.remove(target_f)

            model = LinearRegression()
            model = model.fit(d_train[features], d_train[target_f])
            #lr_models[col] = copy.deepcopy(model)

            p_x = model.predict(d_train[features])
            e = d_train[target_f].values - p_x
            e_df[col] = e

        # Then build another regression based on all errors
        e_df = pd.DataFrame.from_dict(e_df)
        e_df.set_index(d_train.index, inplace=True)

        model = LinearRegression() # SVM / logistic regression - compare against original data on SVM
        #model = svm.SVC(kernel='poly', degree=3, gamma='auto', C=1.0)
        model = model.fit(e_df[columns2], d_train['class'])

        # How well does this model do at classifying?

        # Build a univariate density model based on a learned regression from all errors
        td = sm.nonparametric.KDEUnivariate(model.predict(e_df[columns2]))
        td.fit()

        # Add missing test data
        #e_df_test = {}
        #for col in columns2:
        #    target_f = col
        #    features = columns2.copy()
        #    features.remove(target_f)

        #    p_x = lr_models[col].predict(target[features])
        #    e = target[target_f].values - p_x

        #    e_df_test[col] = e

        #e_df_test = pd.DataFrame.from_dict(e_df_test)
        #e_df_test.set_index(target.index, inplace=True)
        
        #e_df = e_df.append(e_df_test)
        #e_df = e_df.drop_duplicates()

        # Get some predictions 
        predictions = td.evaluate(model.predict(e_df[columns2]))

        res = pd.DataFrame.from_dict({"pdf": predictions})
        res.set_index(e_df.index, inplace=True)

        min_fp = len(d_train)
        found = False

        for percentile in range(0, 101, 1):
            perc = np.percentile(res['pdf'], percentile)
            idxs = res[res['pdf'] <= perc].index

            tp = sum(d_train['class'].loc[idxs])
            fp = len(idxs) - tp
            fn = sum(d_train['class']) - tp   

            if tp == sum(d_train['class']):
                found = True
                
                if fp < min_fp:
                    min_fp = fp

            if percentile == 5:
                fith_p["TP"][run] = tp
                fith_p["FP"][run] = fp
                fith_p["FN"][run] = fn
                fith_p["PDF"].append(pd.DataFrame.from_dict({'Density' : predictions, 'class' : d_train['class'].loc[e_df.index]}))

        if found:
            fps.append(min_fp)
        else:
            fps.append(len(d_train) - sum(d_train['class']))

    print("Plotting results and saving as %s_regression_error.pdf" % name)
    
    best_tp = 0
    best_col = None
    best_i = 0
    best_f1 = 0

    # Find best result, using F1 score
    for i in range(0, runs):
        tp = fith_p["TP"][i]
        fp = fith_p["FP"][i]
        fn = fith_p["FN"][i]

        f1 = tp/(tp + ((fp+fn)*0.5))

        if f1 > best_f1:
            best_f1 = f1
            best_idx = i

        # Avoid situation where the model doesn't alert on anything and we later want to plot the PDF of this which doesn't exist
        if tp > best_tp:
            best_tp = tp
            best_col = col
            best_i = best_idx

    fig = sns.displot(fith_p["PDF"][best_i], x='Density', hue='class', element="step", fill=True)
    plt.title(name+": Regression error PDF")
    fig.savefig("Results/" + name+"_regression_error.pdf")

    best_fps = 10e10
    best_c = None

    print()

    if len(fps) > 0:
        if np.mean(fps) < best_fps:
            best_fps = np.mean(fps)

        print("FPs: %s +- %s" % (np.mean(fps), np.std(fps)))

    t1_results = {"tp": "%.2f +- %.2f " % (np.mean(fith_p['TP']), np.std(fith_p['TP'])), "fp" : "%.2f +- %.2f" % (np.mean(fith_p['FP']), np.std(fith_p['FP'])), "fn" : "%.2f +- %.2f" % (np.mean(fith_p['FN']), np.std(fith_p['FN']))}
    t2_results = {"fp" : "%.2f +- %.2f " % (np.mean(fps), np.std(fps))}
    
    print("Best TP: %s" % np.mean(fith_p['TP']))

    return t1_results, t2_results
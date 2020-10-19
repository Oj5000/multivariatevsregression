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

def preprocess(data, columns):
    # Simple pre-processing
    columns2 = columns.copy()
    
    # del singular valued cols
    del_fields = []

    for col in columns:
        if len(pd.unique(data[col])) == 1:
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
            del_fields.append(col)

    data2 = data2.drop(columns = del_fields)

    for d in del_fields:
        columns2.remove(d)

    return data2, columns2

def eval_multivariate(data, columns, name, runs):
    print()
    print("Multivariate analysis")

    target = data[data['class'] == 1.0]
    fps = []

    fith_p_tp = []
    fith_p_fp = []
    fith_p_fn = []

    for run in range(0, runs):
        # design training data using a random sample of outliers
        data2 = data[data['class'] == 0.0]
        target_sample = target.sample(n= int(len(target) * 0.75), replace=True)
        data2 = data2.append(target_sample)

        data2, columns2 = preprocess(data2, columns)

        v_type = ""
        for i in range(0, len(columns2)):
            v_type += 'c'
        
        try:
            dens_u = sm.nonparametric.KDEMultivariate(data=data2[columns2], var_type=v_type, bw='normal_reference')
            predictions = dens_u.pdf(data2[columns2])
            sorted_p = np.sort(predictions)
        except:
        #if (np.var(sorted_p) == 0.0) | (np.isnan(sorted_p).any()):
            print("statsmodels multivariate did not work, using method 2")
            predictions = multivariate_2(data2, columns2)    
        
        y_pos = np.arange(len(predictions))

        if run == 0:
            print("Plotting results and saving as multivariate_ranked.pdf")
            f = plt.figure()
            plt.title(name+": multivariate PDF")
            plt.ylabel('PDF')
            plt.bar(y_pos, np.sort(predictions), align='center', alpha=0.5)
            f.savefig("Results/" + name+"_multivariate_ranked.pdf", bbox_inches='tight')

        pdfs = []
        
        try:
            td = stats.gaussian_kde(predictions)
            pdfs = td.pdf(predictions)
        except:
            print("statsmodels multivariate did not work, using method 2")
            sorted_p = multivariate_2(data2, columns2)    

            y_pos = np.arange(len(predictions))

            if run == 0:
                print("Plotting results and saving as multivariate_ranked.pdf")
                f = plt.figure()
                plt.title(name+": multivariate PDF")
                plt.ylabel('PDF')
                plt.bar(y_pos, sorted_p, align='center', alpha=0.5)
                f.savefig("Results/" + name+"_multivariate_ranked.pdf", bbox_inches='tight')

        td = stats.gaussian_kde(predictions)
        pdfs = td.pdf(predictions)
        res = pd.DataFrame.from_dict({"pdf": pdfs})

        results = {}
        found = False
        for percentile in range(0, 101, 1):
            perc = np.percentile(res['pdf'], percentile)
            idxs = res[res['pdf'] < perc].index

            tp = sum(data2['class'].iloc[idxs])
            fp = len(data2.iloc[idxs]) - sum(data2['class'].iloc[idxs])
            fn = sum(data2['class']) - sum(data2['class'].iloc[idxs])
            
            if tp == len(target_sample):
                if found == False:
                    #print("FP:" + str(fp) + "percentile:" + str(percentile))
                    found = True
                    fps.append(fp)
            
            if percentile == 5:
                fith_p_tp.append(tp)
                fith_p_fp.append(fp)
                fith_p_fn.append(fn)

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

    target = data[data['class'] == 1.0]
    fps = {}
    fith_p = {}

    for col in columns:
        fps[col] = []

        fith_p_tp = np.zeros(runs)
        fith_p_fp = np.zeros(runs)
        fith_p_fn = np.zeros(runs)
        fith_p[col] = {"TP" : fith_p_tp, "FP": fith_p_fp, "FN": fith_p_fn}

    for run in range(0, runs):
        # design training data using a random sample of outliers
        d_train = data[data['class'] == 0.0]

        target_sample = target.sample(n= int(len(target) * 0.75), replace=False)
        d_train = d_train.append(target_sample)
        d_train, columns2 = preprocess(d_train, columns)
        
        # Find outlier indices      
        outliers_idx = d_train[d_train['class'] > 0].index

        for col in columns2:
            # Assign learning and target features
            target_f = col
            features = columns2.copy()
            features.remove(target_f)

            model = LinearRegression()
            model = model.fit(d_train[features], d_train[target_f])

            p_x = model.predict(d_train[features])
            e = p_x - d_train[target_f].values

            td = stats.gaussian_kde(e)
            
            p_x = model.predict(d_train[features])
            e = p_x - d_train[target_f].values

            pdfs = td.pdf(e)

            res = pd.DataFrame.from_dict({"pdf": pdfs})
            res.set_index(d_train.index, inplace=True)

            min_fp = len(d_train)
            found = False

            for percentile in range(0, 101, 1):
                perc = np.percentile(res['pdf'], percentile)
                idxs = res[res['pdf'] < perc].index

                tp = sum(d_train['class'].loc[idxs])
                fp = len(d_train.loc[idxs]) - sum(d_train['class'].loc[idxs])
                fn = sum(d_train['class']) - sum(d_train['class'].loc[idxs])    

                if tp == sum(d_train['class']):
                    found = True
                    
                    if fp < min_fp:
                        min_fp = fp

                if percentile == 5:
                    fithp = fith_p[col]
                    fithp["TP"][run] = tp
                    fithp["FP"][run] = fp
                    fithp["FN"][run] = fn
                    fith_p[col] = fithp

            if found:
                tmp = fps[col]
                tmp.append(min_fp)
                fps[col] = tmp

    print("Plotting results and saving as %s_regression_error.pdf" % name)
    f = plt.figure()
    plt.title(name+": Regression error")
    plt.ylabel('True positives')
    plt.xlabel('False positives')

    low_fp = 1e10
    best_col = None

    for col in columns:
        print(col)
        print("TP: " + str(np.mean(fith_p[col]["TP"])) + "+-" + str(np.std(fith_p[col]["TP"])) + " FP: " + str(np.mean(fith_p[col]["FP"])) + "+-" + str(np.std(fith_p[col]["FP"])) + " FN: " + str(np.mean(fith_p[col]["FN"])) + "+-" + str(np.std(fith_p[col]["FN"])))

        best_idx = 0
        best_f1 = 0

        # Find best result, using F1 score
        for i in range(0, runs):
            tp = fith_p[col]["TP"][i]
            fp = fith_p[col]["FP"][i]
            fn = fith_p[col]["FN"][i]

            f1 = tp/(tp + ((fp+fn)*0.5))

            if f1 > best_f1:
                best_f1 = f1
                best_idx = i

        plt.scatter(fith_p[col]["FP"][best_idx], fith_p[col]["TP"][best_idx], label=col)

        if np.mean(fith_p[col]["FP"]) < low_fp:
            low_fp = np.mean(fith_p[col]["FP"])
            best_col = col
    
    f.legend()
    f.savefig("Results/" + name+"_regression_error.pdf", bbox_inches='tight')

    for c in fps:
        print(c, np.mean(fps[c]), "+-", np.std(fps[c]))

    return {"c" : best_col, "tp": "%.2f +- %.2f " % (np.mean(fith_p[best_col]['TP']), np.std(fith_p[best_col]['TP'])), "fp" : "%.2f +- %.2f" % (np.mean(fith_p[best_col]['FP']), np.std(fith_p[best_col]['FP'])), "fn" : "%.2f +- %.2f" % (np.mean(fith_p[best_col]['FN']), np.std(fith_p[best_col]['FN']))}
import sys
sys.path.append('config/')

import numpy as np
from eval_funcs import *

from Ecoli import Ecoli
from Thyroid import Thyroid
from Avila import Avila
from Secom import Secom
from Fertility import Fertility
from Wine import Wine
from Banknote import Banknote
from Wisconsin import Wisconsin
from Shuttle import Shuttle
from Yeast import Yeast

runs = 10
datasets = [Fertility(), Thyroid(), Ecoli(), Wisconsin(), Banknote(), Yeast(), Avila(), Secom(), Wine(), Shuttle()]

dataset_descriptions = {
    'Name' : [],
    'Total records' : [],
    'Total features' : [],
    'Total outliers' : []
}

table1 = {
    'Outliers Detected' : [],
    'Outliers Missed' : [], 
    'False Positives' : []
}

table2 = {
    'False Positives' : []
}

midx1 = []
midx2 = []

for dataset in datasets:
    name = type(dataset).__name__

    print("Loading %s data" % name)
    data, columns = dataset.get_data()

    results = evaluate(data, columns, name, runs)

    # Dataset descriptions LaTeX table
    dataset_descriptions['Name'].append(name)
    dataset_descriptions['Total records'].append(len(data))
    dataset_descriptions['Total features'].append(len(columns))
    dataset_descriptions['Total outliers'].append(np.sum(data['class']))

    for result in results:
        # Multivariate KDE section in T1
        midx1.append((name, result.type))
        table1['Outliers Detected'].append(result.t1results['tp'])
        table1['Outliers Missed'].append(result.t1results['fn'])
        table1['False Positives'].append(result.t1results['fp'])

        # Multivariate KDE section in T2
        midx2.append((name, result.type))
        table2['False Positives'].append(result.t2results)

df0 = pd.DataFrame(dataset_descriptions)
df1 = pd.DataFrame(table1, index=pd.MultiIndex.from_tuples(midx1, names=['Dataset', 'Method']))
df2 = pd.DataFrame(table2, index=pd.MultiIndex.from_tuples(midx2, names=['Dataset', 'Method']))

print(df0.to_latex(index=False, caption='Dataset overview', label='datasetoverview'))
print(df1.to_latex(index=True, multirow = True, caption='Outlier detection performance at the 5th percentile', label='table1'))
print(df2.to_latex(index=True, multirow = True, caption='False positives when detecting all outliers', label='table2'))
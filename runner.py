import sys
sys.path.append('config/')

import numpy as np
from eval_funcs import *

from Ecoli import Ecoli
from Thyroid import Thyroid
from Avila import Avila
from Parkinsons import Parkinsons
from Secom import Secom
from Fertility import Fertility
from Wine import Wine
from Banknote import Banknote
from Wisconsin import Wisconsin
from Shuttle import Shuttle
from Yeast import Yeast

runs = 10
datasets = [Fertility(), Thyroid(), Ecoli(), Wisconsin(), Banknote(), Yeast(), Avila(), Parkinsons(), Secom(), Wine(), Shuttle()]

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
    
    midx1.append((name, 'M-KDE'))
    midx1.append((name, 'R-E-KDE'))
    midx2.append((name, 'M-KDE'))
    midx2.append((name, 'R-E-KDE'))

    t1results, t2results = eval_multivariate(data, columns, name, runs)
    t1r, t2r = regression_err(data, columns, name, runs)

    dataset_descriptions['Name'].append(name)
    dataset_descriptions['Total records'].append(len(data))
    dataset_descriptions['Total features'].append(len(columns))
    dataset_descriptions['Total outliers'].append(np.sum(data['class']))

    table1['Outliers Detected'].append(t1results['tp'])
    table1['Outliers Missed'].append(t1results['fn'])
    table1['False Positives'].append(t1results['fp'])

    table1['Outliers Detected'].append(t1r['tp'])
    table1['Outliers Missed'].append(t1r['fn'])
    table1['False Positives'].append(t1r['fp'])

    table2['False Positives'].append(t2results)
    table2['False Positives'].append(t2r['fp'])

df0 = pd.DataFrame(dataset_descriptions)
df1 = pd.DataFrame(table1, index=pd.MultiIndex.from_tuples(midx1, names=['Dataset', 'Method']))
df2 = pd.DataFrame(table2, index=pd.MultiIndex.from_tuples(midx2, names=['Dataset', 'Method']))

print(df0.to_latex(index=False, caption='Dataset overview', label='datasetoverview'))
print(df1.to_latex(index=True, multirow = True, caption='Outlier detection performance at the 5th percentile', label='table1'))
print(df2.to_latex(index=True, multirow = True, caption='False positives when detecting all outliers', label='table2'))
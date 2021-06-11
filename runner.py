import sys
import os
sys.path.append('config/')

import numpy as np
from eval_funcs import *

from WaveEnergyConverters import WaveEnergyConverters
from ClickStreamShopping import ClickStreamShopping
from QSARFishToxicity import QSARFishToxicity

runs = 10
datasets = [WaveEnergyConverters(), ClickStreamShopping(), QSARFishToxicity()]

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

all_results = []
data_sizes = []

if not os.path.exists('data/'):
    os.makedirs('data/')

for dataset in datasets:
    name = type(dataset).__name__

    print("Loading %s data" % name)
    dataset.load_data()
    data_sizes.append({"n": len(dataset.data), "features": len(dataset.columns)})
    
    results = evaluate(dataset, name, runs)

    all_results.append(results)

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

# Final performance chart
fig = plt.figure()
plt.title("Performance by method")

seen_methods = []

# data set
for r in range(0, len(all_results)):
    # method
    for method in all_results[r]:
        if method.chartName == 'linear_regression':
            colour = 'blue'
        else:
            colour = 'orange'

        if method.type not in seen_methods:
            label = method.type
            seen_methods.append(method.type)
        else:
            label = None
        
        plt.scatter(data_sizes[r]['n'], method.fith_p_tp[0], c=colour, label=label, marker=method.marker, s=data_sizes[r]['features']*10, edgecolor='black', alpha=0.5)

plt.xlabel("Number of records")
plt.ylabel("Outliers detected")
plt.legend(loc='upper left')
fig.savefig("Results/performance.pdf")
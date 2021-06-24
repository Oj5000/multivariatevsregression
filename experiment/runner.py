import os

import numpy as np
from eval_funcs import *

from config.QSARFishToxicity import QSARFishToxicity
from config.GasEmission import GasEmission
from config.AirQuality import AirQuality
from config.CommunitiesAndCrime import CommunitiesAndCrime
from config.Superconductivity import Superconductivity
from config.CTSliceAxial import CTSliceAxial

runs = 10
mutation = 0.3 # 20% mutation
datasets = [QSARFishToxicity()]#[CTSliceAxial(), Superconductivity(), CommunitiesAndCrime(), AirQuality(), GasEmission(), QSARFishToxicity()]

dataset_descriptions = {
    'Name' : [],
    'Total records' : [],
    'Total features' : [],
    'Total outliers' : []
}

table1 = {
    'Dataset mutation (*std)' : [],
    'Area under ROC' : []
}

table2 = {
    'Dataset mutation (*std)' : [],
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
    
    results = evaluate(dataset, runs, mutation)

    all_results.append(results)

    # Dataset descriptions LaTeX table
    dataset_descriptions['Name'].append(name)
    dataset_descriptions['Total records'].append(len(dataset.data))
    dataset_descriptions['Total features'].append(len(dataset.columns))
    dataset_descriptions['Total outliers'].append(round(len(dataset.data)*mutation))

    for result in results:
        for mut_amp in result.results.keys():
            # Multivariate KDE section in T1
            midx1.append((name, result.type))
            table1['Dataset mutation (*std)'].append("%.1f" %mut_amp)
            table1['Area under ROC'].append(result.get_auc_result(mut_amp))

            # Multivariate KDE section in T2
            midx2.append((name, result.type))
            table2['Dataset mutation (*std)'].append("%.1f" %mut_amp)
            table2['False Positives'].append("%.1f +- %.4f" % (np.mean(result.results[mut_amp]['all_tp']), np.std(result.results[mut_amp]['all_tp'])))

df0 = pd.DataFrame(dataset_descriptions)
df1 = pd.DataFrame(table1, index=pd.MultiIndex.from_tuples(midx1, names=['Dataset', 'Method']))
df2 = pd.DataFrame(table2, index=pd.MultiIndex.from_tuples(midx2, names=['Dataset', 'Method']))

print(df0.to_latex(index=False, caption='Dataset overview', label='datasetoverview'))
print(df1.to_latex(index=True, multirow = True, caption='Area under the ROC curve', label='table1'))
print(df2.to_latex(index=True, multirow = True, caption='False positives when detecting all outliers', label='table2'))

with open('results/results.txt', 'w') as writer:
    writer.write(df0.to_latex(index=False, caption='Dataset overview', label='datasetoverview'))
    writer.write("")
    writer.write(df1.to_latex(index=True, multirow = True, caption='Area under the ROC curve', label='table1'))
    writer.write("")
    writer.write(df2.to_latex(index=True, multirow = True, caption='False positives when detecting all outliers', label='table2'))

    writer.close()


# Final performance chart
#fig = plt.figure()
#plt.title("Performance by method")
#
#seen_methods = []
#
## data set
#for r in range(0, len(all_results)):
#    # method
#    for method in all_results[r]:
#        if method.chartName == 'linear_regression':
#            colour = 'blue'
#        else:
#            colour = 'orange'
#
#        if method.type not in seen_methods:
#            label = method.type
#            seen_methods.append(method.type)
#        else:
#            label = None
#        
#        plt.scatter(data_sizes[r]['n'], method.fith_p_tp[0], c=colour, label=label, marker=method.marker, s=data_sizes[r]['features']*10, edgecolor='black', alpha=0.5)
#
#plt.xlabel("Number of records")
#plt.ylabel("Outliers detected")
#plt.legend(loc='upper left')
#fig.savefig("Results/performance.pdf")
import sys
sys.path.append('config/')

from eval_funcs import *

from Ecoli import Ecoli
from Thyroid import Thyroid
from Yeast import Yeast
from Wisconsin import Wisconsin
from Shuttle import Shuttle
from Parkinsons import Parkinsons
from Secom import Secom
from Fertility import Fertility
from Wine import Wine
from APS_Failure import APS_Failure

runs = 5
datasets = [Fertility(), Wine()]#[Ecoli(), Thyroid(), Yeast(), Wisconsin(), Shuttle(), Parkinsons(), Secom(), Fertility(), Wine(), APS_Failure()]

table1 = {
    'Target Feature' : [],
    'Outliers Detected' : [],
    'Outliers Missed' : [], 
    'False Positives' : []
}

table2 = {
    'Target Feature' : [],
    'False Positives' : []
}

midx1 = []
midx2 = []

for dataset in datasets:
    name = type(dataset).__name__

    print("Loading %s data" % name)
    data, columns = dataset.get_data()

    data_info = "%s records, %s features" % (len(data), len(columns))
    
    midx1.append((name, 'M-KDE', data_info))
    midx1.append((name, 'R-E-KDE', data_info))
    midx2.append((name, 'M-KDE'))
    midx2.append((name, 'R-E-KDE'))

    t1results, t2results = eval_multivariate(data, columns, name, runs)
    t1r, t2r = regression_err(data, columns, name, runs)

    table1['Target Feature'].append("")
    table1['Outliers Detected'].append(t1results['tp'])
    table1['Outliers Missed'].append(t1results['fn'])
    table1['False Positives'].append(t1results['fp'])

    table1['Target Feature'].append(t1r['c'])
    table1['Outliers Detected'].append(t1r['tp'])
    table1['Outliers Missed'].append(t1r['fn'])
    table1['False Positives'].append(t1r['fp'])

    table2['Target Feature'].append("")
    table2['False Positives'].append(t2results)

    table2['Target Feature'].append(t2r['c'])
    table2['False Positives'].append(t2r['fp'])

df1 = pd.DataFrame(table1, index=pd.MultiIndex.from_tuples(midx1, names=['Dataset', 'Method', 'Dataset Size']))
df2 = pd.DataFrame(table2, index=pd.MultiIndex.from_tuples(midx2, names=['Dataset', 'Method']))

print(df1.to_latex(index=True, multirow = True))
print(df2.to_latex(index=True, multirow = True))
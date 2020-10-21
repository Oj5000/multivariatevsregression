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

runs = 200
datasets = [Ecoli(), Thyroid(), Yeast(), Wisconsin(), Shuttle(), Parkinsons(), Secom(), Fertility(), Wine()]

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

midx = []

for dataset in datasets:
    name = type(dataset).__name__

    midx.append((name, 'M-KDE'))
    midx.append((name, 'R-E-KDE'))

    print("Loading %s data" % name)
    data, columns = dataset.get_data()
    
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

df1 = pd.DataFrame(table1, index=pd.MultiIndex.from_tuples(midx, names=['Dataset', 'Method']))
df2 = pd.DataFrame(table2, index=pd.MultiIndex.from_tuples(midx, names=['Dataset', 'Method']))

print(df1.to_latex(index=True, multirow = True))
print(df2.to_latex(index=True, multirow = True))
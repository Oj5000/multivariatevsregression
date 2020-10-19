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

runs = 20
datasets = [Ecoli(), Thyroid(), Yeast(), Wisconsin(), Shuttle(), Parkinsons(), Secom(), Fertility(), Wine()]

for dataset in datasets:

    print("Loading data")
    data, columns = dataset.get_data()
    
    eval_multivariate(data, columns, type(dataset).__name__, runs)
    regression_err(data, columns, type(dataset).__name__, runs)
import copy
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from evaluators.RegressionBase import RegressionBase

class LinearEvaluator(RegressionBase):

    def __init__(self, columns, polynomials):
        RegressionBase.__init__(self, columns, polynomials)

    def fitpredict(self, data):
        e_df = {}

        columns = list(data.columns)

        # Learn a regression focused on each column as a target
        for col in columns:
            # Assign learning and target features
            target_f = col
            features = columns.copy()
            features.remove(target_f)

            model = make_pipeline(PolynomialFeatures(self.polynomials), LinearRegression())
            model = model.fit(data[features], data[target_f])
            self.lr_models[col] = copy.deepcopy(model)

            p_x = model.predict(data[features])
            e = (data[target_f].values - p_x)**2
            e_df[col] = e

        e_df = pd.DataFrame.from_dict(e_df)
        e_df.set_index(data.index, inplace=True)

        return e_df
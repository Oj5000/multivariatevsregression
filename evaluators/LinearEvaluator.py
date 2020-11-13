from evaluators.RegressionBase import RegressionBase

import statsmodels.api as sm

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

class LinearEvaluator(RegressionBase):

    def __init__(self, polynomials):
        RegressionBase.__init__(self, polynomials)

        self.type = 'Linear Regression degrees = %i' % polynomials
        self.chartName = 'linear_regression'

    def fitpredict(self, data, target):
        e_df = super(LinearEvaluator, self).prebuild(data)

        # Then build another regression based on all errors
        model = make_pipeline(PolynomialFeatures(self.polynomials), LinearRegression())
        model.fit(e_df, target)

        # Build a univariate density model based on a learned regression from all errors
        td = sm.nonparametric.KDEUnivariate(model.predict(e_df))
        td.fit()

        # Get some predictions - need to do a memory thing here
        if len(data) > 100000:
            predictions = []

            n = 0
            batch_size = 2000
            x = n + batch_size

            while n < len(e_df):
                predictions.extend(td.evaluate(model.predict(e_df.iloc[n:x])))

                n += batch_size

                if (x + batch_size) >= len(e_df):
                    x = len(e_df)
                else:
                    x += batch_size

        else:
            predictions = td.evaluate(model.predict(e_df))

        return predictions
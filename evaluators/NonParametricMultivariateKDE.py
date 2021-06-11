from evaluators.EvaluatorBase import EvaluatorBase

import statsmodels.api as sm

class NonParametricMultivariateKDE (EvaluatorBase):

    def __init__(self):
        EvaluatorBase.__init__(self)

        self.type = 'Non Parametric Multivariate KDE'
        self.chartName = 'nonparametric_multivariate'
        self.marker = 'd'

    def fitpredict(self, data, target=None):
        v_type = ""
        for i in range(0, len(data.columns)):
            v_type += 'c'

        dens_u = sm.nonparametric.KDEMultivariate(data=data, var_type=v_type, bw='normal_reference')
        predictions = dens_u.pdf(data)

        return predictions
from multivariate.MultivariateBase import MultivariateBase

import statsmodels.api as sm

class NonParametricMultivariateKDE (MultivariateBase):

    def __init__(self):
        MultivariateBase.__init__(self)

        self.type = 'Non Parametric Multivariate KDE'
        self.chartName = 'nonparametric_multivariate_ranked'

    def fitpredict(self, data):
        v_type = ""
        for i in range(0, len(data.columns)):
            v_type += 'c'

        dens_u = sm.nonparametric.KDEMultivariate(data=data, var_type=v_type, bw='normal_reference')
        predictions = dens_u.pdf(data)

        return predictions
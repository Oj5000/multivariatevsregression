
class RegressionBase():

    def __init__(self, polynomials):
        self.polynomials = polynomials
        self.lr_models = {}

        self.type = 'Linear Regression, %i degrees' % polynomials
        self.chartName = 'linear_regression'
        self.marker = ['o', 'v', '^', '<', '>', 'x'][polynomials]
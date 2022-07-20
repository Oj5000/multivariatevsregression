import numpy as np
import pandas as pd
from config.DataSetBase import DataSetBase

class Synthetic(DataSetBase):

    def __init__(self, n_cols, n_rows):
        rng = np.random.default_rng(12345)
        rints = rng.integers(low=0.1, high=100, size=n_cols)

        cols = []
        for x in range(n_cols):
            cols.append(str(x))

        self.data = pd.DataFrame(columns=cols)

        for x in range(n_cols):
            s = np.random.normal(0, rints[x], n_rows)
            self.data[cols[x]] = s

        self.columns = self.data.columns
        self.mutation_cols = self.data.columns
import pandas as pd
import os
import zipfile
import numpy as np
from config.DataSetBase import DataSetBase

class Cargo2000(DataSetBase):

    def load_data(self):
        name = "Cargo2000"
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00382/c2k_data_comma.csv'

        if not os.path.exists("data/"+name+".csv"):
            self.download_data(url, name)

            os.rename('data/c2k_data_comma.csv', 'data/' + name + '.csv')
            print("Done. Cleaning up")

        self.data = pd.read_csv("data/"+name+".csv", sep=',', header=0, na_values='?')

        self.data.drop(columns=['nr','i1_legid','i1_dep_1_place','i1_rcf_1_place','i1_dep_2_place','i1_rcf_2_place','i1_dep_3_place','i1_rcf_3_place','i2_legid','i2_dep_1_place','i2_rcf_1_place','i2_dep_2_place','i2_rcf_2_place','i2_dep_3_place','i2_rcf_3_place','i3_legid','i3_dep_1_place','i3_rcf_1_place','i3_dep_2_place','i3_rcf_2_place','i3_dep_3_place','i3_rcf_3_place','o_legid','o_dep_1_place','o_rcf_1_place','o_dep_2_place','o_rcf_2_place','o_dep_3_place','o_rcf_3_place'], inplace=True)
        self.columns = self.data.columns
        self.mutation_cols = ['i1_rcs_p', 'i1_rcs_e', 'i1_dep_1_p', 'i1_dep_1_e', 'i1_rcf_1_p', 'i1_rcf_1_e', 'i1_dep_2_p', 'i1_dep_2_e', 'i1_rcf_2_p', 'i1_rcf_2_e', 'i1_dep_3_p', 'i1_dep_3_e', 'i1_rcf_3_p', 'i1_rcf_3_e', 'i1_dlv_p', 'i1_dlv_e', 'i1_hops', 'i2_rcs_p', 'i2_rcs_e', 'i2_dep_1_p', 'i2_dep_1_e', 'i2_rcf_1_p', 'i2_rcf_1_e', 'i2_dep_2_p', 'i2_dep_2_e', 'i2_rcf_2_p', 'i2_rcf_2_e', 'i2_dep_3_p', 'i2_dep_3_e', 'i2_rcf_3_p', 'i2_rcf_3_e', 'i2_dlv_p', 'i2_dlv_e', 'i2_hops', 'i3_rcs_p', 'i3_rcs_e', 'i3_dep_1_p', 'i3_dep_1_e', 'i3_rcf_1_p', 'i3_rcf_1_e', 'i3_dep_2_p', 'i3_dep_2_e', 'i3_rcf_2_p', 'i3_rcf_2_e', 'i3_dep_3_p', 'i3_dep_3_e', 'i3_rcf_3_p', 'i3_rcf_3_e', 'i3_dlv_p', 'i3_dlv_e', 'i3_hops', 'o_rcs_p', 'o_rcs_e', 'o_dep_1_p', 'o_dep_1_e', 'o_rcf_1_p', 'o_rcf_1_e', 'o_dep_2_p', 'o_dep_2_e', 'o_rcf_2_p', 'o_rcf_2_e', 'o_dep_3_p', 'o_dep_3_e', 'o_rcf_3_p', 'o_rcf_3_e', 'o_dlv_p', 'o_dlv_e']
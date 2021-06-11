import pandas as pd

#  Class Distribution. The class is the localization site. Please see Nakai &
#               Kanehisa referenced above for more details.
#  CYT (cytosolic or cytoskeletal)                    463
#  NUC (nuclear)                                      429
#  MIT (mitochondrial)                                244
#  ME3 (membrane protein, no N-terminal signal)       163
#  ME2 (membrane protein, uncleaved signal)            51
#  ME1 (membrane protein, cleaved signal)              44
#  EXC (extracellular)                                 37
#  VAC (vacuolar)                                      30
#  POX (peroxisomal)                                   20
#  ERL (endoplasmic reticulum lumen)                    5

class Yeast:

    def get_data(self):
        data = pd.read_csv("data/"+type(self).__name__+".csv")
        columns = list(data.columns[0:8])

        # Pick a class
        classes = ['CYT','NUC','MIT','ME3','ME2','ME1','EXC','VAC','POX','ERL']
        target = 'MIT'

        data['class'] = data['class'].replace([target], 1)
        data['class'] = data['class'].replace([classes], 0)

        return data, columns
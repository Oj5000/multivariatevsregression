import pandas as pd

class Fraud:

    def get_data(self):
        ppdata = pd.read_csv('C:\\Users\\olivert\\Desktop\\data\\pp_datasub.csv', sep=',')
        enrdata = pd.read_csv('C:\\Users\\olivert\\Desktop\\data\\enrich.csv', sep=',')

        ppdata.set_index('id', inplace=True)
        enrdata.set_index('id', inplace=True)

        data = ppdata[['class']].join(enrdata)
        data.drop(columns=['is_fraud', 'cardkey_cd', 'transactiondategmt_ta'], inplace=True)

        columns = list(data.columns[1:])

        return data, columns
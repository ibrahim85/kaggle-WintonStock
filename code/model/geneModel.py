'''

@author: FZY
'''
from meanModel import predictRaw_test,param_raw_mean
import pandas as pd
if __name__ == '__main__':
    test = pd.read_csv("../../data/test_clean_fill_all_1.csv")
    test_price = pd.read_csv("../../data/test_ts_price.csv")
    test['Ret_120_price'] = test_price['Ret_120_price']
    predictRaw_test(param_raw_mean, test)
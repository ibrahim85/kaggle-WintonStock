'''
Created on 2015/12/15

@author: FZY
'''
import pandas as pd 
import numpy as np
if __name__ == '__main__':
    train = pd.read_csv("../../data/train.intra_1.csv")
    ret_values = []
    for i in range(121,181):
        tmp = np.median(train['Ret_%d'%(i)])
        ret_values.append(tmp)
    ret_values.append(0)
    ret_values.append(0)
    ret_values = ret_values*60000
    pd.Series(ret_values).to_csv("../../data/result/temp.csv")
        
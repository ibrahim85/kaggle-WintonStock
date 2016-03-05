'''
Created on 2015/11/12

@author: FZY
'''

import pandas as pd 
import numpy as np

#change the format p2-p1/p2 to p2/p1
def changeFormat(data,feature):
    return data[feature]+1
           
if __name__ == '__main__':

    train = pd.read_csv("../../data/train_clean_fill_all_1.csv")
    test = pd.read_csv("../../data/test_clean_fill_all_1.csv")
    #this model is base on mean of value 
    print 'change format'
    
    for i in range(2,121):
        feature = 'Ret_'+str(i)
        train['Ret_%d_p'%(i)] = list(train.apply(lambda x : changeFormat(x,feature),axis=1))
        test['Ret_%d_p'%(i)] = list(test.apply(lambda x : changeFormat(x,feature),axis=1))
    Rets = []
    for i in range(2,121):
        Ret = "Ret_"+str(i)+"_p"
        Rets.append(Ret)
    train[Rets].to_csv("../../data/train_ts_rate.csv")
    test[Rets].to_csv("../../data/test_ts_rate.csv")
    
    for i in range(2,121):
        if i == 2:
            train['Ret_2_price'] = train['Ret_2_p']
            test['Ret_2_price'] = test['Ret_2_p']
        else:
            train['Ret_%d_price'%(i)] = train['Ret_%d_price'%(i-1)]*train['Ret_%d_p'%(i)]
            test['Ret_%d_price'%(i)] = test['Ret_%d_price'%(i-1)]*test['Ret_%d_p'%(i)]    
    Rets = []
    for i in range(2,121):
        Ret = "Ret_"+str(i)+"_price"
        Rets.append(Ret)
       
    train[Rets].to_csv("../../data/train_ts_price.csv")
    test[Rets].to_csv("../../data/test_ts_price.csv")
    
        
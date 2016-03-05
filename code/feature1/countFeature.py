'''
Created on 2015/11/26

@author: FZY
'''
import pandas as pd , numpy as np

def calPrice(x):
    return (x['Ret_MinusOne']+1)*(x['Ret_120_price'])

def calMean(x):
    features = []
    for i in range(2,121):
        feature = "Ret_%d"%(i)
        features.append(feature)
    return np.mean(x[features])

def calMean_price(x):
    features = []
    for i in range(2,121):
        feature = "Ret_%d_price"%(i)
        features.append(feature)
    return np.mean(x[features])
def calMax_price(x):
    features = []
    for i in range(2,121):
        feature = "Ret_%d_price"%(i)
        features.append(feature)
    return np.max(x[features])
def calMax(x):
    features = []
    for i in range(2,121):
        feature = "Ret_%d"%(i)
        features.append(feature)
    return np.max(x[features])

def calMin_price(x):
    features = []
    for i in range(2,121):
        feature = "Ret_%d_price"%(i)
        features.append(feature)
    return np.min(x[features])
def calMin(x):
    features = []
    for i in range(2,121):
        feature = "Ret_%d"%(i)
        features.append(feature)
    return np.min(x[features])

def calVar_price(x):
    features = []
    for i in range(2,121):
        feature = "Ret_%d_price"%(i)
        features.append(feature)
    return np.std(x[features])
def calVar(x):
    features = []
    for i in range(2,121):
        feature = "Ret_%d"%(i)
        features.append(feature)
    return np.std(x[features])
if __name__ == '__main__':
    #in this section,I will find the other count Feature
    train_price = pd.read_csv("../../data/train_ts_price.csv")
    train = pd.read_csv("../../data/train_clean_fill_all_1.csv")
    test_price = pd.read_csv("../../data/test_ts_price.csv")
    test = pd.read_csv("../../data/test_clean_fill_all_1.csv")
    
    #at first I will count the mean,variance , max,min,Price_total
    train['Ret_120_price'] = train_price['Ret_120_price']
    test['Ret_120_price'] = test_price['Ret_120_price']
    train['Ret_total_price'] = list(train.apply(lambda x :calPrice(x),axis=1))
    test['Ret_total_price'] = list(test.apply(lambda x :calPrice(x),axis=1))
    
    #cal mean
    train['Ret_mean'] = list(train.apply(lambda x :calMean(x),axis=1))
    train['Ret_mean_price'] = list(train.apply(lambda x :calMean_price(x),axis=1)) 
    test['Ret_mean'] = list(test.apply(lambda x :calMean(x),axis=1))
    test['Ret_mean_price'] = list(test.apply(lambda x :calMean_price(x),axis=1))
    
    #cal max 
    
    train['Ret_max'] = list(train.apply(lambda x :calMax(x),axis=1))
    train['Ret_max_price'] = list(train.apply(lambda x :calMax_price(x),axis=1)) 
    test['Ret_max'] = list(test.apply(lambda x :calMax(x),axis=1))
    test['Ret_max_price'] = list(test.apply(lambda x :calMax_price(x),axis=1))
    
    #cal min 
    
    train['Ret_min'] = list(train.apply(lambda x :calMin(x),axis=1))
    train['Ret_min_price'] = list(train.apply(lambda x :calMin_price(x),axis=1)) 
    test['Ret_min'] = list(test.apply(lambda x :calMin(x),axis=1))
    test['Ret_min_price'] = list(test.apply(lambda x :calMin_price(x),axis=1))
    
    #cal variance 
    
    train['Ret_var'] = list(train.apply(lambda x :calVar(x),axis=1))
    train['Ret_var_price'] = list(train.apply(lambda x :calVar_price(x),axis=1)) 
    test['Ret_var'] = list(test.apply(lambda x :calVar(x),axis=1))
    test['Ret_var_price'] = list(test.apply(lambda x :calVar_price(x),axis=1))
    
    #save data
    train.to_csv("../../train_clean_fill_all_2.csv")
    test.to_csv("../../data/test_clean_fill_all_2.csv")
    
    
    
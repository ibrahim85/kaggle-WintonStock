'''
Created on 2015/11/9

@author: FZY
'''

#now we use some TS function to predict 
#we need to analysis the value of RET
import pandas as pd 
import numpy as np
def getMessageByNorm(data,features):
    res = np.float64(1.0)
    for feature in features:
        res = np.float64(1-data[feature])*res
    return res
def getMessageByLog(data,features):
    res = np.float64(1.0)
    for feature in features:
        res = (1-np.exp(data[feature]))*res
    return res

def countVarMessage(data,features):
    return np.std(data[features])
        
def countSumMessage(data,features):
    return np.sum(data[features])

def catRet(data):
    return (1-data['Ret_MinusOne'])*(data['Ret_intraday'])
    
def countLastThreeDay(data):
    return data['Ret_D']*(1-data['Ret_PlusOne'])*(1-data['Ret_PlusTwo'])  
if __name__ == "__main__":
    #at first we need to analysis what is reture
    #from the range of RET ,the range of RET is (-0.01,0.01)
    #Suppose that the fist price is P1,the second Price is P2,the RET 
    #mabye is ((P2-P1)/P2) or log((P2-P1)/P2)
    #we can analysis the value of reture
    #by analysis the feature , we know it can not be true,using log
    
    train = pd.read_csv("../../data/train_clean_fill_all_1.csv")
    features = []
    for i in range(121,181):
        feature= "Ret_" + str(i)
        features.append(feature)
    test_train = train[features]
    #print test_train.shape
    message1 = list(test_train.apply(lambda x :getMessageByNorm(x, features),axis=1))   
    #now we need to calucate the reture value
    train['Ret_intraday'] = message1
    #calucate the value of the day of D 
    train['Ret_D'] = list(train.apply(lambda x : catRet(x),axis=1)) 
    #calucate the last three days, D D+1 D+2
    train['Ret_last3'] = list(train.apply(lambda x : countLastThreeDay(x),axis=1))
    #now we need to calucate the Ret_121 to Ret_180
    train.to_csv("../../data/train_weight.csv")
    
    
    

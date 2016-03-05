'''
Created on 2015/11/25

@author: FZY
'''
from CVModel import loadCVIndex
import pandas as pd 
import numpy as np
def getErr(x):
    error = 0
    error = error + x['Weight_Daily']*np.abs(x['Ret_PlusOne']-0)
    error = error + x['Weight_Daily']*np.abs(x['Ret_PlusTwo']-0)
    return error 

def getErr_1(x):
    error = 0
    error = error + x['Weight_Daily']*np.abs(x['Ret_PlusOne']-0)
    return error

def getErr_2(x):
    error = 0
    error = error + x['Weight_Daily']*np.abs(x['Ret_PlusTwo']-0)
    return error
def getPartError(data):
    data['error'] = list(data.apply(lambda x : getErr(x),axis=1))
    error = np.sum(data['error'])
    print "error_all:%f"%(error)
    
def getPartError_1(data):
    data['error'] = list(data.apply(lambda x : getErr_1(x),axis=1))
    error = np.sum(data['error'])
    print "error_1:%f"%(error)
def getPartError_2(data):
    data['error'] = list(data.apply(lambda x : getErr_2(x),axis=1))
    error = np.sum(data['error'])
    print "error_2:%f"%(error)   
    
if __name__ == '__main__':
    train = pd.read_csv("../../data/train_clean_fill_all_2.csv")
    train_index = loadCVIndex("../../data/cv/valid.run%d.txt"%(1))
    print 'the information of index 1'
    train_1 = train.iloc[train_index]
    getPartError(train_1)
    getPartError_1(train_1)
    getPartError_2(train_1)
    
    print 'the information of index 2'
    train_index = loadCVIndex("../../data/cv/valid.run%d.txt"%(2))
    train_1 = train.iloc[train_index]
    getPartError(train_1)
    getPartError_1(train_1)
    getPartError_2(train_1)
    
    
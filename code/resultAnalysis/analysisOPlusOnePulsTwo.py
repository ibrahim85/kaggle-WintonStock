'''
Created on 2015/11/25

@author: FZY
'''

import pandas as pd
import numpy as np
#read data 
def getDailyerror(x):
    error1 = x['Weight_Daily']*(np.abs(x['61']-x['Ret_PlusOne']))
    error2 = x['Weight_Daily']*(np.abs(x['62'])-x['Ret_PlusTwo'])
    return error1 + error2

def getIntradayerror(x):
    error1 = 0
    for i in range(121,181):
        error1 = x['Weight_Intraday']*(np.abs(x['Ret_%d'%(i)]-x['Ret_%d_pred'%(i)])) + error1
    return error1
def getIntradayErrorOfZeroBenchMark(x):
    error1 = 0
    for i in range(121,181):
        error1 = x['Weight_Intraday']*(np.abs(x['Ret_%d'%(i)]-0)) + error1
    return error1
def getDailyErrorOfZeroBenchMark(x):
    error1 = x['Weight_Daily']*(np.abs(x['Ret_PlusOne']-0))
    error2 = x['Weight_Daily']*(np.abs(x['Ret_PlusTwo']-0))
    return error1 + error2
if __name__ == '__main__':
    train =  pd.read_csv("../../data/analysis/result/data.res.csv")
    train['error_daily']= list(train.apply(lambda x : getDailyerror(x),axis=1))
    train['error_intraday'] = list(train.apply(lambda x : getIntradayerror(x),axis=1))
    train['error_intray_0'] = list(train.apply(lambda x : getIntradayErrorOfZeroBenchMark(x),axis=1))
    train['error_daily_0'] = list(train.apply(lambda x : getDailyErrorOfZeroBenchMark(x),axis=1))
    print'error_daily_Predict:'
    print np.sum(train['error_daily'])
    print'error_intray_Predict:'
    print np.sum(train['error_intraday'])
    print 'all_error_predict'
    print (np.sum(train['error_intraday'])+np.sum(train['error_daily']))
    print 'mase error of all_error_predict'
    print (np.sum(train['error_intraday'])+np.sum(train['error_daily']))/(40000*62)
    print'error_intray_0:'
    print np.sum(train['error_intray_0'])
    print'error_daily_0'
    print np.sum(train['error_daily_0'])
    print'all_error_0'
    print (np.sum(train['error_intray_0'])+np.sum(train['error_daily_0']))
    print "mase error of all_error_0"
    print (np.sum(train['error_intray_0'])+np.sum(train['error_daily_0']))/(40000*62)
    
    
    
    
    """
    ########
    
    
    
    
    ########


    #######\
    for the data as we count,we know that the main influnece of is daily_mydata,
    so we can not use the ts to predict 
    
    """
    
    
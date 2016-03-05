'''
Created on 2015/11/23
@author: FZY
'''
import pandas as pd 
import  rpy2.robjects as robj
from rpy2.robjects.packages import importr
from hyperopt import hp,Trials,tpe,fmin
import numpy as np
list_values = []
model_param = {
    'task':'arima_all_mean',
    'frequency':60,
    'max_evals':1
               
}

def changeFormat(data,feature):
    return 1.0 + data[feature]
def transform_format(data):
    print 'change format' 
    data['Ret_180_price'] = data
    
    
def creat_All_model(data):
    print 'create model'
    importr('forecast')
    robj.r('''
       all_mean_arima <- function(data){
         train = as.numeric(data[,2])
         train_ts = ts(train,1)
         train_ts<-diff(train_ts,differences=1)
         jpeg(file="myplot_ts.jpeg")
         plot(train_ts)
         dev.off()
         acf = acf(train_ts,lag.max=60,plot=FALSE)
         jpeg(file="myplot_acf.jpeg")
         plot(acf)
         dev.off()
         pacf = pacf(train_ts,lag.max=60,plot=FALSE)
         jpeg(file="myplot_pacf.jpeg")
         plot(pacf)
         dev.off()
         data.fit <- Arima(train_ts,order=c(16,1,0))
         forecast <- forecast.Arima(data.fit,h=60,level=c(99.5))
         return(forecast$mean)   
       } 
    '''
    )
    res = robj.r('all_mean_arima')(data)
    res= robj.r('as.numeric')(res)
    res = np.array(res)
    print res
    return res

def listRet(data):
    for i in range(121,181):
        list_values .append(data['Ret_%d'%(i)])
    list_values.append(data['61'])
    list_values.append(data['62']) 
def predictRaw_test(data,result,res):
    k = 121
    for r  in res:
        data['Ret_%d_pred'%(k)] = r
        k = k + 1
    data['Ret_180_price'] = result
    Ret_1 = data['Ret_MinusTwo']
    Ret_2 = ((Ret_1+1)*data['Ret_120_price']*data['Ret_180_price'])-1
    data['61'] = 0.5*Ret_1 + 0.5*Ret_2
    data['62'] = 0.5*Ret_2 + 0.5*data['61']
    data.apply(lambda x : listRet(x),axis=1)
    data.to_csv("../../data/analysis/result/data.res.csv")
    pd.Series(list_values).to_csv("../../data/all_arima_result.csv")
if __name__ == '__main__':
    #read to by R
    """
    test_price = pd.read_csv("../../data/test_ts_price.csv")
    test = pd.read_csv("../../data/test_clean_fill_all_1.csv")
    test['Ret_120_price'] = test_price['Ret_120_price']
    data = robj.r('read.csv')('../../data/feature/test_count_mean.csv',sep=",",header=False)
    res = list(creat_All_model(data))
    result = 1
    for r in res:
        result = result * (1+r)
    predictRaw_test(test, result,res)
    """

    
    #at first we need to analysis the major influnence of Ret_plusOne,Ret_plusTwo
    test_price = pd.read_csv("../../data/train_ts_price.csv")
    test = pd.read_csv("../../data/train_clean_fill_all_1.csv")
    test['Ret_120_price'] = test_price['Ret_120_price']
    data = robj.r('read.csv')('../../data/feature/count_mean.csv',sep=",",header=False)
    res = list(creat_All_model(data))
    result = 1
    for r in res:
        result = result * (1+r)
    predictRaw_test(test, result,res)
    
    
        
    
    

    
    
    
'''
Created on 2015/11/24

@author: FZY
'''
import pandas as pd
from ml_metrics import WMAE_model
import numpy as np
from rpy2.robjects.vectors import DataFrame
import  rpy2.robjects as robj
from rpy2.robjects.packages import importr
def changeFormat(data,feature):
    return data[feature]+1
def transform_format(data):
    print 'change format' 
    for i in range(121,181):
        feature = 'Ret_%d_pred'%(i)
        data['Ret_%d_p'%(i)] = list(data.apply(lambda x : changeFormat(x,feature),axis=1))
    
    for i in range(121,181):
        if i == 121:
            data['Ret_121_price'] = data['Ret_121_p']
        else:
            data['Ret_%d_price'%(i)] = data['Ret_%d_price'%(i-1)]*data['Ret_%d_p'%(i)] 
            
def createModel(data):
    print 'Create_model'
    importr('forecast')
    robj.r(
           '''
            arima_data <- function(data){
               
               best_arima = auto.arima(data,trace=F,stepwise=T)
               forecast = forecast.Arima(best_arima,h=60,level=c(99.5))
               output = forecast$mean
               return (output)
               
               
            }
            
   '''     
    )
    features_names = ["Ret_%d_pred"%(i) for i in range(121,181)]
    predict  = pd.DataFrame(columns=features_names)
    i = 1
    for tmp in DataFrame.iter_row(data):
        if i % 100 == 0:
            print i 
        tmp = robj.r("as.numeric")(tmp)
        #tmp = robj.r('ts')(tmp,start=2)
        tmp = robj.r('ts')(tmp,start=2,frequency=15)
        forecast = robj.r('arima_data')(tmp)
        forecast = robj.r('as.numeric')(forecast)
        forecast = np.array(forecast)
        predict2 = pd.DataFrame(forecast).T
        predict2.columns= features_names
        predict = pd.concat([predict,predict2],axis=0)
        i =  i + 1
    print predict
    predict.to_csv("tmp1.csv")
    
if __name__ =="__main__":
    #read messsage 
    
    #raw
    
    raw_data = pd.read_csv("raw.csv")
    predict = pd.read_csv("predict.csv")
    price_train = pd.read_csv("../../data/train_ts_price.csv")
    for i in range(121,181):
        feature = "Ret_%d_pred"%(i)
        raw_data[feature] = predict[feature]
        raw_data.ix[np.abs(raw_data[feature])>0.1,feature] = 0.0
    data = raw_data
    data['Ret_120_price'] = price_train['Ret_120_price']
    transform_format(data)
    Ret_1 = data['Ret_MinusTwo']
    Ret_2 = ((Ret_1+1)*data['Ret_120_price']*data['Ret_180_price'])-1
    data['Ret_PlusOne_pred'] = 0.5*Ret_1 + 0.5*Ret_2
    data['Ret_PlusTwo_pred'] = 0.5*Ret_2 + 0.5*data['Ret_PlusOne_pred']
    data.to_csv("data.csv")
    WMAE_model(data)
    mase = np.sum(data['error'])/(40000*62)
    data.to_csv("data.res.csv")
    print 'loss:%f'%(mase)
    """
    pred = pd.read_csv("data.res.csv")
    value =np.max(pred['Ret_PlusOne_pred'])
    print np.where(pred['Ret_PlusOne_pred']==value)
    """
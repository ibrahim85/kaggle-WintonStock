'''
Created on 2015/11/19

@author: FZY
'''
import pandas as pd
from hyperopt import hp,fmin,tpe,Trials,STATUS_OK
import  rpy2.robjects as robj
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import DataFrame
import time
import numpy as np
from ml_metrics import WMAE_model
model_param = {
    'task':'arima_norml',
    'frequency':hp.choice('frequency',[15]),
    'max_evals':1
               
}
def changeFormat(data,feature):
    return 1.0/np.float((1.0 - data[feature]))
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
def dumpMessage(bestParams,loss,source_name,start,end):
    f = open("../../data/analysis/model/%s_bestParamodel_log.txt"%(source_name),"wb") 
    f.write('loss:%.6f \n'%(loss))
    for(key,value) in bestParams.items():
        f.write("%s:%s\n"%(key,str(value))) 
    f.write("start_time:%s\n"%(start))
    f.write("end_time:%s\n"%(end))
    f.close()
def createModel(data,param):
    print 'Create_model'
    importr('forecast')
    robj.r(
           '''
            arima_data <- function(data){
               
               best_arima = auto.arima(data,trace=F,stepwise=T,max.P=8,max.Q=8,max.p=10,max.q=10,max.order=10,
               ,start.p=1,start.q=0,start.P=1,start.Q=0,seasonal=T,ic=('bic'))
               forecast = forecast.Arima(best_arima,h=60,level=c(99.5),stationary=T)
               output = forecast$mean
               return (output)
               
               
            }
            
   '''     
    )
    print 'the frequency is %d'%(param['frequency'])
    features_names = ["Ret_%d_pred"%(i) for i in range(121,181)]
    predict  = pd.DataFrame(columns=features_names)
    i = 1
    for tmp in DataFrame.iter_row(data):
        if i % 100 == 0:
            print i 
        tmp = robj.r("as.numeric")(tmp)
        #tmp = robj.r('ts')(tmp,start=2)
        tmp = robj.r('ts')(tmp,start=2,frequency=param['frequency'])
        forecast = robj.r('arima_data')(tmp)
        forecast = robj.r('as.numeric')(forecast)
        forecast = np.array(forecast)
        predict2 = pd.DataFrame(forecast).T
        predict2.columns= features_names
        predict = pd.concat([predict,predict2],axis=0)
        i =  i + 1
    
    #this way I will get forecast_data , train_data
    raw_data.to_csv("raw.csv")
    predict.to_csv("predict.csv")
    data = predict.join(raw_data,rsuffix='_2')
    data.to_csv("data.raw.csv")
    data['Ret_120_price'] = price_train['Ret_120_price']
    transform_format(data)
    Ret_1 = data['Ret_MinusTwo']
    Ret_2 = 1 - (1.0/((1.0/(1-data['Ret_MinusOne']))*data['Ret_120_price']*data['Ret_180_price']))
    data['Ret_PlusOne_pred'] = 0.5*Ret_1 + 0.5*Ret_2
    data['Ret_PlusTwo_pred'] = 0.5*Ret_2 + 0.5*data['Ret_PlusOne_pred']
    data.to_csv("data.csv")
    WMAE_model(data)
    mase = np.sum(data['error'])/(40000*62)
    print 'loss:%f'%(mase)
    return {'loss':mase,'status':STATUS_OK}

def predictModel(data,param):
    print 'Create_model'
    importr('forecast')
    robj.r(
           '''
            arima_data <- function(data){
               
               best_arima = auto.arima(data,trace=F,stepwise=T,max.P=8,max.Q=8,max.p=10,max.q=10,max.order=10,
               ,start.p=1,start.q=0,start.P=1,start.Q=0,seasonal=T,ic=('bic'))
               forecast = forecast.Arima(best_arima,h=60,level=c(99.5),stationary=T)
               output = forecast$mean
               return (output)
               
               
            }
            
   '''     
    )
    print 'the frequency is %d'%(param['frequency'])
    features_names = ["Ret_%d_pred"%(i) for i in range(121,181)]
    predict  = pd.DataFrame(columns=features_names)
    i = 1
    for tmp in DataFrame.iter_row(data):
        if i % 100 == 0:
            print i 
        tmp = robj.r("as.numeric")(tmp)
        #tmp = robj.r('ts')(tmp,start=2)
        tmp = robj.r('ts')(tmp,start=2,frequency=param['frequency'])
        forecast = robj.r('arima_data')(tmp)
        forecast = robj.r('as.numeric')(forecast)
        forecast = np.array(forecast)
        predict2 = pd.DataFrame(forecast).T
        predict2.columns= features_names
        predict = pd.concat([predict,predict2],axis=0)
        i =  i + 1
    
    #this way I will get forecast_data , train_data
    raw_data.to_csv("raw.csv")
    predict.to_csv("predict.csv")
    data = predict.join(raw_data,rsuffix='_2')
    data.to_csv("data.raw.csv")
    data['Ret_120_price'] = price_train['Ret_120_price']
    transform_format(data)
    Ret_1 = data['Ret_MinusTwo']
    Ret_2 = 1 - (1.0/((1.0/(1-data['Ret_MinusOne']))*data['Ret_120_price']*data['Ret_180_price']))
    data['Ret_PlusOne_pred'] = 0.5*Ret_1 + 0.5*Ret_2
    data['Ret_PlusTwo_pred'] = 0.5*Ret_2 + 0.5*data['Ret_PlusOne_pred']
    data.to_csv("data.csv")
    WMAE_model(data)
    
    mase = np.sum(data['error'])/(40000*62)
    print 'loss:%f'%(mase)
    
        
#in this model , I will find best model 
if __name__ == '__main__':
    #now we must use ARIMA_model to predict the model 
    #read data for csv file
    print 'read_data'
    price_train = pd.read_csv("../../data/train_ts_price.csv")
    all_data = pd.read_csv("../../data/train_clean_fill_all_1.csv")
    raw_features_names = ["Ret_%d"%(i) for i in range(2,181)]
    raw_features_names.append('Ret_MinusTwo')
    raw_features_names.append('Ret_MinusOne')
    raw_features_names.append('Ret_PlusOne')
    raw_features_names.append('Ret_PlusTwo')
    raw_features_names.append('Weight_Intraday')
    raw_features_names.append('Weight_Daily')
    raw_data = all_data[raw_features_names]
    train = robj.r('read.csv')("../../data/arima_features.csv",sep=',',header=True)
    #train = DataFrame.from_csvfile("../../data/arima_features.csv",as_is=True,header=False)
    importr('forecast')
    ISOTIMEFORMAT='%Y-%m-%d %X'
    trials = Trials()
    start_time = time.strftime( ISOTIMEFORMAT, time.localtime() )
    objective = lambda  x: createModel(train,x)
    best_params = fmin(objective,model_param,trials=trials,algo=tpe.suggest,max_evals=model_param['max_evals'])
    #find best param
    trial_acc = np.asanyarray(trials.losses(), dtype=float)
    best_acc_mean = min(trial_acc)
    end_time = time.strftime(ISOTIMEFORMAT,time.localtime())
    dumpMessage(best_params, best_acc_mean, model_param['task'], start_time, end_time)
    
        
        
        
        
        
        
      
    
    
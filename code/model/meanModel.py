#this model, I will use fmin get best length of move

import pandas as pd
import numpy as np
import time
from hyperopt import Trials,fmin,tpe,hp,STATUS_OK
from ml_metrics import WMAE_model

list_values = []
debug = False
if debug:
    hyperopt_param = {}
    hyperopt_param['lasso_max_evals'] = 2
    hyperopt_param['ridge_max_evals'] = 2
    hyperopt_param['lr_max_evals'] = 2
    hyperopt_param["xgb_max_evals"] = 2
    xgb_min_num_round = 2
    xgb_max_num_round = 10
    xgb_nthread=4
    xgb_num_round_step = 1
else:
    hyperopt_param = {}
    hyperopt_param['ridge_max_evals'] = 400
    hyperopt_param['lasso_max_evals'] = 400
    hyperopt_param['lr_max_evals'] = 400
    hyperopt_param["xgb_max_evals"] = 400
    xgb_min_num_round = 100
    xgb_max_num_round = 250
    xgb_nthread= 8
    xgb_num_round_step = 2


param_raw_mean= {
    'task':'mean_raw_scroll',
    'length':119,
    'rate':0.5,
    #'rate':hp.quniform('rate',0.02,0.5,0.02),
    'max_evals':1 
}

param_price_mean= {
    'task':'mean_price_scroll',
    'length':100,
    'max_evals':1       
}

#in this mode , I will change the format from (p2-p1)/p2 to p2/p1
def changeFormat(data,feature):
    return 1.0/np.float((1.0 - data[feature]))

def toPrice(data,i):
    print type(data)
    if i == 2 :
        print type(data['Ret_2_p'])
        return  data['Ret_2_p']
       
    else:
        return data['Ret_%d_p'%(i-1)]*data['Ret_%d_p'%(i)]
def getWeight(length):
    sum = (length*(length+1))/2
    weight  = {}
    for i in range(1,length+1):
        weight[i] = i/float(sum)
    return weight
def transform_format(data):
    print 'change format'
    for i in range(121,181):
        feature = 'Ret_'+str(i)
        data['Ret_%d_p'%(i)] = list(data.apply(lambda x : changeFormat(x,feature),axis=1))
    
    for i in range(121,181):
        if i == 121:
            data['Ret_121_price'] = data['Ret_121_p']
        else:
            data['Ret_%d_price'%(i)] = data['Ret_%d_price'%(i-1)]*data['Ret_%d_p'%(i)] 
def dumpMessage(bestParams,loss,std,source_name,start,end):
    f = open("../../data/analysis/model/%s_bestParamodel_log.txt"%(source_name),"wb") 
    f.write('loss:%.6f \nStd:%.6f \n'%(loss,std))
    for(key,value) in bestParams.items():
        f.write("%s:%s\n"%(key,str(value))) 
    f.write("start_time:%s\n"%(start))
    f.write("end_time:%s\n"%(end))
    f.close()       
    
# we use the raw data to predict the value      
def rawPredict(param,data):
    length = int(param['length'])
    weight = getWeight(length)
    for i in range(121,181):
        count = 0
        k = 1 
        for j in range(i-length,i):
            count = count + weight[k]*data['Ret_%d'%(j)]
            k = k + 1
        data["Ret_%d_pred"%(i)] = count
    transform_format(data)
    print 'format end'
    Ret_1 = data['Ret_MinusTwo']
    Ret_2 = 1 - (1.0/((1.0/(1-data['Ret_MinusOne']))*data['Ret_120_price']*data['Ret_180_price']))
    data['Ret_PlusOne_pred'] = (1-param['rate'])*Ret_1 + param['rate']*Ret_2
    data['Ret_PlusTwo_pred'] = (1-param['rate'])*Ret_2 + param['rate']*data['Ret_PlusOne_pred']
    WMAE_model(data)
    mase = np.sum(data['error'])/(40000*62)
    variance = 0
    data.to_csv("raw.csv")
    print 'loss:%f'%(mase)
    return {'loss':mase,'attachments':{'std':variance},'status':STATUS_OK}  

#we use the the price data to predict the Ret value

def pricePredict(param,data,raw_data):
    length = int(param['length'])
    weight = getWeight(length)
    for i  in range(121,181):
        count = 0 
        k = 1 
        for j in range(i-length,i):
            count = count + weight[k]*data['Ret_%d_price'%(j)]
            k = k + 1
        data['Ret_%d_price'%(i)] = count
    
    #we need to transform the price to raw message
    for i in range(180,120,-1):
        if i == 121:
            data["Ret_%d_pred"%(i)] = data["Ret_%d_price"%(i)]
        else:
            data["Ret_%d_pred"%(i)] = 1.0 - (1.0/(data['Ret_%d_price'%(i)]/data['Ret_%d_price'%(i-1)]))
    #we need to transform the all Ret value to Price , the we can predict the the last the Ret value 
    Ret_1 = raw_data['Ret_MinusTwo']
    Ret_2 = 1 - (1.0/((1.0/(1-raw_data['Ret_MinusOne']))*data['Ret_120_price']*data['Ret_180_price']))
    #Ret_2 = (1.0/(1-raw_data['Ret_MinusOne']))
    print Ret_2
    data['Ret_PlusOne_pred'] = 0.2*Ret_1 + 0.8*Ret_2
    data['Ret_PlusTwo_pred'] = 0.2*Ret_2 + 0.8*data['Ret_PlusOne_pred']
    data['Weight_Daily'] = raw_data['Weight_Daily']
    data['Weight_Intraday'] = raw_data['Weight_Intraday']
    data['Ret_PlusOne'] = raw_data['Ret_PlusOne']
    data['Ret_PlusTwo'] = raw_data['Ret_PlusTwo']
    for i in range(121,181):
        data['Ret_%d'%(i)] = raw_data['Ret_%d'%(i)]  
    WMAE_model(data)
    mase = np.sum(data['error'])/(40000*62)
    variance = 0
    data.to_csv("price.csv")
    
    print 'loss:%f'%(mase)
    return {'loss':mase,'attachments':{'std':variance},'status':STATUS_OK}      
       
def TunningParamter(param,data,price_data):
    ISOTIMEFORMAT='%Y-%m-%d %X'
    start_time = time.strftime( ISOTIMEFORMAT, time.localtime() )
    trials = Trials()
    if param['task'] == 'mean_raw_scroll':
        objective = lambda p : rawPredict(p,data)
    elif param['task'] == 'mean_price_scroll':
        objective = lambda p : pricePredict(p,price_data,data)
    
    best_params = fmin(objective,param,algo=tpe.suggest,
                          trials=trials, max_evals=param["max_evals"])
    print best_params
    trial_acc = np.asanyarray(trials.losses(), dtype=float)
    best_acc_mean = min(trial_acc)
    ind = np.where(trial_acc==best_acc_mean)[0][0]
    best_loss_std = trials.trial_attachments(trials.trials[ind])['std']
    end_time = time.strftime( ISOTIMEFORMAT, time.localtime())
    dumpMessage(best_params, best_acc_mean,best_loss_std,param['task'],start_time,end_time)
    print ("Best stats")

def listRet(data):
    for i in range(121,181):
        list_values .append(data['Ret_%d'%(i)])
    list_values.append(data['61'])
    list_values.append(data['62'])   
def predictRaw_test(param,data):
    length = 90
    weight = getWeight(length)
    for i in range(121,181):
        count = 0
        k = 1 
        for j in range(i-length,i):
            count = count + weight[k]*data['Ret_%d'%(j)]
            k = k + 1
        data["Ret_%d"%(i)] = count
    transform_format(data)
    Ret_1 = data['Ret_MinusTwo']
    Ret_2 = 1 - (1.0/((1.0/(1-data['Ret_MinusOne']))*data['Ret_120_price']*data['Ret_180_price']))
    data['61'] = 0.2*Ret_1 + 0.8*Ret_2
    data['62'] = 0.2*Ret_2 + 0.8*data['61']
    data.apply(lambda x : listRet(x),axis=1)
    pd.Series(list_values).to_csv("../../data/result.csv")
if __name__ == "__main__":
    #this way, I have two type of data to predict
    #the raw data to predict
    train = pd.read_csv("../../data/train_clean_fill_all_1.csv")
    print train.shape
    price_train = pd.read_csv("../../data/train_ts_price.csv")
    print price_train.shape
    train['Ret_120_price'] = price_train['Ret_120_price']
    TunningParamter(param_raw_mean,train,price_train)
    #the price data to predict
    #TunningParamter(param_price_mean,train,price_train)
 
    
    
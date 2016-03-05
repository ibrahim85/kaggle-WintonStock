#coding=utf-8
'''
Created on 2015/11/25

@author: FZY
'''

#in this model ,
import numpy as np , pandas as pd
from sklearn.linear_model import Lasso,Ridge
from sklearn.cross_validation import train_test_split
#from sklearn.svm import LinearSVR
import xgboost as xgb
Lasso_param = {
    'task':'skl_lasso',
    'alpha':0.24460944908709087,
    'random_state':2015,   
}

Ridge_param= {
    'task':'skl_ridge',
    'alpha':1.65976945576,  
    'random_state':2015           
}

skl_linearSVR_param= {
    'task' : 'skl_linearSVR',
    'epsilon':0.0001,
    'C' : 8.0,
    'loss':'squared_epsilon_insensitive',
    'seed':2015,
    'dual':False,
}

xgb_RetPulsOne_param = {
    'booster': 'gblinear',
    'objective': 'reg:linear',
    'lambda_bias':0.2,
    'alpha':0.0,
    'lambda':0.1, 
    'num_round':300,
    'eta':0.7,
    'silent': 1,
    'seed': 2015,
    "max_evals":1,      
}

xgb_RetPulsTwo_param = {
    'booster': 'gblinear',
    'objective': 'reg:linear',
    'lambda_bias':1.5,
    'alpha':0.0,
    'lambda':0.1, 
    'num_round':300,
    'eta':0.5,
    'silent': 1,
    'seed': 2015,
    "max_evals":1,       
}
xgb_tree_RetPulsOne_param = {
    'booster': 'gbtree',
    'objective': 'reg:linear',
    'colsample_bytree':0.6,
    'min_child_weight':1.0,
    'eta':0.8,
    'num_round':80,
    'max_depth':12.0,
    'gamma':4.8,
    'nthread': 4,
    'silent' : 1,
    'seed': 2015,
    "max_evals":1,      
}

xgb_tree_RetPulsTwo_param = {
    'booster': 'gbtree',
    'objective': 'reg:linear',
    'colsample_bytree':0.9,
    'min_child_weight':3.0,
    'eta':0.8,
    'num_round':80,
    'max_depth':14,
    'gamma':5,
    'nthread': 4,
    'silent' : 1,
    'seed': 2015,
    "max_evals":1,       
}
def getRet(data,list_values):
    for i in range(121,181):
        list_values.append(0)
    value1 = float(data['1']) 
    list_values.append(value1)
    value2 = float(data['2'])
    list_values.append(value2)   
def listRet(data,list_values):
    data.apply(lambda x : getRet(x, list_values),axis=1)
    return list_values
    
    
"""
def predictValueByLinearSVM(param,rate):
    print 'read data'
    train_price = pd.read_csv("../../data/train_ts_price.csv")
    train = pd.read_csv("../../data/train_clean_fill_all_2.csv")
    test_price = pd.read_csv("../../data/test_ts_price.csv")
    test = pd.read_csv("../../data/test_clean_fill_all_2.csv")
    #add price message to train
    features = ["Ret_%d_price"%(i) for i in range(2,121)]
    train[features] = train_price[features]
    test[features] = test_price[features]
    all_features= []
    #add the feature to predict
    #features = [ "Feature_%d"%(i) for i in range(1,26)]
    #all_features.extend(features)
    features =["Ret_%d"%(i) for i in range(2,121)]
    all_features.extend(features)
    features = ["Ret_%d_price"%(i) for i in range(2,121)]
    all_features.extend(features)
    #predict the value
    all_features.append('Ret_PlusOne')
    all_features.append('Ret_PlusTwo')
    all_features.append('Ret_PlusOne')
    all_features.append('Ret_PlusTwo')
    all_features.append('Ret_total_price')
    all_features.append('Ret_mean')
    all_features.append('Ret_mean_price')
    all_features.append('Ret_max_price')
    all_features.append('Ret_max')
    all_features.append('Ret_min_price')
    all_features.append('Ret_min')
    all_features.append('Ret_var')
    all_features.append('Ret_var_price')
    #now we need to predict the value 
    
    #now we need to predict the value 
    pred_all_label = ['Ret_PlusOne','Ret_PlusTwo']
    train_feature = list(set(all_features)-set(pred_all_label))
    #set the train data
    
    #guess the data have too many data same ,so I use the cv 
    k=300
    for i in range(0,k):
        print i 
        X_train,X_test,Y_train,Y_test = train_test_split(train[train_feature],train[pred_all_label],test_size=0.3,random_state=i)
        svr = LinearSVR(C=param['C'],epsilon=param['epsilon'],dual=param['dual'],loss=param['loss'],random_state=param['seed'])
        svr.fit(X_train,Y_train['Ret_PlusOne'])
        pred_value1 = svr.predict(test[train_feature])
        svr.fit(X_train,Y_train['Ret_PlusTwo'])
        pred_value2 = svr.predict(test[train_feature])
        if i == 0:
            df = pd.DataFrame({'1':pred_value1,'2':pred_value2})
        else:
            df['1'] = df['1']+pred_value1
            df['2'] = df['2']+pred_value2
    for i in rate:
        tp = df.copy()
        list_values = []
        tp['1'] = df['1']/(k*i)
        tp['2'] = df['2']/(k*i)
        tp.to_csv("../../data/result/RetPlus/svm.pred_%f.csv"%(i),columns=['1','2'])
        list_v = listRet(tp,list_values)
        pd.Series(list_v).to_csv("../../data/result/regression_svm_ret_%f.csv"%(i))
"""
def predictValueByLasso(param):
    train_price = pd.read_csv("../../data/train_ts_price.csv")
    train = pd.read_csv("../../data/train_clean_fill_all_1.csv")
    test_price = pd.read_csv("../../data/test_ts_price.csv")
    test = pd.read_csv("../../data/test_clean_fill_all_1.csv")
    #add price message to train
    features = ["Ret_%d_price"%(i) for i in range(2,121)]
    train[features] = train_price[features]
    test[features] = test_price[features]
    #add the feature to predict
    all_features= []
    #add the feature to predict
    
    features = [ "Feature_%d"%(i) for i in range(1,26)]
    all_features.extend(features)
    features =["Ret_%d"%(i) for i in range(2,121)]
    all_features.extend(features)
    features = ["Ret_%d_price"%(i) for i in range(2,121)]
    all_features.extend(features)
    #predict the value
    all_features.append('Ret_PlusOne')
    all_features.append('Ret_PlusTwo')
    all_features.append('Ret_total_price')
    all_features.append('Ret_mean')
    all_features.append('Ret_mean_price')
    all_features.append('Ret_max_price')
    all_features.append('Ret_max')
    all_features.append('Ret_min_price')
    all_features.append('Ret_min')
    all_features.append('Ret_var')
    all_features.append('Ret_var_price')
    #now we need to predict the value 
    pred_all_label = ['Ret_PlusOne','Ret_PlusTwo']
    train_feature = list(set(all_features)-set(pred_all_label))
    #set the train data
    X_train = train[train_feature]
    Y_train = train[pred_all_label]
    X_test =  test[train_feature]
    lasso = Lasso(alpha=param['alpha'],normalize=True)
    lasso.fit(X_train,Y_train)
    pred_value = lasso.predict(X_test)
    print lasso.coef_
    pred_value = pd.DataFrame(pred_value,columns=['1','2'])
    pd.DataFrame(pred_value).to_csv("../../data/result/RetPlus/lasso.pred.csv",columns=['1','2'])
    pred_value = pred_value.apply(lambda x : listRet(x),axis=1)
    #pd.Series(list_values).to_csv("../../data/result/regression_lasso_ret.csv")
    
  
def predictValueByRidge(param,rate):
    print 'read data'
    train_price = pd.read_csv("../../data/train_ts_price.csv")
    train = pd.read_csv("../../data/train_clean_fill_all_2.csv")
    test_price = pd.read_csv("../../data/test_ts_price.csv")
    test = pd.read_csv("../../data/test_clean_fill_all_2.csv")
    #add price message to train
    features = ["Ret_%d_price"%(i) for i in range(2,121)]
    train[features] = train_price[features]
    test[features] = test_price[features]
    all_features= []
    #add the feature to predict
    features = [ "Feature_%d"%(i) for i in range(1,26)]
    all_features.extend(features)
    features =["Ret_%d"%(i) for i in range(2,121)]
    all_features.extend(features)
    features = ["Ret_%d_price"%(i) for i in range(2,121)]
    all_features.extend(features)
    #predict the value
    all_features.append('Ret_PlusOne')
    all_features.append('Ret_PlusTwo')
    all_features.append('Ret_PlusOne')
    all_features.append('Ret_PlusTwo')
    all_features.append('Ret_total_price')
    all_features.append('Ret_mean')
    all_features.append('Ret_mean_price')
    all_features.append('Ret_max_price')
    all_features.append('Ret_max')
    all_features.append('Ret_min_price')
    all_features.append('Ret_min')
    all_features.append('Ret_var')
    all_features.append('Ret_var_price')
    #now we need to predict the value 
    
    #now we need to predict the value 
    pred_all_label = ['Ret_PlusOne','Ret_PlusTwo']
    train_feature = list(set(all_features)-set(pred_all_label))
    #set the train data
    
    #guess the data have too many data same ,so I use the cv 
    k=1
    for i in range(0,k):
        print i 
        X_train,X_test,Y_train,Y_test = train_test_split(train[train_feature],train[pred_all_label],test_size=0.3,random_state=i)
        ridge = Ridge(alpha=param['alpha'],normalize=True)
        ridge.fit(X_train,Y_train)
        pred_value = ridge.predict(test[train_feature])
        if i == 0:
            df = pd.DataFrame(pred_value,columns=['1','2'])
        else:
            tmp = pd.DataFrame(pred_value,columns=['1','2'])
            df['1'] = df['1']+tmp['1']
            df['2'] = df['2']+tmp['2']
    for i in rate:
        tp = df.copy()
        list_values = []
        tp['1'] = df['1']/(k*i)
        tp['2'] = df['2']/(k*i)
        tp.to_csv("../../data/result/RetPlus/ridge.pred_%f.csv"%(i),columns=['1','2'])
        list_v = listRet(tp,list_values)
        pd.Series(list_v).to_csv("../../data/result/regression_ridge_ret_%f.csv"%(i))
        
def predictValueByXGboost(param1,param2,rate):
    print 'read data'
    train_price = pd.read_csv("../../data/train_ts_price.csv")
    train = pd.read_csv("../../data/train_clean_fill_all_2.csv")
    test_price = pd.read_csv("../../data/test_ts_price.csv")
    test = pd.read_csv("../../data/test_clean_fill_all_2.csv")
    #add price message to train
    features = ["Ret_%d_price"%(i) for i in range(2,121)]
    train[features] = train_price[features]
    test[features] = test_price[features]
    all_features= []
    #add the feature to predict
    features = [ "Feature_%d"%(i) for i in range(1,26)]
    all_features.extend(features)
    features =["Ret_%d"%(i) for i in range(2,121)]
    all_features.extend(features)
    features = ["Ret_%d_price"%(i) for i in range(2,121)]
    all_features.extend(features)
    #predict the value
    all_features.append('Ret_PlusOne')
    all_features.append('Ret_PlusTwo')
    all_features.append('Ret_PlusOne')
    all_features.append('Ret_PlusTwo')
    all_features.append('Ret_total_price')
    all_features.append('Ret_mean')
    all_features.append('Ret_mean_price')
    all_features.append('Ret_max_price')
    all_features.append('Ret_max')
    all_features.append('Ret_min_price')
    all_features.append('Ret_min')
    all_features.append('Ret_var')
    all_features.append('Ret_var_price')
    #now we need to predict the value 
    
    #now we need to predict the value 
    pred_all_label = ['Ret_PlusOne','Ret_PlusTwo']
    train_feature = list(set(all_features)-set(pred_all_label))
    #set the train data
    
    #guess the data have too many data same ,so I use the cv 
    k=1
    for i in range(0,k):
        print i 
        X_train_1,X_test_1,Y_train_1,Y_test_1 = train_test_split(train[train_feature],train['Ret_PlusOne'],test_size=0.1,random_state=i)
        X_train_2,X_test_2,Y_train_2,Y_test_2 = train_test_split(train[train_feature],train['Ret_PlusTwo'],test_size=0.1,random_state=i)
        train_data_one = xgb.DMatrix(X_train_1,label=Y_train_1)
        train_data_two = xgb.DMatrix(X_train_2,label=Y_train_2)
        watchlist_1 = [(train_data_one,'train')]
        watchlist_2 = [(train_data_two,'train')]
        bst1 = xgb.train(param1,train_data_one,int(param1['num_round']),watchlist_1)
        bst2 = xgb.train(param2,train_data_two,int(param2['num_round']),watchlist_2)
        pred_value_1 = bst1.predict(xgb.DMatrix(test[train_feature]))
        pred_value_2 = bst2.predict(xgb.DMatrix(test[train_feature]))
        if i == 0:
            df = pd.DataFrame({'1':pred_value_1,'2':pred_value_2})
        else:
            tmp = pd.DataFrame({'1':pred_value_1,'2':pred_value_2})
            df['1'] = df['1']+tmp['1']
            df['2'] = df['2']+tmp['2']
    for i in rate:
        tp = df.copy()
        list_values = []
        tp['1'] = df['1']/(k*i)
        tp['2'] = df['2']/(k*i)
        tp.to_csv("../../data/result/RetPlus/xgb.pred_%f.csv"%(i),columns=['1','2'])
        list_v = listRet(tp,list_values)
        pd.Series(list_v).to_csv("../../data/result/regression_xgb_ret_%f.csv"%(i))       

if __name__ == '__main__':
    print 'train_model'
    #predictValueByLasso(Lasso_param)
    rate = [6.1,6.0,7.0,5.0,4.0,3.0,8.0,9.0,10.0]
    #predictValueByRidge(Ridge_param,rate)
    #predictValueByLinearSVM(skl_linearSVR_param,rate)
    predictValueByXGboost(xgb_tree_RetPulsOne_param, xgb_tree_RetPulsTwo_param, rate)
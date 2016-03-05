'''
Created on 2015/11/4

@author: FZY
'''
import random
from sklearn.linear_model import Ridge,Lasso,LogisticRegression
from hyperopt import hp
import numpy as np
from hyperopt.pyll_utils import hyperopt_param
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error,explained_variance_score,mean_absolute_error,auc
from sklearn.metrics import average_precision_score
from hyperopt import Trials,tpe,fmin
from hyperopt.base import STATUS_OK
from ml_metrics import accuracy_model
from sklearn.datasets import dump_svmlight_file
import os
import xgboost as xgb
import time
from CVModel import loadCVIndex
#fit the  param 
debug = False
xgb_random_seed = 2015

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
    xgb_num_round_step = 20
    
Ridge_param= {
    'task':'skl_ridge',
    'alpha': hp.loguniform("alpha", np.log(0.001), np.log(40)),
    'alpha': 0,
    'random_state':2015,
    'max_evals':hyperopt_param['ridge_max_evals']           
}

Lasso_param = {
    'task':'skl_lasso',
    'alpha': hp.loguniform("alpha", np.log(0.00001), np.log(np.exp(1))),
    'random_state':2015,
    'max_evals':hyperopt_param['lasso_max_evals']       
}

xgb_regression_param = {
    'task': 'regression',
    'booster': 'gblinear',
    'objective': 'reg:linear',
    'eta' : hp.quniform('eta', 0.1, 1, 0.1),
    'lambda' : hp.quniform('lambda', 0, 5, 0.1),
    'alpha' : hp.quniform('alpha', 0, 1, 0.1),
    'lambda_bias' : hp.quniform('lambda_bias', 0, 3, 0.1),
    'num_round' : hp.quniform('num_round', xgb_min_num_round, xgb_max_num_round, xgb_num_round_step),
    'nthread': xgb_nthread,
    'silent' : 1,
    'seed': xgb_random_seed,
    "max_evals": hyperopt_param["xgb_max_evals"],                    
}

xgb_regression_param_by_tree = {
    'task': 'regression',
    'booster': 'gbtree',
    'objective': 'reg:linear',
    'eta' : hp.quniform('eta', 0.1, 1, 0.1),
    'lambda' : hp.quniform('lambda', 0, 5, 0.1),
    'alpha' : hp.quniform('alpha', 0, 1, 0.01),
    'lambda_bias' : hp.quniform('lambda_bias', 0, 3, 0.1),
    'num_round' : hp.quniform('num_round', xgb_min_num_round, xgb_max_num_round, xgb_num_round_step),
    'nthread': xgb_nthread,
    'silent' : 1,
    'seed': xgb_random_seed,
    "max_evals": hyperopt_param["xgb_max_evals"],                    
}

xgb_tree_param = {
    'task': 'class',
    'booster': 'gbtree',
    'objective': 'multi:softmax',
    'eta' : hp.quniform('eta', 0.1, 1, 0.1),
    'gamma': hp.quniform('gamma',0.1,2,0.1),
    'num_round' : hp.quniform('num_round', xgb_min_num_round, xgb_max_num_round, xgb_num_round_step),
    'nthread': xgb_nthread,
    'silent' : 1,
    'seed': xgb_random_seed,
    "max_evals": hyperopt_param["xgb_max_evals"], 
    "num_class":  9,    
    'max_depth': hp.quniform('max_depth', 6, 12, 1),             
}

skl_lr_param = {
    'task' : 'skl_lr',
    'C' : hp.quniform('C',1,20,0.1),
    'seed':xgb_random_seed,
    'max_evals':hyperopt_param['lr_max_evals']
}



def dumpMessage(bestParams,loss,loss_std,f_name,start,end,feature):
     
    f = open("../../data/feature/weight/%s_%s_bestParamodel_log.txt"%(f_name,feature),"wb") 
    f.write('loss:%.6f \nStd:%.6f \n'%(loss,loss_std))
    for(key,value) in bestParams.items():
        f.write("%s:%s\n"%(key,str(value))) 
    f.write("start_time:%s\n"%(start))
    f.write("end_time:%s\n"%(end))
    f.close()  
def trainModel(param,data,features,feature):
    #we just judge our model
    #so we do not use bagging ,just one loop of CV
   
    pred_all_label = ['Weight_Intraday','Weight_Daily']
    train_feature = list(set(features)-set(pred_all_label))
    pred_label = feature
    #create CV
    err_cv = []
    std_cv = []
    for run in range(0,2):
        print "this is run:%d"%(run+1)
        train_index = loadCVIndex("../../data/cv/train.run%d.txt"%(run+1))
        test_index = loadCVIndex("../../data/cv/valid.run%d.txt"%(run+1))
        X_train = data.iloc[train_index][train_feature]
        X_test = data.iloc[test_index][train_feature]
        Y_train = data.iloc[train_index][pred_label]
        Y_test = data.iloc[test_index][pred_label]
        if param['task'] == 'skl_ridge': 
            ridge = Ridge(alpha=param['alpha'],normalize=True)
            ridge.fit(X_train,Y_train)
            pred_value = ridge.predict(X_test)
            error_train = mean_absolute_error(Y_test,pred_value)
            variance = explained_variance_score(Y_test,pred_value)
            err_cv.append(error_train)
            std_cv.append(variance)
           
            
        elif param['task'] == 'skl_lasso':
            lasso = Lasso(alpha=param['alpha'],normalize=True)
            lasso.fit(X_train,Y_train)
            pred_value = lasso.predict(X_test)
            error_train = mean_absolute_error(Y_test,pred_value)
            print error_train
            variance = 0
            err_cv.append(error_train)
            std_cv.append(variance)
            print pred_value
            print Y_test
        elif param['task'] == 'skl_lr':
            clf = LogisticRegression(C=param['C'])
            clf.fit(X_train,Y_train)
            pred_value = clf.predict(X_test)
            error_train = 1 - accuracy_model(pred_value, Y_test)
            variance = error_train
            err_cv.append(error_train)
            std_cv.append(variance)
            
        elif param['task'] == 'regression':
            train_data = xgb.DMatrix(X_train,label=np.array(Y_train))
            valid_data = xgb.DMatrix(X_test,label=np.array(Y_test))
            watchlist = [(train_data,'train'),(valid_data,'valid')]
            bst = xgb.train(param, train_data, int(param['num_round']),watchlist)
            valid_data = xgb.DMatrix(X_test)
            pred_value = bst.predict(valid_data)
            error_train = mean_absolute_error(Y_test,pred_value)
            variance = 0
            err_cv.append(error_train)
            std_cv.append(variance)
            print error_train
        elif param['task'] == 'class':

            train_data = xgb.DMatrix(X_train,label=Y_train)
            valid_data = xgb.DMatrix(X_test,label=Y_test)
            watchlist = [(train_data,'train'),(valid_data,'valid')]
            bst = xgb.train(param, train_data, int(param['num_round']),watchlist)
            valid_data = xgb.DMatrix(X_test)
            pred_value = bst.predict(valid_data)
            error_train = 1 - accuracy_model(pred_value, Y_test)
            variance = 0
            err_cv.append(error_train)
            std_cv.append(variance)
            print error_train
    #print "error.train:%f error.test:%f"%(error_train,error)
    error = np.mean(err_cv)
    std_cv = np.mean(err_cv)
    print "error:%f"%(error)
    return {'loss':error,'attachments':{'std':variance},'status':STATUS_OK}


def TunningParamter(param,data,features,feature):
    ISOTIMEFORMAT='%Y-%m-%d %X'
    start = time.strftime(ISOTIMEFORMAT, time.localtime())
    trials = Trials()
    objective = lambda p : trainModel(p, data, features,feature)
    best_parameters = fmin(objective, param, algo =tpe.suggest,max_evals=param['max_evals'],trials= trials)
    #now we need to get best_param
    trials_loss = np.asanyarray(trials.losses(),dtype=float)
    best_loss = min(trials_loss)
    ind = np.where(trials_loss==best_loss)[0][0]
    best_loss_std = trials.trial_attachments(trials.trials[ind])['std']
    end = time.strftime(ISOTIMEFORMAT,time.localtime())
    dumpMessage(best_parameters, best_loss, best_loss_std,param['task'],start,end,feature)
    
    
















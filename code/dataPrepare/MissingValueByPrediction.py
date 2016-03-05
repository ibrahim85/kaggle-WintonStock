'''
Created on 2015/11/4

@author: FZY
'''
import random
import pandas as pd 
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



def dumpMessage(bestParams,loss,loss_std,f_name,source_name,start,end):
     
    f = open("../../data/analysis/na/%s_%s_bestParamodel_log.txt"%(f_name,source_name),"wb") 
    f.write('loss:%.6f \nStd:%.6f \n'%(loss,loss_std))
    for(key,value) in bestParams.items():
        f.write("%s:%s\n"%(key,str(value))) 
    f.write("start_time:%s\n"%(start))
    f.write("end_time:%s\n"%(end))
    f.close()  
def trainModel(param,data,features,feature,source_name):
    #we juet judge our model
    #so we do not use bagging ,just one loop of CV
   
    seed = random.randrange(1,200)
    train_feature = list(features)
    train_feature.remove(feature)
    #create CV
    X_train,X_test,Y_train,Y_test = train_test_split(data[train_feature],data[feature],test_size=0.3)
    if param['task'] == 'skl_ridge':
        
        ridge = Ridge(alpha=param['alpha'],normalize=True)
        ridge.fit(X_train,Y_train)
        pred_value = ridge.predict(X_test)
        error = mean_absolute_error(Y_test,pred_value)
        pred_train = ridge.predict(X_train)
        variance = explained_variance_score(Y_test,pred_value)
        error_train = mean_absolute_error(Y_train,pred_train)
        
    elif param['task'] == 'skl_lasso':
        lasso = Lasso(alpha=param['alpha'],normalize=True)
        lasso.fit(X_train,Y_train)
        pred_value = lasso.predict(X_test)
        pred_train = lasso.predict(X_train)
        error = mean_absolute_error(Y_test,pred_value)
        variance = explained_variance_score(Y_test,pred_value)
        error_train = mean_absolute_error(Y_train,pred_train)
        #load data
    
    elif param['task'] == 'skl_lr':
        clf = LogisticRegression(C=param['C'])
        clf.fit(X_train,Y_train)
        pred_value = clf.predict(X_test)
        pred_train = clf.predict(X_train)
        error = 1 - accuracy_model(pred_value, Y_test)
        error_train =1 - accuracy_model(pred_train, Y_train)
        variance = error_train
        
    elif param['task'] == 'regression':
        if not os.path.exists("../../data/analysis/svm/train.%s.svm.txt"%(source_name)):
            dump_svmlight_file(X_train,Y_train,"../../data/analysis/svm/train.%s.svm.txt"%(source_name))
            dump_svmlight_file(X_test,Y_test,"../../data/analysis/svm/valid.%s.svm.txt"%(source_name))
        #train_data = xgb.Matrix(X_train,Y_train)
        #train_data = xgb.Matrix(X_test,Y_test)
        train_data = xgb.DMatrix("../../data/analysis/svm/train.%s.svm.txt"%(source_name))
        valid_data = xgb.DMatrix("../../data/analysis/svm/valid.%s.svm.txt"%(source_name))
        watchlist = [(train_data,'train'),(valid_data,'valid')]
        bst = xgb.train(param, train_data, int(param['num_round']),watchlist)
        valid_data = xgb.DMatrix(X_test)
        pred_value = bst.predict(valid_data)
        error = mean_absolute_error(Y_test,pred_value)
        variance = 0
    
    elif param['task'] == 'class':
        """
        if not os.path.exists("../../data/analysis/svm/train.%s.svm.txt"%(source_name)):
            dump_svmlight_file(X_train,Y_train,"../../data/analysis/svm/train.%s.svm.txt"%(source_name))
            dump_svmlight_file(X_test,Y_test,"../../data/analysis/svm/valid.%s.svm.txt"%(source_name))
        train_data = xgb.DMatrix("../../data/analysis/svm/train.%s.svm.txt"%(source_name))
        valid_data = xgb.DMatrix("../../data/analysis/svm/valid.%s.svm.txt"%(source_name))
        """
        train_data = xgb.DMatrix(X_train,label=Y_train)
        valid_data = xgb.DMatrix(X_test,label=Y_test)
        watchlist = [(train_data,'train'),(valid_data,'valid')]
        bst = xgb.train(param, train_data, int(param['num_round']),watchlist)
        valid_data = xgb.DMatrix(X_test)
        pred_value = bst.predict(valid_data)
        error = 1 - accuracy_model(pred_value, Y_test)
        variance = 0
    #print "error.train:%f error.test:%f"%(error_train,error)
    print "error:%f"%(error)
    return {'loss':error,'attachments':{'std':variance},'status':STATUS_OK}


def TunningParamter(param,data,features,feature,source_name):
    ISOTIMEFORMAT='%Y-%m-%d %X'
    start = time.strftime(ISOTIMEFORMAT, time.localtime())
    trials = Trials()
    objective = lambda p : trainModel(p, data, features, feature,source_name)
    
    best_parameters = fmin(objective, param, algo =tpe.suggest,max_evals=param['max_evals'],trials= trials)
    #now we need to get best_param
    trials_loss = np.asanyarray(trials.losses(),dtype=float)
    best_loss = min(trials_loss)
    ind = np.where(trials_loss==best_loss)[0][0]
    best_loss_std = trials.trial_attachments(trials.trials[ind])['std']
    end = time.strftime(ISOTIMEFORMAT,time.localtime())
    dumpMessage(best_parameters, best_loss, best_loss_std,param['task'],source_name,start,end)
    
    
if __name__ == "__main__":
    #at first , we could rbind the train,test data 
    train = pd.read_csv("../../data/train.csv")
    #test = pd.read_csv("../../data/test.csv")
    """
    all_data = pd.concat([train,test])
    print all_data
    cols = list(test.columns)
    cols.remove('Id')
    #all_data = 
    """
    















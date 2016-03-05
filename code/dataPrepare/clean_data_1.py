'''
Created on 2015/11/6

@author: FZY
'''
"""
this model is deal with the feature_20,feature_10,feature_1,feature_2,feature_4
"""
import pandas as pd 
from hyperopt import hp
import numpy as np
import random
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from cleandata import fillNaByMean
feature_10_xgb_tree_param = {
    'task': 'class',
    'booster': 'gbtree',
    'objective': 'multi:softmax',
    'eta' : 0.6,
    'gamma': 0.1,
    'num_round' :220,
    'nthread': 8,
    'silent' : 1,
    'seed': 2015, 
    "num_class":  6,                 
}

feature_1_xgb_tree_param = {
    'task': 'class',
    'booster': 'gbtree',
    'objective': 'multi:softmax',
    'eta' : 0.2,
    'gamma': 0.1,
    'num_round' :250,
    'nthread': 64,
    'silent' : 1,
    'seed': 2015, 
    "num_class":  10, 
    "max_depth":  9              
}
feature_4_xgb_regression_param = {
    'task': 'regression',
    'booster': 'gbtree',
    'objective': 'reg:linear',
    'eta' : 0.1,
    'lambda': 1.0,
    'num_round' :410,
    'nthread': 64,
    'silent' : 1,
    'seed': 2015, 
    "max_depth":  11,
    "alpha":0.7,
    "min_child_weight":2          
}

feature_20_xgb_tree_param = {
    'task': 'class',
    'booster': 'gbtree',
    'objective': 'multi:softmax',
    'eta' : 0.1,
    'gamma': 0.1,
    'num_round' :320,
    'nthread': 64,
    'silent' : 1,
    'seed': 2015, 
    "num_class":  9,  
    "max_depth":14,               
}

if __name__ == '__main__':
    #from the class I have got 93% acc for feaure-10
    #now I will fill the na value by the model
    #using class model,the acc of  feature is 82%,so I will use
    #predict value to 
    """
    print 'load data'
    train = pd.read_csv("../../data/train_clean_s1.csv")
    test = pd.read_csv("../../data/test_clean_s1.csv")
    train_use = train[~ np.isnan(train['Feature_10'])]
    print train_use.shape
    test_use  = test[~ np.isnan(test['Feature_10'])]
    features = []
    for i in range(2,121):
        feature = "Ret_"+ str(i)
        features.append(feature)
    for i in range(1,26):
        feature = "Feature_" + str(i)
        features.append(feature)
    
    nullFeatures = ['Feature_20','Feature_1','Feature_2','Feature_4']
    features = list(set(features) -set(nullFeatures))
    features = features + ['Ret_MinusTwo','Ret_MinusOne']
    pieces = [train_use[features],test_use[features]]
    data_combine = pd.concat(pieces)
    data_combine['Feature_10'] = data_combine['Feature_10'] - 1
    seed = random.randrange(1,200)
    train_feature = list(features)
    train_feature.remove('Feature_10')
    train_data = xgb.DMatrix(data_combine[train_feature],label=data_combine['Feature_10'])
    watchlist = [(train_data,'train')]
    model = xgb.train(feature_10_xgb_tree_param, train_data,feature_10_xgb_tree_param['num_round'],watchlist)
    #get predict_data
    train_pred_data = train[np.isnan(train['Feature_10'])][train_feature]
    test_pred_data = test[np.isnan(test['Feature_10'])][train_feature]
    train_pred_data = xgb.DMatrix(train_pred_data)
    test_pred_data = xgb.DMatrix(test_pred_data)
    train_pred = model.predict(train_pred_data)
    test_pred = model.predict(test_pred_data)
    train['Feature_10'][np.isnan(train['Feature_10'])] = train_pred + 1
    test['Feature_10'][np.isnan(test['Feature_10'])] = test_pred + 1
    print train['Feature_10']
    train.to_csv("../../data/train_clean_fill_F10.csv")
    test.to_csv("../../data/test_clean_fill_F10.csv")
    
    
    train = pd.read_csv("../../data/train_clean_fill_F10.csv")
    test = pd.read_csv("../../data/test_clean_fill_F10.csv")
    train_use = train[~ np.isnan(train['Feature_1'])]
    print train_use.shape
    test_use  = test[~ np.isnan(test['Feature_1'])]
    features = []
    for i in range(2,121):
        feature = "Ret_"+ str(i)
        features.append(feature)
    for i in range(1,26):
        feature = "Feature_" + str(i)
        features.append(feature)
    nullFeatures = ['Feature_20','Feature_2','Feature_4']
    features = list(set(features) -set(nullFeatures))
    features = features + ['Ret_MinusTwo','Ret_MinusOne']
    pieces = [train_use[features],test_use[features]]
    data_combine = pd.concat(pieces)
    data_combine['Feature_1'] = data_combine['Feature_1'] - 1
    seed = random.randrange(1,200)
    train_feature = list(features)
    train_feature.remove('Feature_1')
    train_data = xgb.DMatrix(data_combine[train_feature],label=data_combine['Feature_1'])
    watchlist = [(train_data,'train')]
    model = xgb.train(feature_1_xgb_tree_param, train_data,feature_1_xgb_tree_param['num_round'],watchlist)
    #get predict_data
    train_pred_data = train[np.isnan(train['Feature_1'])][train_feature]
    test_pred_data = test[np.isnan(test['Feature_1'])][train_feature]
    train_pred_data = xgb.DMatrix(train_pred_data)
    test_pred_data = xgb.DMatrix(test_pred_data)
    train_pred = model.predict(train_pred_data)
    test_pred = model.predict(test_pred_data)
    train['Feature_1'][np.isnan(train['Feature_1'])] = train_pred + 1
    test['Feature_1'][np.isnan(test['Feature_1'])] = test_pred + 1
    print train['Feature_1']
    train.to_csv("../../data/train_clean_fill_F10_F1.csv")
    test.to_csv("../../data/test_clean_fill_F10_F1.csv")
    """
    #now we need to fill the na of Feature_4
    """
    train = pd.read_csv("../../data/train_clean_fill_F10_F1.csv")
    test = pd.read_csv("../../data/test_clean_fill_F10_F1.csv")
    train_use = train[~ np.isnan(train['Feature_4'])]
    print train_use.shape
    test_use  = test[~ np.isnan(test['Feature_4'])]
    features = []
    for i in range(2,121):
        feature = "Ret_"+ str(i)
        features.append(feature)
    for i in range(1,26):
        feature = "Feature_" + str(i)
        features.append(feature)
    nullFeatures = ['Feature_20','Feature_2']
    features = list(set(features) -set(nullFeatures))
    features = features + ['Ret_MinusTwo','Ret_MinusOne']
    pieces = [train_use[features],test_use[features]]
    data_combine = pd.concat(pieces)
    seed = random.randrange(1,200)
    train_feature = list(features)
    train_feature.remove('Feature_4')
    train_data = xgb.DMatrix(data_combine[train_feature],label=data_combine['Feature_4'])
    watchlist = [(train_data,'train')]
    model = xgb.train(feature_4_xgb_regression_param, train_data,feature_4_xgb_regression_param['num_round'],watchlist)
    #get predict_data
    train_pred_data = train[np.isnan(train['Feature_4'])][train_feature]
    test_pred_data = test[np.isnan(test['Feature_4'])][train_feature]
    train_pred_data = xgb.DMatrix(train_pred_data)
    test_pred_data = xgb.DMatrix(test_pred_data)
    train_pred = model.predict(train_pred_data)
    test_pred = model.predict(test_pred_data)
    train.to_csv("../../data/train_clean_fill_F10_F1_F4.csv")
    test.to_csv("../../data/test_clean_fill_F10_F1_F4.csv")
    """
    
    #now we need to get fill the feature_20
    #we use the class prediction to predict the feature
    #the accuracy to classifition is 63%
    """
    train = pd.read_csv("../../data/train_clean_fill_F10_F1_F4.csv")
    test = pd.read_csv("../../data/test_clean_fill_F10_F1_F4.csv")
    train_use = train[~ np.isnan(train['Feature_20'])]
    print train_use.shape
    test_use  = test[~ np.isnan(test['Feature_20'])]
    features = []
    for i in range(2,121):
        feature = "Ret_"+ str(i)
        features.append(feature)
    for i in range(1,26):
        feature = "Feature_" + str(i)
        features.append(feature)
    nullFeatures = ['Feature_2']
    features = list(set(features) -set(nullFeatures))
    features = features + ['Ret_MinusTwo','Ret_MinusOne']
    pieces = [train_use[features],test_use[features]]
    data_combine = pd.concat(pieces)
    data_combine['Feature_20'] = data_combine['Feature_20'] - 2
    seed = random.randrange(1,200)
    train_feature = list(features)
    train_feature.remove('Feature_20')
    train_data = xgb.DMatrix(data_combine[train_feature],label=data_combine['Feature_20'])
    watchlist = [(train_data,'train')]
    model = xgb.train(feature_20_xgb_tree_param, train_data,feature_20_xgb_tree_param['num_round'],watchlist)
    #get predict_data
    train_pred_data = train[np.isnan(train['Feature_20'])][train_feature]
    test_pred_data = test[np.isnan(test['Feature_20'])][train_feature]
    train_pred_data = xgb.DMatrix(train_pred_data)
    test_pred_data = xgb.DMatrix(test_pred_data)
    train_pred = model.predict(train_pred_data)
    test_pred = model.predict(test_pred_data)
    train['Feature_20'][np.isnan(train['Feature_20'])] = train_pred + 2
    test['Feature_20'][np.isnan(test['Feature_20'])] = test_pred + 2
    train.to_csv("../../data/train_clean_fill_F10_F1_F4_F20.csv")
    test.to_csv("../../data/test_clean_fill_F10_F1_F4_F20.csv")
    """
    
    ##for the feature 2,the result of prediction,I will use median to fill the na value
    ## 
    """
    train = pd.read_csv("../../data/train_clean_fill_F10_F1_F4_F20.csv")
    test = pd.read_csv("../../data/test_clean_fill_F10_F1_F4_F20.csv")
    fillNaByMean(train, ['Feature_2'])
    fillNaByMean(test, ['Feature_2'])
    train.to_csv("../../data/train_clean_fill_all.csv")
    test.to_csv("../../data/test_clean_fill_all.csv")
    """
    
    train = pd.read_csv("../../data/train_clean_fill_all.csv")
    test = pd.read_csv("../../data/test_clean_fill_all.csv")
    train_use = train[~ np.isnan(train['Feature_4'])]
    test_use  = test[~ np.isnan(test['Feature_4'])]
    print train_use.shape
    print test_use.shape
    
    features = []
    for i in range(2,121):
        feature = "Ret_"+ str(i)
        features.append(feature)
    for i in range(1,26):
        feature = "Feature_" + str(i)
        features.append(feature)
    features = features + ['Ret_MinusTwo','Ret_MinusOne']
    pieces = [train_use[features],test_use[features]]
    data_combine = pd.concat(pieces)
    print data_combine.shape
    
    seed = random.randrange(1,200)
    train_feature = list(features)
    train_feature.remove('Feature_4')
    train_data = xgb.DMatrix(data_combine[train_feature],label=data_combine['Feature_4'])
    watchlist = [(train_data,'train')]
    model = xgb.train(feature_4_xgb_regression_param, train_data,feature_4_xgb_regression_param['num_round'],watchlist)
    #get predict_data
    train_pred_data = train[np.isnan(train['Feature_4'])][train_feature]
    test_pred_data = test[np.isnan(test['Feature_4'])][train_feature]
    train_pred_data = xgb.DMatrix(train_pred_data)
    test_pred_data = xgb.DMatrix(test_pred_data)
    train_pred = model.predict(train_pred_data)
    test_pred = model.predict(test_pred_data)
    train['Feature_4'][np.isnan(train['Feature_4'])] = train_pred
    test['Feature_4'][np.isnan(test['Feature_4'])] = test_pred
    print train.shape
    print test.shape
    train.to_csv("../../data/train_clean_fill_all_1.csv")
    test.to_csv("../../data/test_clean_fill_all_1.csv")
    
    
    
    
    
    
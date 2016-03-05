'''
Created on 2015/11/10

@author: FZY
'''
import pandas as pd
from WeightPrediction import TunningParamter
from WeightPrediction import Lasso_param,xgb_regression_param_by_tree,xgb_regression_param
#in this  model,I will validate the suppose that the weight is associate with Ret value

if __name__ == '__main__':
    #I will use two vision data  to validate the Model
    #one has the Ret_2 to Ret_180 ,another just have some features
    features_v1 = []
    features_v2 =[]
    train = pd.read_csv("../../data/train_weight.csv")
    allFeatures = train.columns
    #feature of Ret_1 to Ret_180
    RetFeatures= []
    for i in range(1,1):
        feature = "Feature_" + str(i)
        RetFeatures.append(feature)
    features_v1 = list(allFeatures)
    features_v2 = list(set(allFeatures)-set(RetFeatures))
    
    #now we use LR model to fit the data ,of course I will use CV 
    #1.first we use lasso,ridge regression to predict the value
    TunningParamter(xgb_regression_param_by_tree, train, features_v1,['Weight_Intraday'])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
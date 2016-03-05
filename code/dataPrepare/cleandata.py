'''
Created on 2015/11/3

@author: FZY
'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.datasets import dump_svmlight_file
#function for fill unusal missing data
def fillNaByMean(data,features):
    for feature in features :
        mean = np.mean(data[feature])
        data[feature][np.isnan(data[feature])] = mean
        
       
def fillNaByMedian(data,features):
    for feature in features:
        median = np.median(data[feature])
        data[feature][np.isnan(data[feature])] = median
        

def fillNaByost(data,features):
    for feature in features:
        most = np.argmax(np.bincount(np.array(data[feature])))
        data[feature][np.isnan(data[feature])] = most
    
if __name__ == '__main__':
    #read data 
    #some features has value of nan
    #we need to  fill the na value 
    #Feature_1 ,Featur_10,Feature_20  Categorical class
    #so we try to use some method to class our feature 
    train = pd.read_csv("../../data/train.csv")
    test = pd.read_csv("../../data/test.csv")
    train_cols = list(train.columns)
    test_cols = list(test.columns)
    #F1,F2,F4,F10,F20 we need to sperate from cols
    nullFeatures = ['Feature_1','Feature_2','Feature_4','Feature_10','Feature_20']
    
    #now we need to fill the usual feature na value with median vaule
    #for the discrete feature , we need to use mean to copy with the missing data
    
    #category features  <1> 5  9 <10> 8 13 16 <20>
    categoryFeatures = ['Feature_5','Feature_9','Feature_8','Feature_13','Feature_16']
    fillNaByMedian(train, categoryFeatures)
    fillNaByMedian(test, categoryFeatures)
    #discrete feature 
    train_discrete_features = set(train_cols)-set(nullFeatures)- set(categoryFeatures)
    test_discrete_features = set(test_cols)-set(nullFeatures)- set(categoryFeatures)
    
    fillNaByMean(train, list(train_discrete_features))
    fillNaByMean(test, list(test_discrete_features))
    train.to_csv("../../data/train_clean_s1.csv")
    test.to_csv("../../data/test_clean_s1.csv")
    #next we need to deal with unusual features
    #for discrete features,we will use regression to predict the vlaue 
    #and if prediction is most correct ,we will use it,if not ,we will use the median/mean value 
    #to fill it
    
    #now we need to get range of Ret_1 ~ Ret_180
    
    
    
    
    
    
    
    
    
    
    
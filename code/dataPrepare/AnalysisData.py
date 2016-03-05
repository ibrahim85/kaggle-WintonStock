'''
Created on 2015/11/3

@author: FZY
'''
import pandas as pd
import numpy as np
from MissingValueByPrediction import TunningParamter,Ridge_param,Lasso_param,skl_lr_param
from MissingValueByPrediction import xgb_regression_param,xgb_tree_param,xgb_regression_param_by_tree
from sklearn.datasets import dump_svmlight_file
def getNullMessage(data,name):
    rows = data.shape[0]
    messages = ""
    message = ""
    rateMessages = ""
    for col in data.columns:
        rate = data[np.isnan(data[col])].shape[0]/float(rows)
        message = str(col) +":" +str(rate)
        if rate > 0.1 :
            rateMessages = rateMessages + ":" +message + "\n"
        messages = messages + message + '\n'
    f = open("../../data/analysis/%s.null.analysi.txt"%(name),"wb")
    f.write(messages)
    f.write("----------------------speical message--------------------\n")
    f.write(rateMessages)
    f.close()

#plot the picture of missing data
def plotDataScatter(data,features,name):  
    for feature in features :
        data = data[ ~ np.isnan(data[feature])]
        p = data.plot(kind = 'scatter',x='Id',y=feature).get_figure()
        p.savefig("../../data/img/%s.%s.png"%(name,feature))        
        
def dumpDatatoSVMFormat(data,X_feature,Y_feature,filename):
    dump_svmlight_file(data[X_feature], data[Y_feature], "../../data/analysis/svm/%s"%(filename))
          
#plot the picture of category data
if __name__ == '__main__':
    #train = pd.read_csv("../../data/train_clean_fill_F10.csv")
    #test = pd.read_csv("../../data/test_clean_fill_F10.csv")
    #analysis the data 
    #get the columns for analysis
    #np.savetxt("../../data/analysis/train_columns.name.txt",train.columns, fmt="%s")
    #np.savetxt("../../data/analysis/test_columns.name.txt",test.columns,fmt="%s")
    
    #1.now we need to count the null
    #getNullMessage(train, "train")
    #getNullMessage(test, "test")
    
    #2.now we need to analysis the scatter of data
    #features = ['Feature_1','Feature_2','Feature_4','Feature_10','Feature_20']
    """
    features = []
    for i in range(1,26):
        feature = "Feature_" + str(i)
        features.append(feature)
    plotDataScatter(train, features, "train")
    """
    #plotDataScatter(train, features, "train")
    #plotDataScatter(test, features,"test")
    
    #3.the feature1 and feature10 feat20 is category feature
    #train[['Feature_1','Feature_10','Feature_20']].hist()
    #test[['Feature_1','Feature_10','Feature_20']].hist()
    
    #4.now we need to analysis
    #1.for the Feature_2 there are 22.8% have na value ,so we use regreesion to predict 
    #the vlaue 
    """
    train_use = train[~ np.isnan(train['Feature_2'])]
    test_use  = test[~ np.isnan(test['Feature_2'])]
    features = []
    for i in range(1,26):
        feature = "Feature_" + str(i)
        features.append(feature)
    #drop the feature 
    nullFeatures = ['Feature_1','Feature_4','Feature_10','Feature_20']
    features = list(set(features) -set(nullFeatures))
    #use Ridge to predict the bestValue
    #we need to combine the train and test validate our model
    pieces = [train_use[features],test_use[features]]
    data_combine = pd.concat(pieces)
    data_combine.to_csv("../../data/data_combine.csv")
    """
    #Feature_2 has 22.8% na values 
    #data_combine = pd.read_csv("../../data/data_combine.csv")
    #TunningParamter(Ridge_param, data_combine , data_combine.columns,'Feature_2')
    #TunningParamter(Lasso_param, data_combine, data_combine.columns, 'Feature_2')
    #we use the regression to predict the model ,but the result do not have good result
    #I will add  the reture value feature to regression
    
    """
    train_use = train[~ np.isnan(train['Feature_2'])]
    test_use  = test[~ np.isnan(test['Feature_2'])]
    features = []
    for i in range(1,26):
        feature = "Feature_" + str(i)
        features.append(feature)
        
    # add reture value
    #drop the feature 
    nullFeatures = ['Feature_1','Feature_4','Feature_10','Feature_20']
    features = list(set(features) -set(nullFeatures))
    features = features + ['Ret_MinusTwo','Ret_MinusOne']
    for i in range(2,121):
        feature= "Ret_" + str(i)
        features.append(feature)
    
    print features
    #use Ridge to predict the bestValue
    #we need to combine the train and test validate our model
    pieces = [train_use[features],test_use[features]]
    data_combine = pd.concat(pieces)
    data_combine.to_csv("../../data/data_combine_1.csv")
    
    data_combine = pd.read_csv("../../data/data_combine_1.csv")              
    TunningParamter(Ridge_param, data_combine , data_combine.columns,'Feature_2','combine_1')
    TunningParamter(Lasso_param, data_combine, data_combine.columns, 'Feature_2','combine_1')
    """
    #now we need to deal with Feature_4
    """
    train_use = train[~ np.isnan(train['Feature_4'])]
    test_use  = test[~ np.isnan(test['Feature_4'])]
    features = []
    for i in range(1,26):
        feature = "Feature_" + str(i)
        features.append(feature)
    
    nullFeatures = ['Feature_1','Feature_2','Feature_10','Feature_20']
    features = list(set(features) -set(nullFeatures))
    pieces = [train_use[features],test_use[features]]
    data_combine = pd.concat(pieces)
    data_combine.to_csv("../../data/data_combine_2.csv")
    """
    #data_combine = pd.read_csv("../../data/data_combine_2.csv") 
    #dumpDatatoSVMFormat(data_combine, X_feature, Y_feature, 'combine_2')             
    #TunningParamter(Ridge_param, data_combine , data_combine.columns,'Feature_4','combine_2')
    #TunningParamter(Lasso_param, data_combine, data_combine.columns, 'Feature_4','combine_2')
   
   
    #for the feature_20 ,50% na 
    """
    train_use = train[~ np.isnan(train['Feature_1'])]
    test_use  = test[~ np.isnan(test['Feature_1'])]
    features = []
    for i in range(1,26):
        feature = "Feature_" + str(i)
        features.append(feature)
    
    nullFeatures = ['Feature_10','Feature_2','Feature_20','Feature_4']
    features = list(set(features) -set(nullFeatures))
    features = features + ['Ret_MinusTwo','Ret_MinusOne']
    pieces = [train_use[features],test_use[features]]
    data_combine = pd.concat(pieces)
    data_combine.to_csv("../../data/data_combine_5.csv")
    
    data_combine = pd.read_csv("../../data/data_combine_5.csv") 
    TunningParamter(skl_lr_param, data_combine , data_combine.columns,'Feature_1','combine_5')
    """
    """
    train_use = train[~ np.isnan(train['Feature_20'])]
    test_use  = test[~ np.isnan(test['Feature_20'])]
    features = []
    for i in range(1,26):
        feature = "Feature_" + str(i)
        features.append(feature)
    nullFeatures = ['Feature_2','Feature_4','Feature_1']
    features = list(set(features) -set(nullFeatures))
    features = features + ['Ret_MinusTwo','Ret_MinusOne']
    pieces = [train_use[features],test_use[features]]
    data_combine = pd.concat(pieces)
    data_combine['Feature_20'] = data_combine['Feature_20'] - 2 
    
    TunningParamter(xgb_tree_param, data_combine, data_combine.columns,'Feature_20','data_combine_20_feature')
    """
    ##now we need to get range of Ret_1 ~ Ret_180
    """
    train = pd.read_csv("../../data/train_weight.csv")
    features = []
    for i in range(2,181):
        feature = "Ret_"+str(i)
        features.append(feature)
    #plot scatter
    plotDataScatter(train,features,'train')
    """
    train = pd.read_csv("../../data/train_clean_fill_all_1.csv")
    getNullMessage(train, 'train_2')

    
    
       
    
    
    
    

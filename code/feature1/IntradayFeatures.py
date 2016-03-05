'''
Created on 2015/12/2

@author: FZY
'''
#in this model , I will get data by cut last hours to 1 part, 2 parts,3 parts,4 parts , 6 parts , 12 parts
# I think there is a threshold to cut , at first I need to get features for there parts 
import pandas as pd,numpy as np


def cutFeatures(data,num,data_type):
    all_features= []
    #add the feature to predict
    #features = [ "Feature_%d"%(i) for i in range(1,26)]
    #all_features.extend(features)  
    features =["Ret_%d"%(i) for i in range(2,121)]
    all_features.extend(features) 
    features = ["Ret_%d_price"%(i) for i in range(2,121)]
    all_features.extend(features)
    all_features.append('Ret_total_price')
    all_features.append('Ret_mean')
    all_features.append('Ret_mean_price')
    all_features.append('Ret_max_price')
    all_features.append('Ret_max')
    all_features.append('Ret_min_price')
    all_features.append('Ret_min')
    all_features.append('Ret_var')
    all_features.append('Ret_var_price')
    #add pred_value 
    print num
    step = int(60/num)
    for i in range(0,num):
        #count predict value 
        if data_type == 'train':
            start = 121+ i*step
            end = 120+(i+1)*step+1
            count_features = [ "Ret_%d"%(k) for k in range(start,end)]
            all_features.extend(count_features)
            pred_label = "Pred_%d"%(i+1)
            all_features.append(pred_label)
            print data[count_features].shape
            data[pred_label] = np.median(data[count_features],axis=1)
            print "train pred:"
            #print data[pred_label]  
    #add features to train
    for i in range(0,2*num):
        if i == 0:
            start = 2
            end = step + 1
        else:
            start = i*step + 1
            end = (i+1)*step + 1
        count_features = ["Ret_%d"%(k) for k in range(start,end)]
        all_features.extend(count_features)
        data['Ret_all_%d'%(i+1)] = np.median(data[count_features],axis=1)
        print data['Ret_all_%d'%(i+1)]
        data['Ret_all_std_%d'%(i+1)] = np.std(data[count_features],axis=1)
        print 'end:%d'%(i)
    data.to_csv("../../data/%s.intra_%d.csv"%(data_type,num))       
   
if __name__ == '__main__':
    print 'cut feature'
    part_nums = [1,2,3,4,6,12]
    print 'read data'
    train = pd.read_csv("../../data/train_clean_fill_all_2.csv")
    test = pd.read_csv("../../data/test_clean_fill_all_2.csv")
    #read data for csv
    for run in part_nums:
        print 'run:%d'%(run)
        cutFeatures(train, run, 'train')
        cutFeatures(test, run, 'test')        
        
    
    
    
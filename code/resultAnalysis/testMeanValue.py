'''
Created on 2015/12/15

@author: FZY
'''
import pandas as pd 
import numpy as np
def loadCVIndex(path):
    f = open(path,"rb")
    for l in f :
        line = l 
    indexlist = list(line.split(","))
    integerlist = []
    for ind in indexlist:
        integerlist.append(int(ind))
    return integerlist

def calError(x):
    error = 0 
    for i in range(121,181):
        #error = error + x['Weight_Intraday']*(x['Ret_%d'%(i)]-x['Pred_1'])
        error = error +np.abs(x['Ret_%d'%(i)]-x['Pred_1'])
    return error*x['Weight_Intraday']
        
if __name__ == '__main__':
    #read data 
    train = pd.read_csv("../../data/train.intra_1.csv")
    cv = []
    for run in range(0,3):
        print("run:%d"%(run))
        train_index = loadCVIndex("../../data/cv/train.run%d.txt"%(run+1))
        test_index = loadCVIndex("../../data/cv/valid.run%d.txt"%(run+1))
        #use test_index to valid data
        valid_data = train.iloc[test_index]
        valid_data['error']=list(valid_data.apply(lambda x :calError(x),axis=1))
        count_error = np.sum(valid_data['error'])/(28000*62)
        cv.append(count_error)
        print count_error
    print(np.mean(cv))   
   
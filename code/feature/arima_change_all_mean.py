'''
Created on 2015/11/19

@author: FZY
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plotTrend(data):
    features = ["Ret_%d"%(i) for i in range(2,181)]
    list_data = list(data[features])
    plt.plot(range(2,181),list_data)
    plt.savefig("../../data/img/feature/trend_%d.png"%(data['Id']))
    plt.close()
    
    
if __name__ == "__main__":
    train = pd.read_csv("../../data/train_clean_fill_all_1.csv")
    test = pd.read_csv("../../data/test_clean_fill_all_1.csv")
    #this way I need to predict the all message
    result = []
    print 'count of mean of ret value'
    for i in range(2,181):
        mean = np.mean(train['Ret_%d'%(i)])
        result.append(mean)
    print 'count end'
    #save the message
    pd.Series(result).to_csv("../../data/feature/count_mean.csv")
    result = []
    print 'count of mean of ret value'
    for i in range(2,121):
        mean = np.mean(test['Ret_%d'%(i)])
        result.append(mean)
    pd.Series(result).to_csv("../../data/feature/test_count_mean.csv")
   
        
#in this model ,we wiil predict the all change 
#the we use  all - Individual
#so in this  way I will Arima model to predict the final 
import pandas as pd
if __name__ == "__main__":
    train = pd.read_csv("../../data/feature/train_clean_fill_all.csv")
    message = pd.read_csv("../../data/feature/count_mean.csv",names=['id','mean'])
    #we need to use all change to predict final 
    #transform_data
    for i in range(2,121):
        feature = 'Ret_%d'%(i)
        train['mean_%d'%(i)] = train['Ret_%d'%(i)] - message['mean']
    #now we need to dump the message to csv file
    #I will predict the mean of all data
    
    
    
'''
Created on 2015/12/16

@author: FZY
'''
import pandas as pd 
if __name__ == '__main__':
    #analysis the trend of price 
    train = pd.read_csv("../../data/new/train_ts_price.csv")
    for i in range(0,120):
        data = train.iloc[i]
        p = data.plot().get_figure()
        p.savefig("../../data/new/change.%d.png"%(i)) 
    
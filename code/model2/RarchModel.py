'''
Created on 2015/12/16

@author: FZY
'''
import pandas as pd
import pandas.rpy.common as rpy
from rpy2.robjects.packages import importr
def pandas_to_r(df,name):
    from rpy2.robjects import r,globalenv
    r_df = rpy.convert_to_r_dataframe(df)
    globalenv[name] = r_df

#

if __name__ == '__main__':
    #read ret price
    print 
    train = pd.read_csv("../../data/train_clean_fill_all_1.csv")
    pandas_to_r(train, "train")
    importr("rugarch")
    
    
    
    
    
    
    
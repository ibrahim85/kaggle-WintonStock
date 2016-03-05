#in this model , I will get some idea to predict the Intraday and 
#combine  the  Daily data ,and submit it 
import pandas as pd 
import numpy as np
from sklearn.linear_model import Ridge,Lasso
from sklearn.cross_validation import train_test_split

Lasso_param = {
    'task':'skl_lasso',
    'alpha':0.24460944908709087,
    'random_state':2015,   
}

Ridge_param= {
    'task':'skl_ridge',
    'alpha':20,  
    'random_state':2015           
}
def getRet(data,list_values,step):
    num = 60/step
    for s in range(1,num+1):
        tmp = []
        tmp.append(data[str(s)])
        tmp = tmp * step
        list_values.extend(tmp)
    list_values.append(0)
    list_values.append(0)   
def listRet(data,list_values,step):
    data.apply(lambda x : getRet(x, list_values,step),axis=1)
    return list_values


def predictValueByLasso(param,num,rate):
    train = pd.read_csv("../../data/train.intra_%d.csv"%(num))
    test = pd.read_csv("../../data/test.intra_%d.csv"%(num))
    feature = ['Ret_120']
    predict_lable = []
    for i in range(1,num+1):
        predict_lable.append("Pred_%d"%(i))
    k=20
    print 'train model'
    for i in range(0,k):
        print i 
        X_train,X_test,Y_train,Y_test = train_test_split(train[feature],train[predict_lable],test_size=0.3,random_state=i)
        lasso = Lasso(alpha=param['alpha'],normalize=True)
        lasso.fit(X_train,Y_train)
        pred_value = lasso.predict(test[feature])
        col = [str(t) for t in range(1,num+1)]
        if i == 0:
            df = pd.DataFrame(pred_value,columns=col)
        else:
            tmp = pd.DataFrame(pred_value,columns=col)
            for t in col:
                df[t] = df[t] + tmp[t]  
    print 'train end'
    print 'save data'    
    for i in rate:
        tp = df.copy()
        list_values = []
        for t in col:
            tp[t] = tp[t]/(k*i)
        tp.to_csv("../../data/result/RetPlus/lasso.intra_pred_%f.csv"%(i),columns=col)
        step = 60/num
        list_v = listRet(tp,list_values,step)
        pd.Series(list_v).to_csv("../../data/result/regression_lasso_intra_%f.csv"%(i))

def predictValueByRidge(param,num,rate):
    train = pd.read_csv("../../data/train.intra_%d.csv"%(num))
    test = pd.read_csv("../../data/test.intra_%d.csv"%(num))
    feature = ['Ret_120']
    #feature.append('Ret+')
    predict_lable = []
    for i in range(1,num+1):
        predict_lable.append("Pred_%d"%(i))
    k=500
    print 'train model'
    for i in range(0,k):
        print i 
        X_train,X_test,Y_train,Y_test = train_test_split(train[feature],train[predict_lable],test_size=0.3,random_state=i)
        lasso = Ridge(alpha=param['alpha'],normalize=True)
        lasso.fit(X_train,Y_train)
        pred_value = lasso.predict(test[feature])
        col = [str(t) for t in range(1,num+1)]
        if i == 0:
            df = pd.DataFrame(pred_value,columns=col)
        else:
            tmp = pd.DataFrame(pred_value,columns=col)
            for t in col:
                df[t] = df[t] + tmp[t]  
    print 'train end'
    print 'save data'    
    for i in rate:
        tp = df.copy()
        list_values = []
        for t in col:
            tp[t] = tp[t]/(k*i)
        tp.to_csv("../../data/result/RetPlus/ridge.intra_pred_%f.csv"%(i),columns=col)
        step = 60/num
        list_v = listRet(tp,list_values,step)
        pd.Series(list_v).to_csv("../../data/result/regression_ridge_intra_%f.csv"%(i))
if __name__ == "__main__":
    rate = [1.0,2.0,3.0,4.0,5.0,6.0]
    print 'predict_value'
    predictValueByRidge(Ridge_param, 1, rate)

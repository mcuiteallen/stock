import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.models import load_model


foxconndf= pd.read_csv('data/STOCK_EMOME.csv', index_col=0 )

#print(foxconndf)

foxconndf.dropna(how='any',inplace=True)
#print(foxconndf)

cycleTimes = 20;


def normalize(df):
    newdf= df.copy()
    min_max_scaler = preprocessing.MinMaxScaler()    
    newdf['open'] = min_max_scaler.fit_transform(df.open.values.reshape(-1,1))
    newdf['low'] = min_max_scaler.fit_transform(df.low.values.reshape(-1,1))
    newdf['high'] = min_max_scaler.fit_transform(df.high.values.reshape(-1,1))
    newdf['volume'] = min_max_scaler.fit_transform(df.volume.values.reshape(-1,1))
    newdf['close'] = min_max_scaler.fit_transform(df.close.values.reshape(-1,1))
    
    return newdf
foxconndf_norm = normalize(foxconndf)
print(foxconndf_norm)

def denormalize(df, norm_value, labelName):
    original_value = df[labelName].values.reshape(-1,1)
    norm_value = norm_value.reshape(-1,1)
    
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit_transform(original_value)
    denorm_value = min_max_scaler.inverse_transform(norm_value)
    
    return denorm_value


def data_helper_get_last(df, time_frame):
    
    # 資料維度: 量, 開、 高、低、收, 5維
    number_features = len(df.columns)
    datavalue = df.as_matrix()
    result = []
    # 這邊只塞最後一筆
    #for index in range( len(datavalue) - (time_frame - 1) ):
    #result.append(datavalue[(len(datavalue) - (time_frame - 1)): ( len(datavalue) + 1) ])
    result.append(datavalue[(len(datavalue) - (time_frame)): ])
    result = np.array(result)
    #number_train = round(0.9 * result.shape[0]) # 取 result 的前90% instance做為訓練資料
    #print("res = ")
    #print(result)
    #print("======")
    #x_train = result[:, :-1] # 訓練資料中, 只取每一個 time_frame 中除了最後一筆的所有資料做為feature
    #y_train = result[:, -1][:,-1] # 訓練資料中, 取每一個 time_frame 中最後一筆資料的最後一個數值(收盤價)做為答案
    

    #x_test = result[int(number_train):, :-1]
    #y_test = result[int(number_train):, -1][:,-1]
    x_test = result[:,:]
    #y_test = result[:,-1][:,-1]
    y_test = result[:,-1]
	

    #x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], number_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], number_features))  
    return [x_test, y_test]
# 以N天為一區間進行股價預測
x_test, y_test = data_helper_get_last(foxconndf_norm, 15)
#print(X_train);
#print(x_test);

#model = Sequential()



#load model
model = load_model('data/lstm_model.h5')
i = 0
while i < cycleTimes:
    pred = model.predict(x_test)
    x_next = x_test[:,1:,:]
    x_next = np.append(x_next, pred)
    x_next = np.reshape(x_next, (x_test.shape[0], x_test.shape[1], -1))
#    print("time " + str(i) + ":")
#    print("x_test=========")
#    print(x_test)
#    print("pred=========")
#    print(pred)
#    print("x_next=========")
#    print(x_next)
    x_test = x_next
    if(i == 0):
        predAll = pred
    else:
        predAll = np.concatenate((predAll, pred), 0)
#    print("predAll=========")
#    print(predAll)
    i+=1



#print("target")
#print(denorm_xtest)

#print("P_targetY:")
#print(y_test)
#print("P_ans:")
#print(pred)

y_test0 = y_test[:, 0]
y_test1 = y_test[:, 1]
y_test2 = y_test[:, 2]
y_test3 = y_test[:, 3]
y_test4 = y_test[:, 4]



#denorm_ytest0 = denormalize(foxconndf, y_test0, "volume")
#denorm_ytest1 = denormalize(foxconndf, y_test1, "open")
#denorm_ytest2 = denormalize(foxconndf, y_test2, "high")
#denorm_ytest3 = denormalize(foxconndf, y_test3, "low")
#denorm_ytest4 = denormalize(foxconndf, y_test4, "close")

#print("\n\n======")
#print("the last day close price: " + str(denorm_ytest4[len(denorm_ytest4) - 1, 0]))
#print("prediction next day close price: " + str(denorm_pred4[len(denorm_pred4) - 1, 0]))
#print("======\n\n")
#print("end")
#如果總資料筆數大於 N 筆 就把 map 畫出來

#原始資料
oldData = foxconndf_norm.as_matrix()

old0 = oldData[:, 0]
old1 = oldData[:, 1]
old2 = oldData[:, 2]
old3 = oldData[:, 3]
old4 = oldData[:, 4]

denorm_old0 = denormalize(foxconndf, old0, "volume")
denorm_old1 = denormalize(foxconndf, old1, "open")
denorm_old2 = denormalize(foxconndf, old2, "high")
denorm_old3 = denormalize(foxconndf, old3, "low")
denorm_old4 = denormalize(foxconndf, old4, "close")


print("\n\n======inputData====")
print(oldData)
print("======\n\n")
print("\n\n======predData====")
print(predAll)
print("======\n\n")


#pred0 = predAll[:, 0]
#pred1 = predAll[:, 1]
#pred2 = predAll[:, 2]
#pred3 = predAll[:, 3]
#pred4 = predAll[:, 4]

#串起來顯示，讓它時間軸分開
allData = np.concatenate((oldData, predAll), 0)
pred0 = allData[:, 0]
pred1 = allData[:, 1]
pred2 = allData[:, 2]
pred3 = allData[:, 3]
pred4 = allData[:, 4]

denorm_pred0 = denormalize(foxconndf, pred0, "volume")
denorm_pred1 = denormalize(foxconndf, pred1, "open")
denorm_pred2 = denormalize(foxconndf, pred2, "high")
denorm_pred3 = denormalize(foxconndf, pred3, "low")
denorm_pred4 = denormalize(foxconndf, pred4, "close")





if(len(denorm_pred4) > 0):
#交易量的連續預測，形狀很醜，關了(隔3天以上，值就幾乎不動了)
#    f1 = plt.figure(1)
#    plt.plot(denorm_pred0,color='red', label='Volume_Prediction')
#    plt.plot(denorm_old0,color='blue', label='Volume_Answer')
#    plt.legend(loc='best')
#    f1.show()
	
    f2 = plt.figure(2)	
    plt.plot(denorm_pred4,color='red', label='Close_Prediction')
    #plt.plot(denorm_Else4,color='green', label='ELSE')
    plt.plot(denorm_old4,color='blue', label='Close_Answer')
    plt.legend(loc='best')
    f2.show()
    plt.show()

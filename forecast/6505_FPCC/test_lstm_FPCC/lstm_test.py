import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.models import load_model


foxconndf= pd.read_csv('data/STOCK_FPCC.csv', index_col=0 )

#print(foxconndf)

foxconndf.dropna(how='any',inplace=True)
#print(foxconndf)

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
#print(foxconndf_norm)

def denormalize(df, norm_value, labelName):
    original_value = df[labelName].values.reshape(-1,1)
    norm_value = norm_value.reshape(-1,1)
    
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit_transform(original_value)
    denorm_value = min_max_scaler.inverse_transform(norm_value)
    
    return denorm_value


def data_helper(df, time_frame):
    
    # 資料維度: 量, 開、 高、低、收, 5維
    number_features = len(df.columns)
    datavalue = df.as_matrix()
    result = []
    # 改成跑到最後一筆
    for index in range( len(datavalue) - (time_frame - 1) ):
        result.append(datavalue[index: index + (time_frame + 0) ])
    
    result = np.array(result)
    #number_train = round(0.9 * result.shape[0]) # 取 result 的前90% instance做為訓練資料
    #print("res = ")
    #print(result)
    #print("======")
    #x_train = result[:, :-1] # 訓練資料中, 只取每一個 time_frame 中除了最後一筆的所有資料做為feature
    #y_train = result[:, -1][:,-1] # 訓練資料中, 取每一個 time_frame 中最後一筆資料的最後一個數值(收盤價)做為答案
    
    # 測試資料
    #x_test = result[int(number_train):, :-1]
    #y_test = result[int(number_train):, -1][:,-1]
    x_test = result[:,:]
    #y_test = result[:,-1][:,-1]
    y_test = result[:,-1]
	
    # 將資料組成變好看一點
    #x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], number_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], number_features))  
    return [x_test, y_test]
# 以N天為一區間進行股價預測
x_test, y_test = data_helper(foxconndf_norm, 15)
#print(X_train);
#print(x_test);

#model = Sequential()



#load model
model = load_model('data/lstm_model.h5')

pred = model.predict(x_test)


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

pred0 = pred[:, 0]
pred1 = pred[:, 1]
pred2 = pred[:, 2]
pred3 = pred[:, 3]
pred4 = pred[:, 4]

#print("targetY:")
#print(denorm_ytest)
#print("ans:")
#print(denorm_pred)

# volume,open,high,low,close
denorm_pred0 = denormalize(foxconndf, pred0, "volume")
denorm_ytest0 = denormalize(foxconndf, y_test0, "volume")
denorm_pred1 = denormalize(foxconndf, pred1, "open")
denorm_ytest1 = denormalize(foxconndf, y_test1, "open")
denorm_pred2 = denormalize(foxconndf, pred2, "high")
denorm_ytest2 = denormalize(foxconndf, y_test2, "high")
denorm_pred3 = denormalize(foxconndf, pred3, "low")
denorm_ytest3 = denormalize(foxconndf, y_test3, "low")
denorm_pred4 = denormalize(foxconndf, pred4, "close")
denorm_ytest4 = denormalize(foxconndf, y_test4, "close")

print("\n\n======")

print("the last day volume: " + str(denorm_ytest0[len(denorm_ytest0) - 1, 0]))
print("prediction next day volume: " + str(denorm_pred0[len(denorm_pred0) - 1, 0]))

print("the last day close price: " + str(denorm_ytest4[len(denorm_ytest4) - 1, 0]))
print("prediction next day close price: " + str(denorm_pred4[len(denorm_pred4) - 1, 0]))
print("======\n\n")
#print("end")
#如果總資料筆數大於 N 筆 就把 map 畫出來
if(len(denorm_pred4) > 50):
    f1 = plt.figure(1)
    plt.plot(denorm_pred0,color='red', label='Volume_Prediction')
    plt.plot(denorm_ytest0,color='blue', label='Volume_Answer')
    plt.legend(loc='best')
    f1.show()
	
    f2 = plt.figure(2)	
    plt.plot(denorm_pred4,color='red', label='Close_Prediction')
    plt.plot(denorm_ytest4,color='blue', label='Close_Answer')
    plt.legend(loc='best')
    f2.show()
    plt.show()

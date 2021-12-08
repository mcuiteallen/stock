import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM

#接到指令，要改成 input 5 output 5 然後 循環預測

foxconndf= pd.read_csv('data/STOCK_8150_105_107_FTG.csv', index_col=0 )
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
    
    # 資料維度: 開盤價、收盤價、最高價、最低價、成交量, 5維
    number_features = len(df.columns)
    # 將dataframe 轉成 numpy array
    datavalue = df.as_matrix()
    result = []
    # 若想要觀察的 time_frame 為20天, 需要多加一天做為驗證答案
    for index in range( len(datavalue) - (time_frame+1) ): # 從 datavalue 的第0個跑到倒數第 time_frame+1 個
        result.append(datavalue[index: index + (time_frame+1) ]) # 逐筆取出 time_frame+1 個K棒數值做為一筆 instance
    
    result = np.array(result)
    number_train = round(0.9 * result.shape[0]) # 取 result 的前90% instance做為訓練資料
    
    x_train = result[:int(number_train), :-1] # 訓練資料中, 只取每一個 time_frame 中除了最後一筆的所有資料做為feature
    y_train = result[:int(number_train), -1]
    
    # 測試資料
    #x_test = result[int(number_train):, :-1]
    #y_test = result[int(number_train):, -1][:,-1]
    #x_test = result[:int(number_train), :-1]
    #y_test = result[:int(number_train), -1][:,-1]
    x_test = result[:, :-1]
    y_test = result[:, -1]
    
    # 將資料組成變好看一點
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], number_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], number_features))  
    return [x_train, y_train, x_test, y_test]
# 以n天為一區間進行股價預測
X_train, y_train, X_test, y_test = data_helper(foxconndf_norm, 15)
#print(X_train);

"""
def build_model(input_length, input_dim):
    d = 0.3
    model = Sequential()
    model.add(LSTM(256, input_shape=(input_length, input_dim), return_sequences=True))
    model.add(Dropout(d))
    model.add(LSTM(256, input_shape=(input_length, input_dim), return_sequences=False))
    model.add(Dropout(d))
    model.add(Dense(16,kernel_initializer="uniform",activation='relu'))
    model.add(Dense(1,kernel_initializer="uniform",activation='linear'))
    model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])
    return model
"""

def build_model(input_length, input_dim):
    d = 0.25
    model = Sequential()
    model.add(LSTM(225, input_shape=(input_length, input_dim), return_sequences=True))
    model.add(Dropout(d))
    model.add(LSTM(225, input_shape=(input_length, input_dim), return_sequences=False))
    model.add(Dropout(d))
    model.add(Dense(15,kernel_initializer="uniform",activation='relu'))
    model.add(Dense(5,kernel_initializer="uniform",activation='linear'))
    model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])
    return model
	
# n天、5維
model = build_model( 15, 5 )

# 一個batch有128個instance，總共跑50個迭代
model.fit( X_train, y_train, batch_size=128, epochs=500, validation_split=0.1, verbose=1)

#save model
model.save('data/lstm_model.h5')

# 用訓練好的 LSTM 模型對測試資料集進行預測
pred = model.predict(X_test)
# 將預測值與正確答案還原回原來的區間值
#denorm_pred = denormalize(foxconndf, pred)
#denorm_ytest = denormalize(foxconndf, y_test)


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
print("the last day close price: " + str(denorm_ytest4[len(denorm_ytest4) - 1, 0]))
print("prediction next day close price: " + str(denorm_pred4[len(denorm_pred4) - 1, 0]))
print("======\n\n")
#print("end")
#如果總資料筆數大於 N 筆 就把 map 畫出來
if(len(denorm_pred4) > 15):
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

from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model 
from sklearn.metrics import mean_squared_error
from keras import backend
import pandas as pd
 
# 將序列轉換為監督式方法
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all togethe
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
 
# 載入數據
dataset = read_csv('weatherday二崙鄉產銷履歷.csv', header=0, index_col=0)
values = dataset.values
# 整數編碼
encoder = LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4])
# 確保數據為浮點數
values = values.astype('float32')
# 資料正規劃
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# 指定滯後時間大小
n_hours = 4
n_features = 16
# 建構為監督式學習
reframed = series_to_supervised(scaled, n_hours, 1)
reframed.to_csv('weatherday二崙鄉產銷履歷Time.csv')
print(reframed.shape)
 
# 分訓練與測試
values = reframed.values
n_train_hours = 400
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# 分輸入及輸出
n_obs = n_hours * n_features
train_X, train_y = train[:, :n_obs], train[:, -n_features]
test_X, test_y = test[:, :n_obs], test[:, -n_features]
print(train_X.shape, len(train_X), train_y.shape)
# 重塑為3D模型 [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


#計算RMSE
def rmse(y_true, y_pred):
    	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))
 
# 設計網路
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam', metrics=[rmse])
# 擬合模型
history = model.fit(train_X, train_y, epochs=50, batch_size=5, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# 作出預測
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))
# 反轉成實際預測值
inv_yhat = concatenate((yhat, test_X[:, -14:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# 反轉成實際值大小
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, -14:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# 計算RMSE大小
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
#印出數據
#model.save('air.h5')  
print('Test RMSE: %.3f' % rmse)
print(inv_yhat[-10:])

#輸出預測結果
df_list = []
df_list.append(pd.DataFrame(inv_yhat))
df = pd.concat(df_list)
df.to_csv('pred.csv', index=None, header=0)
# 繪圖
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
#pyplot.plot(history.history['rmse'], label='RMSE')
pyplot.legend()
pyplot.show()
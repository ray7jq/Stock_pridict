#
# import csv
#
# with open("stock.csv", "r", encoding = "utf-8") as f:
#     reader = csv.reader(f)
#     i = 0
#     for row in reader:
#         i =i+1
#         if i >3255:
#             break
#         out = open("sh600485.csv", "a", newline="")
#         csv_writer = csv.writer(out, dialect="excel")
#         csv_writer.writerow(row)

from pandas import read_csv
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = read_csv('sh600485.csv',usecols = [3,5,6,7,8,9])
data.columns = ["deal_date","popen","phigh","plow","pclose","vol"]
data.set_index(["deal_date"], inplace=True)
def Stock_Price_LSTM_Data_Processing(data,memory_day = 5,preDay = 10):
    data["label"] = data["pclose"].shift(-preDay)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    sca_X = scaler.fit_transform(data.iloc[:,:-1])

    from collections import deque

    deq = deque(maxlen=memory_day)

    X = []
    for i in sca_X:
        deq.append(list(i))
        if len(deq) == memory_day:
            X.append(list(deq))

    x_late = X[-preDay:]
    X = X[:-preDay]
    y = data["pclose"].values[memory_day-1:-preDay]
    import numpy as np
    X = np.array(X)
    y = np.array(y)
    return X,y,x_late

X,y,x_late = Stock_Price_LSTM_Data_Processing(data,5,10)

print(len(X))
print(len(y))
print(len(x_late))

pre_day = 10
# memory_days = [5, 10, 15]
# lstm_layers = [1, 2, 3]
# dense_layers = [1, 2, 3]
# units = [16, 32]
memory_days = [5]
lstm_layers = [1]
dense_layers = [1]
units = [32]
from tensorflow.keras.callbacks import ModelCheckpoint

for the_memory_days in memory_days:
    for the_lstm_layers in lstm_layers:
        for the_dense_layers in dense_layers:
            for the_units in units:
                filepath = './models/{val_mape:.2f}_{epoch:02d}_' + f'men_{the_memory_days}_lstm_{the_lstm_layers}_dense_{the_dense_layers}_units_{the_units}'

                checkpoint = ModelCheckpoint(
                    filepath=filepath,
                    save_weights_only=False,
                    monitor='val_mape',
                    mode='min',
                    save_best_only=True)

                X, y, x_late = Stock_Price_LSTM_Data_Processing(data, the_memory_days, pre_day)
                from sklearn.model_selection import train_test_split

                X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.1)

                import tensorflow as tf
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import LSTM, Dense, Dropout

                model = Sequential()
                model.add(LSTM(the_units, input_shape=X.shape[1:], activation='relu', return_sequences=True))
                model.add(Dropout(0.1))
                for i in range(the_lstm_layers):
                    model.add(LSTM(the_units, activation='relu', return_sequences=True))
                    model.add(Dropout(0.1))

                model.add(LSTM(the_units, activation='relu'))
                model.add(Dropout(0.1))

                for i in range(the_dense_layers):
                    model.add(Dense(the_units, activation='relu'))
                    model.add(Dropout(0.1))

                model.add(Dense(1))

                model.compile(optimizer='adam',
                              loss='mse',
                              metrics=['mape'])

                model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test),
                          callbacks=[checkpoint])

best_model = load_model('./models/6.33_15_men_5_lstm_1_dense_1_units_32')
best_model.summary()

X_train, X_test, y_train, y_test = train_test_split(X,y,shuffle=False,test_size=0.1)
best_model.evaluate(X_test,y_test)

pred  = best_model.predict(X_test)
print(len(pred))

data_time = data.index[-len(y_test):]
plt.plot(data_time,y_test,color='red',label='price')
plt.plot(data_time,pred,color='blue',label='pred_price')
plt.legend()
plt.show()
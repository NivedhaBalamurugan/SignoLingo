import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from preprocess import *



actions_dig = np.array(['1','2','3','4','5','6','7','8','9','0'])
actions_alpha = np.array(['A','B','C','D','E','F','G','H','I','K','L','M','O','P','R','S','T','U','V','W','X','Y'])


log_dir_dig = os.path.join('Logs_dig')
tb_callback_dig = TensorBoard(log_dir=log_dir_dig)
model_dig = Sequential()

model_dig.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,63)))
model_dig.add(LSTM(128, return_sequences=True, activation='relu'))
model_dig.add(LSTM(64, return_sequences=False, activation='relu'))
model_dig.add(Dense(64, activation='relu'))
model_dig.add(Dense(32, activation='relu'))
model_dig.add(Dense(actions_dig.shape[0], activation='softmax'))


log_dir_alpha = os.path.join('Logs_alpha')
tb_callback_alpha = TensorBoard(log_dir=log_dir_alpha)
model_alpha = Sequential()

model_alpha.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,63)))
model_alpha.add(LSTM(128, return_sequences=True, activation='relu'))
model_alpha.add(LSTM(64, return_sequences=False, activation='relu'))
model_alpha.add(Dense(64, activation='relu'))
model_alpha.add(Dense(32, activation='relu'))
model_alpha.add(Dense(actions_alpha.shape[0], activation='softmax'))


model_dig.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model_dig.fit(X_dig_train, y_dig_train, epochs=200, callbacks=[tb_callback_dig])
model_dig.summary()


model_alpha.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model_alpha.fit(X_alpha_train, y_alpha_train, epochs=500, callbacks=[tb_callback_alpha])
model_alpha.summary()



from sklearn.metrics import accuracy_score

yhat_alpha = model_alpha.predict(X_alpha_test)
ytrue_alpha = np.argmax(y_alpha_test, axis=1).tolist()
yhat_alpha = np.argmax(yhat_alpha, axis=1).tolist()
accuracy_score(ytrue_alpha, yhat_alpha)

yhat_dig = model_dig.predict(X_dig_test)
ytrue_dig = np.argmax(y_dig_test, axis=1).tolist()
yhat_dig = np.argmax(yhat_dig, axis=1).tolist()
accuracy_score(ytrue_dig, yhat_dig)
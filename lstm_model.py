import os
import numpy as np
from preprocess_lnd import *


from keras.models import Sequential # type: ignore
from keras.layers import LSTM, Dense, Dropout # type: ignore
from keras.callbacks import TensorBoard # type: ignore


actions_dig = np.array(['1','2','3','4','5','6','7','8','9','0'])
actions_alpha = np.array(['A','B','C','D','E','F','G','H','I','K','L','M','O','P','R','S','T','U','V','W','X','Y'])


log_dir_dig = os.path.join('Logs_dig')
tb_callback_dig = TensorBoard(log_dir=log_dir_dig)
model_dig = Sequential()

model_dig.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,63)))
model_dig.add(Dropout(0.2))
model_dig.add(LSTM(128, return_sequences=True, activation='relu'))
model_dig.add(Dropout(0.2))
model_dig.add(LSTM(64, return_sequences=False, activation='relu'))
model_dig.add(Dropout(0.2))
model_dig.add(Dense(64, activation='relu'))
model_dig.add(Dropout(0.2))
model_dig.add(Dense(32, activation='relu'))
model_dig.add(Dropout(0.2))
model_dig.add(Dense(actions_dig.shape[0], activation='softmax'))





model_dig.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model_dig.fit(X_dig_train, y_dig_train, epochs=200, callbacks=[tb_callback_dig])
model_dig.summary()

model_dig.save('model_dig.h5')


from sklearn.metrics import accuracy_score



yhat_dig = model_dig.predict(X_dig_test)
ytrue_dig = np.argmax(y_dig_test, axis=1).tolist()
yhat_dig = np.argmax(yhat_dig, axis=1).tolist()
accuracy_score(ytrue_dig, yhat_dig)


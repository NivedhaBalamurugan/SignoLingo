import os
import numpy as np
from preprocess_dig import *

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout 
from keras.callbacks import TensorBoard , EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from keras.models import load_model
import datetime

actions_dig = np.array(['1','2','3','4','5','6','7','8','9','0','10'])


tb_callback_dig = TensorBoard(log_dir="log_dig/fit/"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=1)

model_checkpoint = ModelCheckpoint('best_model_dig_with10.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)

model_dig = Sequential()

model_dig.add(LSTM(128, return_sequences=True, activation='relu', input_shape=(30,63)))
model_dig.add(LSTM(64, return_sequences=True, activation='relu'))
model_dig.add(LSTM(64, return_sequences=False, activation='relu'))
model_dig.add(Dense(64, activation='relu'))
model_dig.add(Dense(32, activation='relu'))
model_dig.add(Dense(actions_dig.shape[0], activation='softmax'))


model_dig.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model_dig.fit(X_dig_train, y_dig_train, validation_split=0.2, epochs=200, callbacks=[tb_callback_dig, early_stopping, model_checkpoint])
model_dig.summary()


best_model_dig = load_model('best_model_dig_with10.h5')


yhat_dig = best_model_dig.predict(X_dig_test)
ytrue_dig = np.argmax(y_dig_test, axis=1).tolist()
yhat_dig = np.argmax(yhat_dig, axis=1).tolist()
print(accuracy_score(ytrue_dig, yhat_dig))




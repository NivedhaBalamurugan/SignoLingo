import os
import numpy as np
from preprocess_alpha import *

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout 
from keras.callbacks import TensorBoard , EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from keras.models import load_model
from keras.regularizers import l2
import datetime


actions_alpha = np.array(['A','B','C','D','E','F','G','H','I','K','L'])


tb_callback_alpha = TensorBoard(log_dir="log_alpha/fit/"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

model_alpha = Sequential()

model_alpha.add(LSTM(128, return_sequences=True, activation='relu', input_shape=(30,63)))
model_alpha.add(LSTM(64, return_sequences=True, activation='relu'))
model_alpha.add(LSTM(64, return_sequences=False, activation='relu'))
model_alpha.add(Dense(64, activation='relu')) 
model_alpha.add(Dropout(0.5))
model_alpha.add(Dense(32, activation='relu')) 
model_alpha.add(Dropout(0.5))
model_alpha.add(Dense(actions_alpha.shape[0], activation='softmax'))


model_alpha.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model_alpha.fit(X_alpha_train, y_alpha_train, batch_size=64, validation_split=0.2, epochs=200, callbacks=[tb_callback_alpha])
model_alpha.summary()


model_alpha.save('m1.h5')

yhat_alpha = model_alpha.predict(X_alpha_test)
ytrue_alpha = np.argmax(y_alpha_test, axis=1).tolist()
yhat_alpha = np.argmax(yhat_alpha, axis=1).tolist()
print(accuracy_score(ytrue_alpha, yhat_alpha))




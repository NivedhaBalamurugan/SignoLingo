import os
import numpy as np
from preprocess import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from keras.layers import Dropout


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


log_dir_alpha = os.path.join('Logs_alpha')
tb_callback_alpha = TensorBoard(log_dir=log_dir_alpha)
model_alpha = Sequential()

model_alpha.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,63)))
model_alpha.add(Dropout(0.2))
model_alpha.add(LSTM(128, return_sequences=True, activation='relu'))
model_alpha.add(Dropout(0.2))
model_alpha.add(LSTM(64, return_sequences=False, activation='relu'))
model_alpha.add(Dropout(0.2))
model_alpha.add(Dense(64, activation='relu'))
model_alpha.add(Dropout(0.2))
model_alpha.add(Dense(32, activation='relu'))
model_alpha.add(Dropout(0.2))
model_alpha.add(Dense(actions_alpha.shape[0], activation='softmax'))


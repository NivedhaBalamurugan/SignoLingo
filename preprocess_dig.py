import numpy as np
import os
from sklearn.model_selection import train_test_split


from keras.utils import to_categorical 
import data_aug



actions_dig = np.array(['1','2','3','4','5','6','7','8','9','0','10'])


no_sequences=20
sequence_length=30
DATA_PATH = os.path.join('Dataset')


label_map_dig = {label:num for num, label in enumerate(actions_dig)}
print(label_map_dig)


sequences, labels = [], []
for action in actions_dig:
    for sequence in range(no_sequences):
        window = []
        rotatedpos_window = []
        rotatedneg_window = []
        flipped_window = []
        flipped_rotpos_window = []
        flipped_rotneg_window = []
        translated_window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
            rotatedpos_window, rotatedneg_window, flipped_window, flipped_rotpos_window,flipped_rotneg_window,translated_window = data_aug.augment_landmarks(res, rotatedneg_window, rotatedpos_window, flipped_window,flipped_rotpos_window,flipped_rotneg_window,translated_window)
        sequences.append(window)
        labels.append(label_map_dig[action])
        sequences.append(rotatedpos_window)
        labels.append(label_map_dig[action])
        sequences.append(rotatedneg_window)
        labels.append(label_map_dig[action])
        sequences.append(flipped_window)
        labels.append(label_map_dig[action])
      #  sequences.append(flipped_rotpos_window)
       # labels.append(label_map_dig[action])
       # sequences.append(flipped_rotneg_window)
       # labels.append(label_map_dig[action])
        sequences.append(translated_window)
        labels.append(label_map_dig[action])  




X_dig = np.array(sequences)
X_dig = np.reshape(X_dig, (X_dig.shape[0] , X_dig.shape[1], -1))
y_dig = np.array(labels)
y_dig = to_categorical(labels).astype(int)
X_dig_train, X_dig_test, y_dig_train, y_dig_test = train_test_split(X_dig, y_dig, test_size=0.2)


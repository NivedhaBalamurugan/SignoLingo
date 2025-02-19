"""import os
import numpy as np

actions_alp = np.array(['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'])
actions_dig = np.array(['1','2','3','4','5','6','7','8','9','0','10'])
no_sequences=20
sequence_length=30
DATA_PATH = os.path.join('Dataset') 


for action in actions_alp: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass
            
for action in actions_dig: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
             pass
         



for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, 'J', str(sequence)))
        except:
            pass
            
"""
import os


for frame in range(12,60):
    if frame % 2 == 0:  
        new_frame = frame // 2
        file_path = os.path.join(os.getcwd(), 'Dataset', 'Z', '0', str(frame) + '.npy')
        new_filepath = os.path.join(os.getcwd(),'Dataset', 'Z', '0', str(new_frame) + '.npy')
        os.rename(file_path, new_filepath)
       
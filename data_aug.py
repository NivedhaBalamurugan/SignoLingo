import numpy as np



def translate_landmarks(landmarks, tx, ty):
    landmarks = landmarks.reshape((-1, 3))
    translated_landmarks = np.copy(landmarks)
    translated_landmarks[:, 0] += tx  
    translated_landmarks[:, 1] += ty  
    return translated_landmarks.flatten()


def rotate_landmarks(landmarks, angle):
    landmarks = landmarks.reshape((-1, 3))
    angle_rad = np.radians(angle)
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                                [np.sin(angle_rad), np.cos(angle_rad), 0],
                                [0, 0, 1]])
    rotated_landmarks = np.dot(landmarks, rotation_matrix)
    return rotated_landmarks.flatten()  



def flip_landmarks(landmarks):
    landmarks = landmarks.reshape((-1, 3))
    flipped_landmarks = np.copy(landmarks)
    flipped_landmarks[:, 0] = -flipped_landmarks[:, 0]
    return flipped_landmarks.flatten()  


def augment_landmarks(landmarks,rotatedneg_window, rotatedpos_window, flipped_window,translated_window):

    rotated_landmarks1 = rotate_landmarks(landmarks, 30)
    rotatedpos_window.append(rotated_landmarks1)

    rotated_landmarks2 = rotate_landmarks(landmarks, -30)
    rotatedneg_window.append(rotated_landmarks2)

    flipped_landmarks = flip_landmarks(landmarks)
    flipped_window.append(flipped_landmarks)

    translated_landmarks1 = translate_landmarks(landmarks, 1, 1)  
    translated_window.append(translated_landmarks1)

    return rotatedpos_window, rotatedneg_window, flipped_window, translated_window

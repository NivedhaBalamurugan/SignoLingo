import cv2
import mediapipe as mp
import numpy as np
import os
from keras.models import load_model


actions_alpha = np.array(['A','B','C','D','E','F','G','H','I','K','L','M','O','P','R','S','T','U','V','W','X','Y'])


try:
    model_alpha = load_model('alpha_model4')
    print(model_alpha.summary())
except Exception as e:
    print("Error loading model:", e)


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands



def mp_detection(image):
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image,results


def draw_boundarybox(hand_landmarks,image):
    landmarks_x = [lm.x for lm in hand_landmarks.landmark]
    landmarks_y = [lm.y for lm in hand_landmarks.landmark]

    min_x = min(landmarks_x) * image.shape[1] - 20 
    max_x = max(landmarks_x) * image.shape[1] + 20  
    min_y = min(landmarks_y) * image.shape[0] - 20  
    max_y = max(landmarks_y) * image.shape[0] + 20  
    
    cv2.rectangle(image, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (255, 0, 0), 2)
    


def draw_landmarks(image):
   mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
   

def extract_key_points(results):
    all_landmarks= []

    for hand_landmarks in results.multi_hand_landmarks:
            single_landmark = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark]).flatten()
            all_landmarks.append(single_landmark)
            
    all_landmarks = np.concatenate(all_landmarks)

    return all_landmarks


DATA_PATH = os.path.join('Dataset') 

sequence = []


cap = cv2.VideoCapture(0)
with mp_hands.Hands(max_num_hands=1, model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

    while cap.isOpened():

        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame...")
            continue

        image,results = mp_detection(frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                draw_boundarybox(hand_landmarks,image)
                draw_landmarks(image)

        keypoints = extract_key_points(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        
        if len(sequence) == 30:
            res = model_alpha.predict(np.expand_dims(sequence,axis=0))[0]
            ans = actions_alpha[np.argmax(res)]
            cv2.putText(image,ans , (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
        

        cv2.imshow('Sign Langauge',image)   

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()                    



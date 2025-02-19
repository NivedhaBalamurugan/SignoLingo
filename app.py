import base64
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import mediapipe as mp
import cv2
from keras.models import load_model


actions_dig = np.array(['1','2','3','4','5','6','7','8','9','0','10'])
actions_alpha = np.array(['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'])



mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands



def decode_base64_image(image_data):
    encoded_data = image_data.split(',')[1]  
    decoded_data = base64.b64decode(encoded_data)
    np_data = np.frombuffer(decoded_data, np.uint8)
    img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
    return img



def mp_detection(image,hands):

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image,results


def extract_key_points(results):
    all_landmarks= []

    for hand_landmarks in results.multi_hand_landmarks:
            single_landmark = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark]).flatten()
            all_landmarks.append(single_landmark)
            
    all_landmarks = np.concatenate(all_landmarks)

    return all_landmarks




app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000","https://signolingo.vercel.app"]}})

try:
    model_dig = load_model('best_model_dig_with10.h5')
    model_alpha = load_model('best_model_alpha_withjz.h5')
    print("Loaded the model")
except Exception as e:
    print("Error loading model:", e)



@app.route('/')
def home():
    return render_template('index.html')





@app.route('/predictdig', methods=['POST'])
def predictdig():
    data = request.json
    frames = data.get('frames', [])
    print("got frames", len(frames))
    
    with mp_hands.Hands(max_num_hands=1, model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        
        sequence = []  
        rt=0
        lt=0

        for frame_data in frames:
            frame = decode_base64_image(frame_data)
            image, results = mp_detection(frame,hands)
            if not results.multi_hand_landmarks:
                print("answer - no")
                return jsonify({'prediction' : ' '})
            keypoints = extract_key_points(results)
            for idx, hand_handedness in enumerate(results.multi_handedness):
                handtype=hand_handedness.classification[0].label
                if(handtype  == 'Left'):
                    lt=lt+1
                else:
                    rt=rt+1 
            sequence.append(keypoints)  

        
        if(rt == 30):
            sequence.clear()
            for frame_data in frames:
            
                frame = decode_base64_image(frame_data)
                frame = cv2.flip(frame,1)
                image, results = mp_detection(frame,hands)
                if not results.multi_hand_landmarks:
                    print("answer - no")
                    return jsonify({'prediction' : ' '})
                keypoints = extract_key_points(results)
                sequence.append(keypoints)  

        print("hell")

        res = model_dig.predict(np.expand_dims(sequence, axis=0))[0]
        ans = actions_dig[np.argmax(res)]

    print("answer" , ans)
    return jsonify({'prediction': ans})       




@app.route('/predictalpha', methods=['POST'])
def predictalpha():
    data = request.json
    frames = data.get('frames', [])

    
    with mp_hands.Hands(max_num_hands=1, model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        
        sequence = []  
        rt=0
        lt=0

        for frame_data in frames:
            
            frame = decode_base64_image(frame_data)
            image, results = mp_detection(frame,hands)
            if not results.multi_hand_landmarks:
                print("answer - no")
                return jsonify({'prediction' : ' '})
            keypoints = extract_key_points(results)
            for idx, hand_handedness in enumerate(results.multi_handedness):
                handtype=hand_handedness.classification[0].label
                if(handtype  == 'Left'):
                    lt=lt+1
                else:
                    rt=rt+1 
            sequence.append(keypoints)

        
        if(rt == 30):
            sequence.clear()
            for frame_data in frames:
            
                frame = decode_base64_image(frame_data)
                frame = cv2.flip(frame,1)
                image, results = mp_detection(frame,hands)
                if not results.multi_hand_landmarks:
                    print("answer - no")
                    return jsonify({'prediction' : ' '})
                keypoints = extract_key_points(results)
                sequence.append(keypoints)  


        res = model_alpha.predict(np.expand_dims(sequence, axis=0))[0]
        ans = actions_alpha[np.argmax(res)]

    print("answer" , ans)
    return jsonify({'prediction': ans})       
  



if __name__ == '__main__':
    app.run(debug=True )



@app.route('/predictalpha', methods=['POST'])
def predict():
    data = request.json
    frames = data.get('frames', [])

    actions_alpha = np.array(['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y'])

    mp_hands = mp.solutions.hands


    def decode_base64_image(image_data):
        encoded_data = image_data.split(',')[1]  
        decoded_data = base64.b64decode(encoded_data)
        np_data = np.frombuffer(decoded_data, np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        return img



    def mp_detection(image):

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
    

    
    with mp_hands.Hands(max_num_hands=1, model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        

        sequence = []  
        
        for frame_data in frames:
          
            frame = decode_base64_image(frame_data)
          
            image, results = mp_detection(frame)

            if not results.multi_hand_landmarks:
                return jsonify({'prediction' : ' '})
          
            keypoints = extract_key_points(results)
            sequence.append(keypoints)
            

        res = model_alpha.predict(np.expand_dims(sequence, axis=0))[0]
        ans = actions_alpha[np.argmax(res)]

    print("answer" , ans)
    
    return jsonify({'prediction': ans})





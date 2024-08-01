import cv2
import pandas as pd
import numpy as np
import mediapipe as mp
import pickle

with open('model/pose_detection.pkl','rb') as f:
    model = pickle.load(f)



mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor feed
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        # Make detections
        results = holistic.process(image)

        image.flags.writeable=True
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

        # 1. Draw face landmarks
        mp_drawing.draw_landmarks(image,results.face_landmarks,mp_holistic.FACEMESH_TESSELATION,
                              mp_drawing.DrawingSpec(color=(255,0,0),thickness=1,circle_radius=1),
                              mp_drawing.DrawingSpec(color=(240,0,0),thickness=1,circle_radius=1))

        # 2. Left hand landmarks
        mp_drawing.draw_landmarks(image,results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(255,0,0),thickness=2,circle_radius=2),
                              mp_drawing.DrawingSpec(color=(240,0,0),thickness=2,circle_radius=2))

        # 3. Right hand landmarks
        mp_drawing.draw_landmarks(image,results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(255,0,0),thickness=2,circle_radius=2),
                              mp_drawing.DrawingSpec(color=(240,0,0),thickness=2,circle_radius=2))

        # 4. Pose detection
        mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(255,0,0),thickness=2,circle_radius=2),
                              mp_drawing.DrawingSpec(color=(240,0,0),thickness=2,circle_radius=2))
        
        # Export coordinates
        try:
            # Extract pose landmarks
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x,landmark.y,landmark.z,landmark.visibility] for landmark in pose]).flatten())

            # Extract face landmarks
            face = results.face_landmarks.landmark
            face_row = list(np.array([[landmark.x,landmark.y,landmark.z,landmark.visibility] for landmark in face]).flatten())

            # # Concat rows
            row = pose_row+face_row

            # # Insert row
            # row.insert(0,class_name)

            # # Export to csv
            # with open('coords.csv',mode='a',newline='') as f:
            #     csv_writer = csv.writer(f,delimiter=",",quotechar="'",quoting=csv.QUOTE_MINIMAL)
            #     csv_writer.writerow(row)

            # Make detections
            X = pd.DataFrame([row])
            body_language_class = model.predict(X)[0]
            body_language_pred = model.predict_proba(X)[0]
            print(body_language_class,body_language_pred)

            # Grab ear coords
            coords = tuple(np.multiply(np.array((results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x,
                                                results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y)
                                                ),[640,480]).astype(int))
            
            cv2.rectangle(image,
                         (coords[0],coords[1]+5),
                         (coords[0]+len(body_language_class)*20,coords[1]-30),
                         (245,117,16),-1)
            cv2.putText(image,body_language_class,coords,
                        cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
            
            # Display class
            cv2.putText(image,"CLASS",
                        (95,12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
            cv2.putText(image,body_language_class.split(" ")[0],
                        (90,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
            
            # Display prob
            cv2.putText(image,"PROB",
                        (15,12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
            cv2.putText(image,str(round(body_language_pred[np.argmax(body_language_pred)],2)),
                        (10,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)

        except Exception as e:
            print(e)

        cv2.imshow("Holistic model detection",image)

        if cv2.waitKey(10) & 0xFF==ord('q'):
            break
cap.release()
cv2.destroyAllWindows()            
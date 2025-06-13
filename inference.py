import cv2
import mediapipe as mp
import pickle
import numpy as np
import json

model_dict=pickle.load(open('./model.p', 'rb'))
model=model_dict['model']

with open("dataset_labels.json", "r") as file:
    labels = json.load(file)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

cap=cv2.VideoCapture(0)



while True:
    ret,frame=cap.read()

    data_aux=[]
    x_=[]
    y_=[]

    H,W,_=frame.shape

    results=hands.process(frame)
    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmark,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        for hand_landmark in results.multi_hand_landmarks:
            for i in range(len(list(hand_landmark.landmark))):
                x=hand_landmark.landmark[i].x
                y=hand_landmark.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(list(hand_landmark.landmark))):
                x = hand_landmark.landmark[i].x
                y = hand_landmark.landmark[i].y

                data_aux.append(x-min(x_))
                data_aux.append(y-min(y_))


        x1=int(min(x_)*W)-10
        y1=int(min(y_)*H)-10
        x2=int(max(x_)*W)-10
        y2=int(max(y_)*H)-10

        prediction=model.predict([np.asarray(data_aux)])
        print(prediction)

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,125),3)
        cv2.putText(frame,labels[f"{prediction[0]}"],(x1-10,y1-10),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),3)

    cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xFF==8:
        break

cap.release()
cv2.destroyAllWindows()

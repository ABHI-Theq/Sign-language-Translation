import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands=mp_hands.Hands(static_image_mode=True,min_detection_confidence=0.3)


DATADIR="./data"
data=[]
label=[]
for idx,dir in enumerate(os.listdir(DATADIR)):
    for img_path in os.listdir(os.path.join(DATADIR,dir)):
        img=cv2.imread(os.path.join(DATADIR,dir,img_path))
        data_aux=[]
        x_=[]
        y_=[]

        img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        results=hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                for i in range(len(list(hand_landmark.landmark))):
                    x=hand_landmark.landmark[i].x
                    y=hand_landmark.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(list(hand_landmark.landmark))):
                    x=hand_landmark.landmark[i].x
                    y=hand_landmark.landmark[i].y

                    data_aux.append(x-min(x_))
                    data_aux.append(y-min(y_))

            data.append(data_aux)
            label.append(idx)

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': label}, f)
f.close()




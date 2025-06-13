import cv2
import json
import os

with open("dataset_labels.json", "r") as file:
    label_mapping = json.load(file)

# print(label_mapping["0"])

DATADIR="./data"
if not os.path.exists(DATADIR):
    os.mkdir(DATADIR)


cap=cv2.VideoCapture(0)
dataset_size = 200
for i in range(5):
    if not os.path.exists(os.path.join(DATADIR,label_mapping[f"{i}"])):
        os.makedirs(os.path.join(DATADIR,label_mapping[f"{i}"]))
    counter=100
    while True:
        ret,frame=cap.read()

        cv2.putText(frame,f"Ready? Press s to start {i}",(100,50),cv2.FONT_HERSHEY_PLAIN,1.3,(0,255,0))
        cv2.imshow("frame",frame)

        if cv2.waitKey(25) & 0xFF==ord("s"):
            break

    while counter<dataset_size:
        ret,frame=cap.read()
        cv2.imshow("frame",frame)
        cv2.waitKey(30)
        cv2.imwrite(os.path.join(DATADIR,label_mapping[f"{i}"],f"img_{counter}.jpg"),frame)
        counter+=1

cap.release()
cv2.destroyAllWindows()



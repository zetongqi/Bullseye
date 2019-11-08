import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN
import tensorflow as tf
import PIL
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
classifier_path = os.path.join(dir_path, "haarcascade_frontalface_default.xml")
model_path = os.path.join(dir_path, "pittqis.h5")
classifier = cv2.CascadeClassifier(classifier_path)
model = tf.keras.models.load_model(model_path)

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
detector = MTCNN()
names = ["P1", "P2", "P3"]

def process_frame(frame):
    frame1 = np.array([frame[:, :, -1], frame[:, :, 1], frame[:, :, 0]])
    frame1 = np.transpose(frame1, (1, 2, 0))
    return frame1


while(True):
    ret, frame = cap.read()
    frame1 = process_frame(frame)
    results = detector.detect_faces(frame1)
    if len(results) == 0:
        print("no faces detected")

    # run infernce and put bounding boxes for each face detected
    for i in range(len(results)):
        x1, y1, width, height = results[i]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = frame1[y1:y2, x1:x2]

        image = PIL.Image.fromarray(face)
        image = image.resize((160, 160))
        face_array = np.asarray(image)
        mean, std = face_array.mean(), face_array.std()
        face_array = (face_array - mean) / std

        y_hat = list(model.predict(np.expand_dims(face_array, axis=0))[0])
        print(len(results), "faces detected")
        print(y_hat)
        idx = y_hat.index(max(y_hat))
        print(results)

        if idx == 0:
            left_eye_x, left_eye_y = results[i]["keypoints"]["left_eye"]
            right_eye_x, right_eye_y = results[i]["keypoints"]["right_eye"]
            center_x = int((left_eye_x + right_eye_x) / 2)
            center_y = int((left_eye_y + right_eye_y) / 2)
            vertical_offset = int(abs(results[i]["keypoints"]["nose"][-1] - center_y))
            cv2.circle(frame,(center_x,center_y-vertical_offset), 5, (0,0,255), -1)

        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
        cv2.putText(frame, names[idx], (x1+5,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (153,255,51), 2)  

    cv2.imshow('video', frame)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
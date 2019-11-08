import numpy as np
import cv2

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
classifier = cv2.CascadeClassifier('/Users/zetong/haarcascade_frontalface_default.xml')

i = 0
j = 0
c = 5
while(True):
    ret, frame = cap.read()
    cv2.circle(frame,(i,j), c, (0,0,255), -1)
    cv2.imshow('video', frame)

    print("(", i, ",", j, ")", sep="")

    k = cv2.waitKey(30) & 0xff

    # w
    if k == 119:
        j -= c
    # s
    if k == 115:
        j += c
    # a
    if k == 97:
        i -= c
    # d
    if k == 100:
        i += c
    # ESC
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
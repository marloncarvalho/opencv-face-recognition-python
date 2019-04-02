# coding: utf-8

import cv2
import os
import numpy as np

def detect_face(img):
    """
    Detecta apenas UMA face na imagem fornecida, retornando o quadrado onde está a face.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    
    if (len(faces) == 0):
        print("No faces detected in the provided image")
        return None, None
    
    (x, y, w, h) = faces[0]
    
    return gray[y:y+w, x:x+h], faces[0]

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

def predict(test_img):
    img = test_img.copy()
    face, rect = detect_face(img)

    if face is None:
        print("Can't predict. No faces found in the test image provided.")
    else:
        label, confidence = face_recognizer.predict(face)
        label_text = subjects[label]

        if confidence == 0:
            print('Totally confident that the face detected in the test image is ' + label_text + '.' )
        else:
            print(str(confidence) + " (the higher, the worse is the confidence) confident that the face detected in the test image is " + label_text + ".")
        
        draw_rectangle(img, rect)
        draw_text(img, label_text, rect[0], rect[1]-5)
    
    return img

subjects = ["", "Ramiz Raja", "Elvis Presley", "Valentina"]

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('model.xml')

print("Predicting images...")

test_img1 = cv2.imread("test-data/test1.jpg")
test_img2 = cv2.imread("test-data/test2.jpg")
test_img3 = cv2.imread("test-data/test3.jpg")

predicted_img1 = predict(test_img1)
predicted_img2 = predict(test_img2)
predicted_img3 = predict(test_img3)
print("Prediction complete")

cv2.imshow(subjects[1], cv2.resize(predicted_img1, (400, 500)))
cv2.imshow(subjects[2], cv2.resize(predicted_img2, (400, 500)))
cv2.imshow(subjects[3], cv2.resize(predicted_img3, (400, 500)))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()
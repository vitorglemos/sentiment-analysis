import cv2
import numpy as np
import pandas as pd


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

import zipfile
%tensorflow_version 2.x

import tensorflow
tensorflow.__version__

if True:
    image = cv2.imread("photo.png")

    path_cascade_faces = './haarcascade_frontalface_default.xml'
    path_model = 'mv1.h5'
    face_detection = cv2.CascadeClassifier(path_cascade_faces)

    classifier_model = load_model(path_model, compile=False)
    labels = []

face_detection = cv2.CascadeClassifier(path_cascade_faces)
classifier_model = load_model(path_model, compile=False)

labels = ["Raiva", "Nojo", "Medo", "Feliz", "Triste", "Surpreso", "Neutro"]
 
original = imagem.copy()
faces = face_detection.detectMultiScale(original, scaleFactor=1.1, minNeighbors=3, minSize=(20,20))
gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
 
if len(faces) > 0:
    for (fX, fY, fW, fH) in faces:
      roi = gray[fY:fY + fH, fX:fX + fW]
      roi = cv2.resize(roi, (48, 48))
      roi = roi.astype("float") / 255.0
      roi = img_to_array(roi)
      roi = np.expand_dims(roi, axis=0)
      preds = classifier_model.predict(roi)[0]
      
      print(f" Predictions: {preds}")
      
      emotion_probability = np.max(preds)
      label = labels
      oes[preds.argmax()]
      cv2.putText(original, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
      cv2.rectangle(original, (fX, fY), (fX + fW, fY + fH),(0, 0, 255), 2)

 
cv2_imshow(original)

probabilities = np.ones((250, 300, 3), dtype="uint8") * 255

if len(faces) == 1:
  for (i, (emotion, prob)) in enumerate(zip(expression, preds)):
      text = "{}: {:.2f}%".format(emotion, prob * 100)
      w = int(prob * 300)
      cv2.rectangle(probabilities, (7, (i * 35) + 5),
                   (w, (i * 35) + 35), (200, 250, 20), -1)
      cv2.putText(probabilities, text, (10, (i * 35) + 23),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

  cv2_imshow(probabilities)

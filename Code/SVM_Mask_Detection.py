import cv2
import numpy as np
from sklearn.svm import SVC


svm_model = SVC(kernel='linear', C=1.0)
svm_model.load('svm_model.pkl')  


cap = cv2.VideoCapture(0)  

while True:
    ret, frame = cap.read()

    if not ret:
        break

    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

   
    for (x, y, w, h) in faces:
       
        face = gray_frame[y:y+h, x:x+w]
        
        
        resized_face = cv2.resize(face, (150, 150))
        flattened_face = resized_face.reshape(1, -1) / 255.0
        
        
        prediction_probability = svm_model.predict_proba(flattened_face)[0][1]  # Probability of class 1 (mask)
        
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f'Mask Probability: {prediction_probability:.2f}'
        cv2.putText(frame, text, (x, y-10), font, 0.5, (0, 255, 0), 2)
    
    
    cv2.imshow('Face Mask Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

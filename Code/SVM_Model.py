import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.externals import joblib  

with_mask_dir = "path_to_with_mask_images"
without_mask_dir = "path_to_without_mask_images"


data = []
labels = []

for image_file in os.listdir(with_mask_dir):
    image = cv2.imread(os.path.join(with_mask_dir, image_file), cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(image, (150, 150))
    flattened_image = resized_image.reshape(-1)
    data.append(flattened_image)
    labels.append(1) 

for image_file in os.listdir(without_mask_dir):
    image = cv2.imread(os.path.join(without_mask_dir, image_file), cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(image, (150, 150))
    flattened_image = resized_image.reshape(-1)
    data.append(flattened_image)
    labels.append(0)  

data = np.array(data, dtype="float32")
labels = np.array(labels)


(train_data, test_data, train_labels, test_labels) = train_test_split(data, labels, test_size=0.2, random_state=42)


svm_model = SVC(kernel="linear", C=1.0)
svm_model.fit(train_data, train_labels)

predictions = svm_model.predict(test_data)
print(classification_report(test_labels, predictions))

# Saving the trained model
joblib.dump(svm_model, "svm_model.pkl") 

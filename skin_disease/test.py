import cv2
import numpy as np
import os
import glob
import mahotas as mt
from sklearn.svm import LinearSVC
from train import extract_features

train_features = np.load("test3_bb.npy")
train_labels = np.load("test3_b.npy")





# have a look at the size of our feature vector and labels
print ("Training features: {}".format(np.array(train_features).shape))
print ("Training labels: {}".format(np.array(train_labels).shape))

print ("[STATUS] Creating the classifier..")
clf_svm = LinearSVC(random_state=9)

# fit the training data and labels
print ("[STATUS] Fitting data/label to model..")
clf_svm.fit(train_features, train_labels)

#test_path = "dataset/test"
#for file in glob.glob(test_path + "/*.jpg"):
# read the input image
image = cv2.imread('b.jpg')

# convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# extract haralick texture from the image
features = extract_features(gray)

# evaluate the model and predict label
prediction = clf_svm.predict(features.reshape(1, -1))[0]

# show the label
cv2.putText(image, prediction, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)
print ("Prediction - {}".format(prediction))

# display the output image
cv2.imshow("Test_Image", image)
cv2.waitKey(0)

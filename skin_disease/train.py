import cv2
import numpy as np
import os
import glob
import mahotas as mt
from sklearn.svm import LinearSVC

# function to extract haralick textures from an image
def extract_features(image):
	# calculate haralick texture features for 4 types of adjacency
	textures = mt.features.haralick(image)

	# take the mean of it and return it
	ht_mean  = textures.mean(axis=0)
	return ht_mean

# load the training dataset
train_path  = "C:/Users/SHINICHI/skin_disease/train"
train_names = os.listdir(train_path)

# empty list to hold feature vectors and train labels
train_features = []
train_labels   = []

# loop over the training dataset
print ("[STATUS] Started extracting haralick textures..")
for train_name in train_names:
	cur_path = train_path + "/" + train_name
	cur_label = train_name
	i = 1

	for file in glob.glob(cur_path + "/*.jpg"):
		print ("Processing Image - {} in {}".format(i, cur_label))
		# read the training image
		image = cv2.imread(file)

		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# extract haralick texture from the image
		features = extract_features(gray)

		# append the feature vector and label
		train_features.append(features)
		train_labels.append(cur_label)

		# show loop update
		i += 1

# have a look at the size of our feature vector and labels
print ("Training features: {}".format(np.array(train_features).shape))
print ("Training labels: {}".format(np.array(train_labels).shape))

with open('C:/Users/SHINICHI/skin_disease/data_bbb.txt', 'w') as file_handler:
    for item in train_features:
        file_handler.write("{}\n".format(item))
np.save('test3_bb',train_features)

with open('C:/Users/SHINICHI/skin_disease/data_b.txt', 'w') as file_handler:
    for item in train_labels:
        file_handler.write("{}\n".format(item))
np.save('test3_b',train_labels)

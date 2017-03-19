import numpy as np
from PIL import Image
import os, sys
from sklearn.cross_validation import train_test_split
import pickle

dir1 = "/Users/Dhanush/Documents/Deeplearn/MathematicalSymbolRecognition/extracted_images"

#Get all Directories, all 82 labels.
label_dict = os.listdir(dir1)


def load_image(infilename):
    img = Image.open( infilename)
    img.load()
    data = np.asarray( img, dtype="float32" )
    return data

X = []
y = []
total_labels = 0
for label in label_dict:
	if label == ".DS_Store":
		continue
	dir2 = dir1 + "/" + str(label)
	temp_dict = os.listdir(dir2)
	count_per_label = 0
	total_labels +=1
	for item in temp_dict:
		y.append(label)
		count_per_label+=1
		img = dir1 + "/" + str(label)+"/"+ str(item)
		X.append(load_image(img))
	print label + " " + str(count_per_label)
print "Total labels considered for classification:  " + str(total_labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=436718)
X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=0.20, random_state=436718)
X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)
X_cv = np.asarray(X_train)
y_cv = np.asarray(y_cv)

print "Train"
print X_train.shape
print y_train.shape
print "Test"
print X_test.shape
print y_test.shape
print "CV"
print X_cv.shape
print y_cv.shape

f = open("MathSymbols_train_test", "wb")
pickle.dump((X_train, y_train, X_test, y_test,X_cv,y_cv),f)

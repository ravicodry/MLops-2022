"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
# Standard scientific Python imports
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import sklearn
from skimage.transform import rescale, resize
from skimage import transform

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from utils import preprocess_digits,train_dev_test_split

#hyperparameter tuning
Gamma_list=[0.01 ,0.001, 0.0001, 0.0005]
c_list=[.1 ,.5, .4, 10, 5, 1]

h_param_comb=[{'gamma':g,'C':c} for g in Gamma_list for c in c_list]

###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.

digits = datasets.load_digits()
#_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
#for ax, image, label in zip(axes, digits.images, digits.target):
    #ax.set_axis_off()
    #ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    #ax.set_title("Training: %i" % label)

###############################################################################
# Classification
# --------------
#
# To apply a classifier on this data, we need to flatten the images, turning
# each 2-D array of grayscale values from shape ``(8, 8)`` into shape
# ``(64,)``. Subsequently, the entire dataset will be of shape
# ``(n_samples, n_features)``, where ``n_samples`` is the number of images and
# ``n_features`` is the total number of pixels in each image.
#
# We can then split the data into train and test subsets and fit a support
# vector classifier on the train samples. The fitted classifier can
# subsequently be used to predict the value of the digit for the samples
# in the test subset.

# flatten the images


import numpy as np 
def resize_a(image,n):
    image = resize(image, (image.shape[0] // n, image.shape[1] // n),anti_aliasing=True)
    return image
def resize_b(image,n):
    image = resize(image, (image.shape[0]*n, image.shape[1]*n),anti_aliasing=True)
    return image


digits_4 = np.zeros((1797, 2, 2))  # image divide by 4
digits_2 = np.zeros((1797, 4, 4))  # image divide by 2
digits_5 = np.zeros((1797, 16, 16))  # image divide by 5

for i in range(0,1797):
    digits_4[i] = resize_a(digits.images[i],4)

for i in range(0,1797):
    digits_2[i] = resize_a(digits.images[i],2)

for i in range(0,1797):
    digits_5[i] = resize_b(digits.images[i],2)
    



n_samples = len(digits_5)
data = digits_5.reshape((n_samples, -1))
print("imagesize in the digits dataset.")
print(digits_5[-1].shape)

    
train_frac=0.8
test_frac=0.1
dev_frac=0.1

assert train_frac+dev_frac+test_frac==1
#print(data.shape)
# Split data into 80% train,10% validate and 10% test subsets
dev_test_frac=1-train_frac

X_train, X_dev_test, y_train, y_dev_test = train_test_split(data ,digits.target, test_size=dev_test_frac, shuffle=True,random_state=42)
X_test, X_dev, y_test, y_dev = train_test_split(X_dev_test, y_dev_test, test_size=(dev_frac)/dev_test_frac, shuffle=True,random_state=42)

best_acc=-1
best_model=None
best_h_params=None 
for com_hyper in h_param_comb:

	# Create a classifier: a support vector classifier
	clf = svm.SVC()

	#Setting hyperparameter
	hyper_params=com_hyper
	clf.set_params(**hyper_params)
	#print(com_hyper)

	# Learn the digits on the train subset
	clf.fit(X_train, y_train)

	# Predict the value of the digit on the test subset
	predicted_train = clf.predict(X_train)
	predicted_dev = clf.predict(X_dev)
	predicted_test = clf.predict(X_test)
	
	#print("shape : ",predicted_dev.shape)
	
	cur_acc_train=metrics.accuracy_score(y_pred=predicted_train,y_true=y_train)
	cur_acc_dev=metrics.accuracy_score(y_pred=predicted_dev,y_true=y_dev)
	cur_acc_test=metrics.accuracy_score(y_pred=predicted_test,y_true=y_test)
	
	
	if cur_acc_dev>best_acc:
	     best_acc=cur_acc_dev
	     best_model=clf
	     best_h_params=com_hyper
	     print("found new best acc with: "+str(com_hyper))
	     print("New best accuracy:"+ " train" + "  "+str(cur_acc_train)+ " "+ "dev" + " "+str(cur_acc_dev)+ " "+ "test" + " " +str(cur_acc_test))
	     
predicted = best_model.predict(X_test)
	###############################################################################
	# Below we visualize the first 4 test samples and show their predicted
	# digit value in the title.

	#_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
	#for ax, image, prediction in zip(axes, X_test, predicted):
	    #ax.set_axis_off()
	    #image = image.reshape(8, 8)
	    #ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
	    #ax.set_title(f"Prediction: {prediction}")

	###############################################################################
	# :func:`~sklearn.metrics.classification_report` builds a text report showing
	# the main classification metrics.

#print(f"Classification report for classifier {best_model}:\n"
#f"{metrics.classification_report(y_test, predicted)}\n"
#)

	###############################################################################
	# We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
	# true digit values and the predicted digit values.

	#disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
	#disp.figure_.suptitle("Confusion Matrix")
	#print(f"Confusion matrix:\n{disp.confusion_matrix}")

#plt.show()
print("Best hyperparameters were: ")
print(com_hyper)
print("Best accuracy on dev: ")
print(best_acc)

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
import pandas as pd
import numpy as np 

# Function to perform Hyperparamter tunning for SVM Classifier


def svm_hyperparameters_tunning(imagedataset) :
    
    n_samples = len(digits.images)
    data = imagedataset.reshape((n_samples, -1))
    train_frac=0.7
    test_frac=0.1
    dev_frac=0.2
    dev_test_frac=1-train_frac
    
    Gamma_list=[0.01 ,0.001, 0.0001, 0.0005]
    c_list=[0.1,0.4,0.5,1.0]
    h_param_comb=[{'gamma':g,'C':c} for g in Gamma_list for c in c_list]
#     digits = datasets.load_digits()

    X_train, X_dev_test, y_train, y_dev_test = train_test_split(data ,digits.target, test_size=dev_test_frac, shuffle=True,random_state=42)
    X_test, X_dev, y_test, y_dev = train_test_split(X_dev_test, y_dev_test, test_size=(dev_frac)/dev_test_frac, shuffle=True,random_state=42)

    best_acc=-1
    best_model=None
    best_h_params=None
#     result_dataframe = pd.DataFrame(columns=['Gamma', 'C', 'train','dev','test'])
    gamma_list = list()
    c_list = list()
    train_list = list()
    val_list = list()
    test_list = list()
    for com_hyper in h_param_comb:

        # Create a classifier: a support vector classifier
        clf = svm.SVC()	
        hyper_params=com_hyper
        clf.set_params(**hyper_params)	
        clf.fit(X_train, y_train)
        predicted_train = clf.predict(X_train)
        predicted_dev = clf.predict(X_dev)
        predicted_test = clf.predict(X_test)
        cur_acc_train=metrics.accuracy_score(y_pred=predicted_train,y_true=y_train)
        cur_acc_dev=metrics.accuracy_score(y_pred=predicted_dev,y_true=y_dev)
        cur_acc_test=metrics.accuracy_score(y_pred=predicted_test,y_true=y_test)
        gamma_list.append(com_hyper['gamma'])
        c_list.append(com_hyper['C'])
        train_list.append(cur_acc_train)
        val_list.append(cur_acc_dev)
        test_list.append(cur_acc_test)

        if cur_acc_dev>best_acc:
             best_acc=cur_acc_dev
             best_model=clf
             best_h_params=com_hyper

    data = {'Gamma':gamma_list,'C':c_list,'Training_Accuracy':train_list,'Val_Accuracy':val_list,'Test_Accuracy':test_list}
    df = pd.DataFrame(data)
    print("Training Accuracy")
    print("Training Accuracy min " , df['Training_Accuracy'].min())
    print("Training Accuracy mean " , df['Training_Accuracy'].mean())
    print("Training Accuracy max " , df['Training_Accuracy'].max())
    print("Training Accuracy median " , df['Training_Accuracy'].median())
    print("*************************************************************")
    print("Val_Accuracy")
    print("Validation  Accuracy min " , df['Val_Accuracy'].min())
    print("Validation Accuracy mean " , df['Val_Accuracy'].mean())
    print("Validation Accuracy max " , df['Val_Accuracy'].max())
    print("Validation Accuracy median " , df['Val_Accuracy'].median())
    print("****************************************************************")
    print("Test Accuracy")
    print("Test Accuracy min " , df['Test_Accuracy'].min())
    print("Test Accuracy mean " , df['Test_Accuracy'].mean())
    print("Test Accuracy max " , df['Test_Accuracy'].max())
    print("Test Accuracy median " , df['Test_Accuracy'].median())
    print(df)
    predicted = best_model.predict(X_test)
    print("********************************************************************************")
    print("Highest accuracy on val dataset is : {0} and corresponsing hyperparamter are {1}.".format(best_acc,com_hyper))

digits = datasets.load_digits()
def resize_image(image,n):
    image = resize(image, (image.shape[0] // n, image.shape[1] // n),anti_aliasing=True)
    return image
resize_2 = np.zeros((1797, 4, 4))  # change the resolution by 2%
resize_5 = np.zeros((1797, 4, 4))  # change the resolution by 5%
resize_4 = np.zeros((1797, 2, 2))  # change the resolution by 4%

for i in range(0,1797):
    resize_2[i] = resize_image(digits.images[i],2) # Creating a new dataset with new resize resolution
for i in range(0,1797):
    resize_5[i] = resize_image(digits.images[i],5) # Creating a new dataset with new resize resolution
for i in range(0,1797):
    resize_4[i] = resize_image(digits.images[i],4) # Creating a new dataset with new resize resolution
print("Question 1 - part 01 and part 02")
svm_hyperparameters_tunning(digits.images)
print("Question 2 - part 01 ")
print("Size of the dataset is",digits.images.size)
print("Shape of the dataset is",digits.images.shape)
print("Size of the digit image is",digits.images[0].size)
print("Shape of the digit image is",digits.images[0].shape)
print("Question 2 - part 02 ")
# print("shape of the resized image of new dataset",resize_2[0].shape)
svm_hyperparameters_tunning(resize_2)
# print("shape of the resized image of new dataset",resize_4[0].shape)
svm_hyperparameters_tunning(resize_4)
# print("shape of the resized image of new dataset",resize_5[0].shape)
svm_hyperparameters_tunning(resize_5)
print("Thank you")
